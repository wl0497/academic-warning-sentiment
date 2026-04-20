# train_v3.py - Strong regularization: Mixup + LabelSmoothing(0.15) + 8 frozen layers + stronger augmentation
import os, time, random, re, torch, torch.nn as nn, numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from config_v2 import Config
from bert_bilstm_cnn import BertBiLSTMCNN
from utils import load_data, split_data, create_data_loader

# === Synonym dict ===
SYN = {
    '确实':['的确','诚然','实在'],'特别':['格外','尤其','十分'],
    '很':['挺','蛮','相当','十分'],'有点':['有些','略微'],
    '可能':['也许','或许','大概'],'但是':['不过','然而','只是'],
    '帮助':['协助','帮忙','扶持'],'压力':['负担','重担','压迫'],
    '焦虑':['担忧','着急','忧虑'],'学习':['学业','上课','念书'],
    '困难':['艰难','费劲','吃力'],'学校':['院校','学府','大学'],
    '成绩':['分数','绩点','得分'],'课程':['课业','功课'],
    '考试':['测评','考查','测验'],'老师':['教师','导师','教授'],
    '进步':['提升','长进','前进'],'绝望':['失望','悲观'],
    '迷茫':['迷惘','困惑'],'有用':['有效','奏效'],
}
PRE = ['说实话','其实吧','怎么说呢','唉','嗯','说真的','反正','感觉','好像','可能','客观来说','嘛','据说','听说']
SUF = ['吧','呢','哦','哈','嘛','呀','...','啊']
AUX = ['的','了','很','也','都','就','还','确实','感觉','好像','应该','可能','大概','其实','不过','但是','反正']

# === Augmentation functions ===
def sub_syn(text):
    for old, alts in SYN.items():
        if old in text and random.random() < 0.7:
            text = text.replace(old, random.choice(alts), 1)
    return text

def del_word(text, p=0.2):
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    if len(words) < 4: return text
    kept = [w for w in words if random.random() > p]
    if len(kept) < 2: return text
    for w in words:
        if w not in kept: text = re.sub(re.escape(w), '', text, count=1)
    text = re.sub(r'[\s,，]{2,}', '，', text).strip(' ，。')
    return text if len(text) > 5 else text[:50]

def add_prefix(text):
    if random.random() > 0.3: return text
    return random.choice(PRE) + random.choice(['，',' ']) + text

def add_suffix(text):
    if random.random() > 0.3: return text
    return text + random.choice(SUF)

def swap_chars(text, n=2):
    chars = list(text)
    cjk = [i for i,c in enumerate(chars) if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f']
    for _ in range(min(n, len(cjk)-1)):
        if len(cjk) < 2: break
        i = random.randint(0, len(cjk)-2)
        j, k = cjk[i], cjk[min(i+1, len(cjk)-1)]
        if k-j <= 2:
            chars[j], chars[k] = chars[k], chars[j]
            cjk.pop(i)
    return ''.join(chars)

def mix_aug(text):
    ops = [sub_syn, lambda t: del_word(t), add_prefix, add_suffix, lambda t: swap_chars(t)]
    random.shuffle(ops)
    result = text
    for op in ops[:random.randint(2,4)]:
        result = op(result)
    return result

# === Dataset ===
class AugDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, do_aug=True, aug_ratio=0.7):
        self.texts = list(texts); self.labels = list(labels)
        self.tokenizer = tokenizer; self.max_len = max_len
        self.do_aug = do_aug; self.aug_ratio = aug_ratio
        self.cache = {}
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.do_aug and random.random() < self.aug_ratio:
            if idx not in self.cache: self.cache[idx] = mix_aug(text)
            text = self.cache[idx]
        enc = self.tokenizer.encode_plus(text, add_special_tokens=True,
            max_length=self.max_len, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def make_loader(df, tokenizer, max_len, bs, shuffle=True, aug=True, aug_ratio=0.7):
    ds = AugDS(df['text'].values, df['label'].values, tokenizer, max_len, aug, aug_ratio)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)

# === Mixup ===
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# === Label smoothing ===
class LSCE(nn.Module):
    def __init__(self, eps=0.15): super().__init__(); self.eps = eps
    def forward(self, pred, target):
        n = pred.size(-1)
        logp = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            td = torch.zeros_like(pred)
            td.fill_(self.eps / (n - 1))
            td.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        return (-td * logp).sum(dim=-1).mean()

# === Gradient noise ===
def add_grad_noise(model, std=5e-4):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * std)

# === Freeze ===
def freeze_bert(model, n=8):
    fc = 0
    for name, param in model.bert.named_parameters():
        parts = name.split('.')
        if 'encoder' in parts and 'layer' in parts:
            try:
                li = parts.index('layer')
                ln = int(parts[li+1])
                if ln < n: param.requires_grad = False; fc += param.numel()
            except: pass
    print('frozen {} BERT layers ({:,} params)'.format(n, fc))

def train_epoch(model, loader, opt, sched, crit, dev, use_mixup=True):
    model.train()
    tl = 0; correct = 0; total = 0
    for batch in tqdm(loader, desc='Train'):
        ids = batch['input_ids'].to(dev); mask = batch['attention_mask'].to(dev)
        labels = batch['label'].to(dev)
        opt.zero_grad()
        if use_mixup and random.random() < 0.5:
            # Mixup at logit level: mix outputs of model with itself (shuffled batch)
            idx = torch.randperm(ids.size(0)).to(dev)
            lam = np.random.beta(0.4, 0.4)
            logits = model(ids, mask)
            logits_shuffled = model(ids[idx], mask[idx])
            logits_mixed = lam * logits + (1 - lam) * logits_shuffled
            loss = lam * crit(logits, labels) + (1 - lam) * crit(logits_shuffled, labels[idx])
            with torch.no_grad():
                _, preds = torch.max(logits_mixed, dim=1)
                correct += (lam * (preds==labels).float() + (1-lam)*(preds==labels[idx]).float()).sum().item()
        else:
            logits = model(ids, mask)
            loss = crit(logits, labels)
            _, preds = torch.max(logits, dim=1)
            correct += (preds==labels).sum().item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        add_grad_noise(model, std=5e-4)
        opt.step(); sched.step()
        tl += loss.item(); total += labels.size(0)
    return tl/len(loader), correct/total

def eval_epoch(model, loader, crit, dev):
    model.eval(); tl = 0; correct = 0; total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval'):
            ids = batch['input_ids'].to(dev); mask = batch['attention_mask'].to(dev)
            labels = batch['label'].to(dev)
            logits = model(ids, mask)
            loss = crit(logits, labels)
            tl += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds==labels).sum().item(); total += labels.size(0)
    return tl/len(loader), correct/total

def train():
    cfg = Config()
    v3_data = 'C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/data/dataset/sentiment_dataset_v3.csv'
    v3_ckpt = 'C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/model/checkpoints/best_model_v3.pt'
    device = cfg.DEVICE
    print('device={}'.format(device))
    random.seed(2024); torch.manual_seed(2024)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(2024)
    np.random.seed(2024)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    print('loading v3 data...')
    df = load_data(v3_data)
    print('dataset={}, dist={}'.format(len(df), df['label'].value_counts().to_dict()))
    train_df, val_df, test_df = split_data(df, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.TEST_RATIO)
    print('train={} val={} test={}'.format(len(train_df), len(val_df), len(test_df)))
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    train_loader = make_loader(train_df, tokenizer, cfg.MAX_LEN, cfg.BATCH_SIZE, shuffle=True, aug=True, aug_ratio=0.7)
    val_loader = create_data_loader(val_df, tokenizer, cfg.MAX_LEN, cfg.BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_df, tokenizer, cfg.MAX_LEN, cfg.BATCH_SIZE, shuffle=False)
    print('building model...')
    model = BertBiLSTMCNN(cfg).to(device)
    freeze_bert(model, n=8)
    tp = sum(p.numel() for p in model.parameters())
    trp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total={:,} trainable={:,} frozen={:,}'.format(tp, trp, tp-trp))
    lr = 3e-5
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = len(train_loader) * cfg.EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(total_steps*0.1), total_steps)
    crit = LSCE(eps=0.15)
    print('training [Plan A+B+C v3 - strong reg]...')
    best_acc = 0; patience = 0; hist = []
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        print('')
        print('--- Epoch {}/{} ---  LR={}'.format(epoch+1, cfg.EPOCHS, lr))
        tl, ta = train_epoch(model, train_loader, opt, sched, crit, device, use_mixup=True)
        vl, va = eval_epoch(model, val_loader, crit, device)
        elapsed = time.time() - t0
        gap = ta - va
        print('train_loss={:.4f} train_acc={:.4f}  val_loss={:.4f} val_acc={:.4f}  gap={:.4f}  time={:.0f}s'.format(tl,ta,vl,va,gap,elapsed))
        hist.append({'epoch':epoch+1,'tl':round(tl,4),'ta':round(ta,4),'vl':round(vl,4),'va':round(va,4),'gap':round(gap,4)})
        if va > best_acc:
            best_acc = va; patience = 0
            torch.save({'epoch':epoch,'state':model.state_dict(),'val_acc':va,'val_loss':vl}, v3_ckpt)
            print('saved best (val_acc={:.4f})'.format(va))
        else:
            patience += 1
            print('no improvement ({}/{})'.format(patience, cfg.EARLY_STOP_PATIENCE))
        if patience >= cfg.EARLY_STOP_PATIENCE:
            print('early stopping!'); break
    print('')
    print('=== Test ===')
    ckpt = torch.load(v3_ckpt, map_location=device)
    model.load_state_dict(ckpt['state'])
    tloss, tacc = eval_epoch(model, test_loader, crit, device)
    print('test_loss={:.4f} test_acc={:.4f}'.format(tloss, tacc))
    import pandas as pd
    hist_path = 'C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/model/training_history_v3.csv'
    pd.DataFrame(hist).to_csv(hist_path, index=False)
    print('history saved: {}'.format(hist_path))
    print(pd.DataFrame(hist).to_string(index=False))

if __name__ == '__main__': train()
