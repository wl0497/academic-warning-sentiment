# train_v2.py - 方案A+C增强版
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time, random, re, torch, torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from config_v2 import Config
from bert_bilstm_cnn import BertBiLSTMCNN
from utils import load_data, split_data, create_data_loader

# 同义词词典
SYNONYM_DICT = {
    '压力': ['负担','重担','压迫','焦虑'], '很大': ['非常大','不小','挺大'],
    '学习': ['学业','上课','念书','读书'], '困难': ['艰难','费劲','吃力'],
    '学校': ['院校','学府','大学','高校'], '成绩': ['分数','绩点','得分'],
    '课程': ['课业','功课','学分课'], '考试': ['测评','考查','测验'],
    '老师': ['教师','导师','教授'], '同学': ['同伴','同窗','室友'],
    '帮助': ['协助','扶持','援手','帮忙'], '支持': ['支撑','力挺','赞成'],
    '问题': ['麻烦','困扰','难题'], '时间': ['工夫','时光','时候'],
    '努力': ['尽力','加油','奋发','刻苦'], '改进': ['改善','提升','提高'],
    '预警': ['警示','提醒','警告'], '毕业': ['结业','离校','完成学业'],
}
AUX_WORDS = ['的','了','很','也','都','就','还','又','再','有点','有些','确实','感觉','好像','应该','可能','大概','也许','其实','不过','但是']

def synonym_replace(text, n=1):
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    if not words: return text
    replaceable = [w for w in words if w in SYNONYM_DICT]
    if not replaceable: return text
    n = min(n, len(replaceable))
    result = text
    for word in random.sample(replaceable, n):
        result = result.replace(word, random.choice(SYNONYM_DICT[word]), 1)
    return result

def random_delete(text, p=0.15):
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    if len(words) <= 3: return text
    kept = [w for w in words if random.random() > p]
    if len(kept) < len(words) * 0.5:
        kept = list(set(random.sample(words, max(3, int(len(words) * 0.5)))))
    result = text
    for w in words:
        if w not in kept:
            result = re.sub(r'[{0}]?{1}'.format('|'.join(AUX_WORDS), re.escape(w)), '', result)
    result = re.sub(r'\s+', '', result).strip()
    return result if len(result) >= 5 else text

def random_swap(text, n=1):
    chars = list(text)
    positions = [i for i,c in enumerate(chars) if '\u4e00' <= c <= '\u9fff']
    for _ in range(min(n, len(positions)-1)):
        idx = random.randint(0, len(positions)-2)
        i,j = positions[idx], positions[idx+1]
        if j-i==1: chars[i],chars[j] = chars[j],chars[i]
    return ''.join(chars)

def add_noise(text, p=0.1):
    if random.random() > p: return text
    noises = ['说实话，','怎么说呢，','唉，','其实吧，','反正就是，','说真的，','感觉吧，','嗯，','嘛，','哎，']
    suffix = random.choice(['...','吧。','呢。','哈。','嗯。','呀。'])
    if random.random() < 0.5: return random.choice(noises) + text
    else: return text + suffix

def augment_text(text):
    result = text
    if random.random() < 0.3: result = synonym_replace(result, random.randint(1,3))
    if random.random() < 0.2: result = random_delete(result, p=0.1)
    if random.random() < 0.1: result = random_swap(result, n=1)
    result = add_noise(result, p=0.15)
    return result

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        n = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-true_dist * log_preds).sum(dim=-1).mean()

def add_gradient_noise(model, std=5e-4):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * std)

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=True, aug_ratio=0.5):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer
        self.max_len = max_len; self.augment = augment; self.aug_ratio = aug_ratio
        self.cache = {}
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment and random.random() < self.aug_ratio:
            if idx not in self.cache: self.cache[idx] = augment_text(text)
            text = self.cache[idx]
        enc = self.tokenizer.encode_plus(text, add_special_tokens=True,
            max_length=self.max_len, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def create_aug_loader(df, tokenizer, max_len, batch_size, shuffle=True, aug=True, aug_ratio=0.5):
    ds = AugmentedDataset(df['text'].values, df['label'].values, tokenizer, max_len, aug, aug_ratio)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def freeze_bert(model, n=6):
    """freeze BERT layers 0..n-1"""
    frozen = 0
    for name, param in model.bert.named_parameters():
        parts = name.split('.')
        if 'encoder' in parts and 'layer' in parts:
            try:
                li = parts.index('layer')
                layer_num = int(parts[li+1])
                if layer_num < n:
                    param.requires_grad = False
                    frozen += param.numel()
            except (ValueError, IndexError): pass
    print('frozen {0} layer params ({1:,})'.format(n, frozen))

def train_epoch(model, loader, optimizer, scheduler, criterion, device, use_noise=False):
    model.train(); total_loss = 0; correct = 0; total = 0
    for batch in tqdm(loader, desc='Training'):
        ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_noise: add_gradient_noise(model)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct += (preds==labels).sum().item(); total += labels.size(0)
    return total_loss/len(loader), correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval(); total_loss = 0; correct = 0; total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds==labels).sum().item(); total += labels.size(0)
    return total_loss/len(loader), correct/total

def train():
    cfg = Config()
    device = cfg.DEVICE
    print('device: {0}'.format(device))
    random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    
    print('loading data...')
    df = load_data(cfg.DATA_PATH)
    print('dataset: {0}'.format(len(df)))
    train_df, val_df, test_df = split_data(df, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.TEST_RATIO)
    
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    train_loader = create_aug_loader(train_df, tokenizer, cfg.MAX_LEN, cfg.BATCH_SIZE, shuffle=True, aug=True, aug_ratio=0.5)
    val_loader = create_data_loader(val_df, tokenizer, cfg.MAX_LEN, cfg.BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_df, tokenizer, cfg.MAX_LEN, cfg.BATCH_SIZE, shuffle=False)
    
    print('building model...')
    model = BertBiLSTMCNN(cfg).to(device)
    freeze_bert(model, n=6)
    
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params: {0:,}  trainable: {1:,}  frozen: {2:,}'.format(total_p, trainable_p, total_p-trainable_p))
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = len(train_loader) * cfg.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*cfg.WARMUP_RATIO), total_steps)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    print('training [Plan A+C]...')
    best_acc = 0; patience = 0
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        print('\n--- Epoch {0}/{1} ---'.format(epoch+1, cfg.EPOCHS))
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, use_noise=True)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print('train_loss={0:.4f} train_acc={1:.4f}  val_loss={2:.4f} val_acc={3:.4f}  time={4:.1f}s'.format(t_loss,t_acc,v_loss,v_acc,elapsed))
        if v_acc > best_acc:
            best_acc = v_acc; patience = 0
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'val_acc':v_acc,'val_loss':v_loss}, cfg.BEST_MODEL_PATH)
            print('saved best (val_acc={0:.4f})'.format(v_acc))
        else:
            patience += 1
            print('no improvement ({0}/{1})'.format(patience, cfg.EARLY_STOP_PATIENCE))
        if patience >= cfg.EARLY_STOP_PATIENCE:
            print('early stopping!'); break
    
    print('\n=== test ===')
    ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print('test_loss={0:.4f} test_acc={1:.4f}'.format(test_loss, test_acc))
    if test_acc >= 0.884: print('PASSED!')
    else: print('NOT PASSED')

if __name__ == '__main__':
    train()