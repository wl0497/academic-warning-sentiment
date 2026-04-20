# -*- coding: utf-8 -*-
"""
V4 Training - 使用V4数据集（20K训练 + OOD测试）
改进：冻结更多BERT层 + 更高正则化
"""
import os, sys, time, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ========== CONFIG ==========
class ConfigV4:
    BERT_MODEL_NAME = 'bert-base-chinese'
    MAX_LEN = 128
    BATCH_SIZE = 16  # 更小batch减少OOM
    EPOCHS = 15
    LR = 2e-5  # 更低学习率
    NUM_CLASSES = 3
    FREEZE_LAYERS = 8
    HIDDEN_SIZE = 64
    CNN_CHANNELS = 64
    DROPOUT = 0.6  # 更高dropout
    LABEL_SMOOTHING = 0.2  # 更强label smoothing
    MIXUP_ALPHA = 0.0  # 不在input_ids上做mixup（BERT需要离散token）
    AUG_RATIO = 0.8  # 更强增强
    GRADIENT_ACCUMULATION = 2  # 梯度累积，effective batch=32
    PATIENCE = 8
    SEED = 42
    # V4 data
    TRAIN_DATA = 'C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/data/dataset/sentiment_dataset_v4_train.csv'
    TEST_DATA = 'C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/data/dataset/sentiment_dataset_v4_test.csv'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = ConfigV4()

# ========== SEED ==========
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(cfg.SEED)

# ========== AUGMENTATION ==========
CONJUNCTIONS = ['，然后', '，接着', '，于是', '，不过', '，但实际上', '，然而', '，而且']
ADVERBS = ['真的', '确实', '其实', '基本上', '总体上', '大致', '总体而言']
PUNCTUATIONS = ['。', '！', '？', '呀', '吧', '呢', '啊', '哈', '哦']
COLLOQUIAL = ['有点', '稍微', '感觉', '觉得', '好像', '似乎', '大概是']

def augment_text(text):
    if random.random() > cfg.AUG_RATIO:
        return text
    operations = random.sample(['swap', 'delete', 'conjugate', 'adverb', 'puncture', 'colloquial', 'shuffle'], 
                                 k=random.randint(1, 3))
    for op in operations:
        if op == 'swap' and len(text) > 4:
            i, j = sorted(random.sample(range(len(text)), 2))
            text = text[:i] + text[j] + text[i+1:j] + text[i] + text[j+1:]
        elif op == 'delete' and len(text) > 6:
            idx = random.randint(0, len(text)-1)
            text = text[:idx] + text[idx+1:]
        elif op == 'conjugate' and '，' not in text and len(text) > 10:
            insert = random.choice(CONJUNCTIONS)
            pos = random.randint(len(text)//3, 2*len(text)//3)
            text = text[:pos] + insert + text[pos:]
        elif op == 'adverb':
            prefix = random.choice(ADVERBS)
            text = prefix + text
        elif op == 'puncture':
            text = text + random.choice(PUNCTUATIONS)
        elif op == 'colloquial':
            text = text + '，' + random.choice(COLLOQUIAL) + random.choice(PUNCTUATIONS)
        elif op == 'shuffle' and len(text) > 10:
            parts = text.split('，')
            if len(parts) >= 2:
                random.shuffle(parts)
                text = '，'.join(parts)
    return text

# ========== DATASET ==========
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = augment_text(text)
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_dataloader(texts, labels, tokenizer, max_len, batch_size, shuffle=True, augment=False):
    dataset = SentimentDataset(texts, labels, tokenizer, max_len, augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=True if torch.cuda.is_available() else False)

# ========== MODEL ==========
class BertBiLSTMCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(config.FREEZE_LAYERS):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=config.HIDDEN_SIZE,
            num_layers=1, batch_first=True, bidirectional=True, dropout=0
        )
        self.conv = nn.Conv1d(in_channels=config.HIDDEN_SIZE*2, out_channels=config.CNN_CHANNELS, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(config.HIDDEN_SIZE*2 + config.CNN_CHANNELS, config.NUM_CLASSES)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(sequence_output)
        lstm_feat = lstm_out.mean(dim=1)
        
        conv_input = lstm_out.permute(0, 2, 1)
        conv_out = torch.relu(self.conv(conv_input))
        conv_feat = conv_out.mean(dim=2)
        
        fused = torch.cat([lstm_feat, conv_feat], dim=1)
        fused = self.dropout(fused)
        logits = self.fc(fused)
        return logits

# ========== MIXUP ==========
def mixup_data(x, y, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========== METRICS ==========
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return np.array(all_preds), np.array(all_labels)

# ========== MAIN ==========
def main():
    print(f"Device: {cfg.DEVICE}")
    print(f"Config: epochs={cfg.EPOCHS}, lr={cfg.LR}, bs={cfg.BATCH_SIZE}, "
          f"freeze={cfg.FREEZE_LAYERS}, dropout={cfg.DROPOUT}, "
          f"ls={cfg.LABEL_SMOOTHING}, mixup={cfg.MIXUP_ALPHA}, aug={cfg.AUG_RATIO}")
    
    # Load data
    print(f"\nLoading train data...")
    df = pd.read_csv(cfg.TRAIN_DATA, encoding='utf-8')
    print(f"Total: {len(df)}, dist: {df['label'].value_counts().to_dict()}")
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.2, random_state=42, stratify=df['label'].tolist()
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Load OOD test
    ood_df = pd.read_csv(cfg.TEST_DATA, encoding='utf-8')
    ood_texts = ood_df['text'].tolist()
    ood_labels = ood_df['label'].tolist()
    print(f"OOD Test: {len(ood_texts)}")
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    
    # Dataloaders (no augmentation for val/test)
    train_loader = create_dataloader(train_texts, train_labels, tokenizer, cfg.MAX_LEN,
                                     cfg.BATCH_SIZE, shuffle=True, augment=True)
    val_loader = create_dataloader(val_texts, val_labels, tokenizer, cfg.MAX_LEN,
                                   cfg.BATCH_SIZE * 2, shuffle=False, augment=False)
    test_loader = create_dataloader(test_texts, test_labels, tokenizer, cfg.MAX_LEN,
                                    cfg.BATCH_SIZE * 2, shuffle=False, augment=False)
    ood_loader = create_dataloader(ood_texts, ood_labels, tokenizer, cfg.MAX_LEN,
                                   cfg.BATCH_SIZE * 2, shuffle=False, augment=False)
    
    # Model
    print("\nBuilding model...")
    model = BertBiLSTMCNN(cfg).to(cfg.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.bert.encoder.layer[:cfg.FREEZE_LAYERS].parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Trainable: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS * len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    
    # Training
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"\nTraining V4 ({cfg.EPOCHS} epochs, gradient accumulation={cfg.GRADIENT_ACCUMULATION})...")
    start_time = time.time()
    
    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False)
        for step, batch in enumerate(pbar):
            ids = batch['input_ids'].to(cfg.DEVICE)
            mask = batch['attention_mask'].to(cfg.DEVICE)
            labels = batch['label'].to(cfg.DEVICE)
            
            # Mixup
            if cfg.MIXUP_ALPHA > 0:
                ids, labels_a, labels_b, lam = mixup_data(ids, labels, cfg.MIXUP_ALPHA)
            
            # Mixup produces FloatTensor; BERT needs LongTensor for input_ids
            logits = model(ids.long(), mask.long())
            
            if cfg.MIXUP_ALPHA > 0:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)
            
            (loss / cfg.GRADIENT_ACCUMULATION).backward()
            
            if (step + 1) % cfg.GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * cfg.GRADIENT_ACCUMULATION
            _, preds = torch.max(logits, dim=1)
            if cfg.MIXUP_ALPHA > 0:
                correct = (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
            else:
                correct = (preds == labels).sum().item()
            running_correct += correct
            running_total += labels.size(0)
            
            pbar.set_postfix({'loss': running_loss/(step+1), 'acc': running_correct/running_total})
        
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / running_total
        
        # Val
        val_preds, val_labels_arr = evaluate(model, val_loader, cfg.DEVICE)
        val_acc = accuracy_score(val_labels_arr, val_preds)
        val_loss = running_loss / len(train_loader)  # approximate
        
        # Test (in-distribution)
        test_preds, test_labels_arr = evaluate(model, test_loader, cfg.DEVICE)
        test_acc = accuracy_score(test_labels_arr, test_preds)
        
        # OOD Test
        ood_preds, ood_labels_arr = evaluate(model, ood_loader, cfg.DEVICE)
        ood_acc = accuracy_score(ood_labels_arr, ood_preds)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{cfg.EPOCHS}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} ood_acc={ood_acc:.4f} "
              f"time={epoch_time:.0f}s")
        
        record = {
            'epoch': epoch, 'train_loss': round(train_loss, 4), 'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 4),
            'test_acc': round(test_acc, 4), 'ood_acc': round(ood_acc, 4),
            'time': round(epoch_time, 1)
        }
        history.append(record)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt = {
                'epoch': epoch, 'state': model.state_dict(),
                'val_acc': val_acc, 'test_acc': test_acc, 'ood_acc': ood_acc,
                'train_acc': train_acc
            }
            torch.save(ckpt, 'checkpoints/best_model_v4.pt')
            print(f"  *** Saved best (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{cfg.PATIENCE})")
            if patience_counter >= cfg.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time/60:.1f}min, Best val_acc: {best_val_acc:.4f}")
    
    # Save history
    pd.DataFrame(history).to_csv('training_history_v4.csv', index=False)
    print("Saved: training_history_v4.csv")
    
    # Final evaluation
    print("\n=== Final Evaluation on Best Model ===")
    ckpt = torch.load('checkpoints/best_model_v4.pt', map_location=cfg.DEVICE)
    model.load_state_dict(ckpt['state'])
    
    for name, loader, labels_arr in [
        ('Val (In-Dist)', val_loader, val_labels_arr),
        ('Test (In-Dist)', test_loader, test_labels_arr),
        ('OOD Test', ood_loader, ood_labels_arr)
    ]:
        preds, _ = evaluate(model, loader, cfg.DEVICE)
        acc = accuracy_score(labels_arr, preds)
        f1 = f1_score(labels_arr, preds, average='macro')
        print(f"\n{name}: Acc={acc*100:.2f}%, F1={f1:.4f}")
        print(classification_report(labels_arr, preds, target_names=['pos','neu','neg'], digits=4))
        cm = confusion_matrix(labels_arr, preds)
        print(f"CM:\n{cm}")

if __name__ == '__main__':
    os.chdir(r'C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\model')
    main()
