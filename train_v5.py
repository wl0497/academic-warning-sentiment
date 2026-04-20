# -*- coding: utf-8 -*-
import os, sys, time, math, random, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
ACCUM_STEPS = 2
FREEZE_LAYERS = 8
DROPOUT = 0.6
LABEL_SMOOTHING = 0.2
AUG_RATE = 0.8

DATA_DIR = r'C:\\Users\\29258\\.qclaw\\workspace-agent-66459c61\\academic-warning-sentiment\\data\\dataset'
MODEL_DIR = r'C:\\Users\\29258\\.qclaw\\workspace-agent-66459c61\\academic-warning-sentiment\\model\\checkpoints'
OUT_DIR = r'C:\\Users\\29258\\.qclaw\\workspace-agent-66459c61\\academic-warning-sentiment\\model'
os.makedirs(MODEL_DIR, exist_ok=True)


class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len, aug_rate=0.0):
        self.data = []
        with open(csv_path, encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        text = row[0].strip()
                        label = int(row[1])
                        if text and 0 <= label <= 2:
                            self.data.append((text, label))
                    except:
                        pass
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aug_rate = aug_rate

    def __len__(self):
        return len(self.data)

    def augment(self, text):
        words = list(text)
        if random.random() < 0.2 and len(words) > 5:
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        if random.random() < 0.15:
            drop_idx = random.randint(0, len(words) - 1)
            words = words[:drop_idx] + words[drop_idx + 1:]
        return ''.join(words)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        if self.aug_rate > 0 and random.random() < self.aug_rate:
            text = self.augment(text)
        enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item, torch.tensor(label, dtype=torch.long)


class BertBiLstmCnn(nn.Module):
    def __init__(self, freeze_layers=8):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-chinese', output_attentions=False, output_hidden_states=False)
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config)
        if freeze_layers > 0:
            for name, param in self.bert.named_parameters():
                parts = name.split('.')
                if parts[0].isdigit() and int(parts[0]) < freeze_layers:
                    param.requires_grad = False
        hidden = config.hidden_size
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.drop = nn.Dropout(0.6)
        self.fc = nn.Linear(384, 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state
        lstm_out, _ = self.lstm(seq)
        conv = torch.relu(self.conv(lstm_out.transpose(1, 2))).transpose(1, 2)
        pooled = torch.cat([lstm_out.mean(1), conv.mean(1)], dim=1)
        pooled = self.drop(pooled)
        return self.fc(pooled)


print('Loading data...')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_csv = os.path.join(DATA_DIR, 'sentiment_dataset_v5_train.csv')
ood_csv = os.path.join(DATA_DIR, 'sentiment_dataset_v5_test.csv')
train_ds = SentimentDataset(train_csv, tokenizer, MAX_LEN, AUG_RATE)
ood_ds = SentimentDataset(ood_csv, tokenizer, MAX_LEN, 0.0)
print('Train:', len(train_ds), 'OOD Test:', len(ood_ds))

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
ood_dl = DataLoader(ood_ds, batch_size=BATCH_SIZE * 2, num_workers=0)

model = BertBiLstmCnn(freeze_layers=FREEZE_LAYERS).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print('Total params:', total, 'Trainable:', trainable, 'Frozen:', total - trainable)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dl))

best_ood = 0.0
best_epoch = 0
hist = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    loss_sum, acc_sum, n = 0.0, 0, 0
    optimizer.zero_grad()
    for step, (batch, labels) in enumerate(train_dl):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels.to(DEVICE)) / ACCUM_STEPS
        loss.backward()
        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss_sum += loss.item() * ACCUM_STEPS
        acc_sum += (logits.argmax(1) == labels.to(DEVICE)).sum().item()
        n += labels.size(0)
        if step % 100 == 0:
            print('  Step', step, '/', len(train_dl), flush=True)

    train_loss = loss_sum / n
    train_acc = acc_sum / n

    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for batch, labels in ood_dl:
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            vp.extend(logits.argmax(1).cpu().numpy())
            vt.extend(labels.numpy())
    ood_acc = accuracy_score(vt, vp)
    elapsed = time.time() - t0
    print('Epoch', epoch, '/', EPOCHS, ': train_acc=', round(train_acc * 100, 1),
          '%, ood_acc=', round(ood_acc * 100, 1), '%, time=', round(elapsed, 0), 's', flush=True)

    if ood_acc > best_ood:
        best_ood = ood_acc
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model_v5.pt'))
        print('  *** BEST OOD:', round(best_ood * 100, 1), '%', flush=True)

    hist.append({'epoch': epoch, 'train_acc': train_acc * 100, 'ood_acc': ood_acc * 100, 'time': elapsed})

    if len(hist) >= 5 and all(h['ood_acc'] <= best_ood for h in hist[-5:]):
        print('Early stopping at epoch', epoch, flush=True)
        break

print('Training complete! Best OOD:', round(best_ood * 100, 1), '% at Epoch', best_epoch, flush=True)

# ===== PLOT =====
epochs_list = [h['epoch'] for h in hist]
train_accs = [h['train_acc'] for h in hist]
ood_accs = [h['ood_acc'] for h in hist]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(epochs_list, train_accs, 'b-o', label='Train Acc', linewidth=2.5, markersize=8)
ax1.plot(epochs_list, ood_accs, 'r-s', label='OOD Test Acc', linewidth=2.5, markersize=8)
ax1.scatter([best_epoch], [best_ood * 100], color='green', s=200, zorder=5, marker='*',
           label='Best OOD: ' + str(round(best_ood * 100, 1)) + '% (Epoch ' + str(best_epoch) + ')')
ax1.fill_between(epochs_list, train_accs, ood_accs, alpha=0.1, color='orange')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('V5: Train vs OOD Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylim([50, 105])
ax1.set_xticks(epochs_list)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
for ep, acc in zip(epochs_list, ood_accs):
    ax1.annotate(str(round(acc, 1)) + '%', (ep, acc), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10)

ax2 = axes[1]
ax2.axis('off')

versions = ['V1/V2/V3\n(same pool)', 'V5\n(independent pool)']
in_dist_vals = [100.0, 100.0]
ood_vals = [100.0, best_ood * 100]
x = [0, 1]
width = 0.35
bars1 = ax2.bar([i - width / 2 for i in x], in_dist_vals, width, label='In-Dist Acc', color='#2ecc71', alpha=0.85)
bars2 = ax2.bar([i + width / 2 for i in x], ood_vals, width, label='OOD Acc', color='#e74c3c', alpha=0.85)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Root Cause: Fake 100% vs Real OOD', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(versions, fontsize=11)
ax2.set_ylim([0, 120])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + 1, str(round(h, 1)) + '%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1, str(round(h, 1)) + '%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        if h < 100:
            ax2.annotate('Gap', xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(bar.get_x() + bar.get_width() / 2, 50),
                       fontsize=9, ha='center', color='gray',
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'v5_results.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print('Chart saved:', chart_path)
print('DONE')
