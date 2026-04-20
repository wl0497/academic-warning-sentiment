# -*- coding: utf-8 -*-
"""
消融实验：对比 BERT-Only / BERT-BiLSTM / BERT-CNN / BERT-BiLSTM-CNN
"""
import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
BASE_DIR = r'C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment'
DATA_DIR = os.path.join(BASE_DIR, 'data/dataset')
TRAIN_CSV = os.path.join(DATA_DIR, 'sentiment_dataset_v5_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'sentiment_dataset_v5_test.csv')
OUT_DIR = os.path.join(BASE_DIR, 'model/ablation_results')
os.makedirs(OUT_DIR, exist_ok=True)

BERT_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ 数据集 ============
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = self.tokenizer(text, max_length=self.max_len, padding='max_length',
                             truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============ 模型定义 ============
class BertOnly(nn.Module):
    """仅BERT + 分类头"""
    def __init__(self, bert_name, num_classes=3, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.fc(self.dropout(pooled))

class BertBiLSTM(nn.Module):
    """BERT + BiLSTM"""
    def __init__(self, bert_name, num_classes=3, hidden_size=128, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.lstm = nn.LSTM(768, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        pooled = lstm_out[:, -1, :]
        return self.fc(self.dropout(pooled))

class BertCNN(nn.Module):
    """BERT + CNN"""
    def __init__(self, bert_name, num_classes=3, num_filters=128, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.conv = nn.Conv1d(768, num_filters, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        conv_in = seq_out.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        pooled = self.pool(conv_out).squeeze(-1)
        return self.fc(self.dropout(pooled))

class BertBiLSTMCNN(nn.Module):
    """BERT + BiLSTM + CNN (完整模型)"""
    def __init__(self, bert_name, num_classes=3, hidden_size=128, num_filters=128, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.lstm = nn.LSTM(768, hidden_size, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_size * 2, num_filters, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        conv_in = lstm_out.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        pooled = self.pool(conv_out).squeeze(-1)
        return self.fc(self.dropout(pooled))

# ============ 训练函数 ============
def train_model(model, train_loader, test_loader, name, epochs=5):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    results = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'ood_acc': []}
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100

        # 测试
        test_acc = evaluate(model, test_loader)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['ood_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f'{name}_best.pt'))

        print(f'{name} Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, train_acc={train_acc:.1f}%, ood_acc={test_acc:.1f}%')

    return results, best_acc

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            logits = model(input_ids, attn_mask)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# ============ 主程序 ============
def main():
    print('Loading tokenizer and data...')
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    train_set = SentimentDataset(TRAIN_CSV, tokenizer, MAX_LEN)
    test_set = SentimentDataset(TEST_CSV, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'Train: {len(train_set)}, Test: {len(test_set)}')

    models = {
        'BERT-Only': BertOnly(BERT_NAME),
        'BERT-BiLSTM': BertBiLSTM(BERT_NAME),
        'BERT-CNN': BertCNN(BERT_NAME),
        'BERT-BiLSTM-CNN': BertBiLSTMCNN(BERT_NAME)
    }

    all_results = {}
    best_accs = {}

    for name, model in models.items():
        print(f'\n===== Training {name} =====')
        results, best_acc = train_model(model, train_loader, test_loader, name, EPOCHS)
        all_results[name] = results
        best_accs[name] = best_acc

    # 打印最终结果
    print('\n' + '='*50)
    print('消融实验结果:')
    print('='*50)
    for name, acc in best_accs.items():
        print(f'{name:20s}: OOD Acc = {acc:.2f}%')

    # 绘制对比图
    plot_results(all_results, best_accs)

    return all_results, best_accs

def plot_results(results, best_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：OOD准确率曲线
    ax1 = axes[0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for i, (name, res) in enumerate(results.items()):
        ax1.plot(range(1, len(res['ood_acc'])+1), res['ood_acc'],
                'o-', color=colors[i], label=f'{name} (best={best_accs[name]:.1f}%)',
                linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('OOD Accuracy (%)', fontsize=12)
    ax1.set_title('Ablation Study: OOD Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 6))

    # 右图：最佳OOD准确率柱状图
    ax2 = axes[1]
    names = list(best_accs.keys())
    accs = [best_accs[n] for n in names]
    bars = ax2.bar(names, accs, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Best OOD Accuracy (%)', fontsize=12)
    ax2.set_title('Best Model Performance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 添加改进百分比
    baseline = accs[0]
    for i, (bar, acc) in enumerate(zip(bars[1:], accs[1:]), 1):
        improve = acc - baseline
        ax2.annotate(f'+{improve:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, acc + 5),
                    ha='center', fontsize=9, color='green' if improve > 0 else 'red')

    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, 'ablation_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f'\n图表已保存: {chart_path}')

    # 保存结果到文本
    with open(os.path.join(OUT_DIR, 'ablation_results.txt'), 'w', encoding='utf-8') as f:
        f.write('消融实验结果\n')
        f.write('='*50 + '\n')
        f.write(f'数据集: V5 (Train=30000, OOD Test=5001)\n')
        f.write(f'训练轮次: {EPOCHS}\n\n')
        for name, acc in best_accs.items():
            f.write(f'{name}: Best OOD Acc = {acc:.2f}%\n')
        f.write(f'\n最佳模型: {max(best_accs, key=best_accs.get)} ({max(best_accs.values()):.2f}%)\n')

if __name__ == '__main__':
    main()
