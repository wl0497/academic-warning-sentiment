"""
训练脚本 — Bert-BiLSTM-CNN 模型训练
====================================
参数严格匹配申报书：batch=32, lr=2e-5, Dropout+EarlyStopping
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from config import Config
from bert_bilstm_cnn import BertBiLSTMCNN
from utils import load_data, split_data, create_data_loader


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        # 梯度裁剪，防止梯度消失/爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def eval_epoch(model, data_loader, criterion, device):
    """验证/测试一个epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    """主训练流程"""
    config = Config()
    device = config.DEVICE
    print(f"使用设备: {device}")
    
    # 确保checkpoint目录存在
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # 1. 加载数据
    print("=" * 60)
    print("加载训练数据...")
    df = load_data(config.DATA_PATH)
    train_df, val_df, test_df = split_data(
        df, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO
    )
    
    # 2. 初始化tokenizer和dataloader
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    train_loader = create_data_loader(train_df, tokenizer, config.MAX_LEN, config.BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_df, tokenizer, config.MAX_LEN, config.BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_df, tokenizer, config.MAX_LEN, config.BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型
    print("=" * 60)
    print("初始化Bert-BiLSTM-CNN模型...")
    model = BertBiLSTMCNN(config).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 优化器与调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    # 5. 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 6. 训练循环 + EarlyStopping
    print("=" * 60)
    print("开始训练...")
    print(f"参数: batch_size={config.BATCH_SIZE}, lr={config.LEARNING_RATE}")
    
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        # 验证
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.1f}s")
        
        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, config.BEST_MODEL_PATH)
            print(f"✅ 保存最优模型 (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ 验证准确率未提升 ({patience_counter}/{config.EARLY_STOP_PATIENCE})")
        
        # EarlyStopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early Stopping! 连续{config.EARLY_STOP_PATIENCE}轮验证准确率未提升")
            break
    
    # 7. 测试集最终评估
    print("\n" + "=" * 60)
    print("加载最优模型，在测试集上评估...")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"测试集结果: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
    
    # 检查是否达标
    if test_acc >= 0.884:
        print(f"✅ 准确率达标! ({test_acc:.4f} ≥ 0.884)")
    else:
        print(f"❌ 准确率未达标 ({test_acc:.4f} < 0.884)，需要调优")
    
    return model, history


if __name__ == "__main__":
    train()
