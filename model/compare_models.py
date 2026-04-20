"""
模型对比实验 - 4种架构公平对比
运行: python model/compare_models.py
"""
import os
import sys
import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn

# 设置随机种子确保可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 导入模型配置
from model_configs import (
    BertOnlyModel, BertCNNModel, BertBiLSTMModel, BertBiLSTMCNNModel
)
from train_utils import train_model, evaluate_model

# 实验配置
EXPERIMENT_CONFIG = {
    'seed': 42,
    'train_file': 'data/dataset/sentiment_dataset_v5_train.csv',
    'test_file': 'data/dataset/sentiment_dataset_v5_test.csv',
    'ood_test_file': 'data/dataset/sentiment_dataset_v4_test.csv',  # 使用v4的测试集作为OOD
    'output_dir': 'model/comparison_results',
    'max_len': 128,
    'batch_size': 16,
    'epochs': 15,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'freeze_bert_layers': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 模型定义
MODELS = {
    'BERT-Only': {
        'class': BertOnlyModel,
        'description': '仅使用BERT [CLS]输出',
        'color': '#3498db'
    },
    'BERT-CNN': {
        'class': BertCNNModel,
        'description': 'BERT + CNN局部特征提取',
        'color': '#e74c3c'
    },
    'BERT-BiLSTM': {
        'class': BertBiLSTMModel,
        'description': 'BERT + BiLSTM序列建模',
        'color': '#2ecc71'
    },
    'BERT-BiLSTM-CNN': {
        'class': BertBiLSTMCNNModel,
        'description': 'BERT + BiLSTM + CNN融合（完整模型）',
        'color': '#9b59b6'
    }
}


def load_datasets(config):
    """加载数据集"""
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 读取数据
    train_df = pd.read_csv(config['train_file'])
    test_df = pd.read_csv(config['test_file'])
    ood_df = pd.read_csv(config['ood_test_file'])
    
    # 划分验证集（从训练集取10%）
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.sample(n=val_size, random_state=config['seed'])
    train_df = train_df.drop(val_df.index)
    
    print(f"数据集大小: 训练={len(train_df)}, 验证={len(val_df)}, 测试={len(test_df)}, OOD={len(ood_df)}")
    
    return train_df, val_df, test_df, ood_df, tokenizer


def create_dataloaders(train_df, val_df, test_df, ood_df, tokenizer, config):
    """创建数据加载器"""
    from train_utils import SentimentDataset
    
    train_dataset = SentimentDataset(train_df, tokenizer, config['max_len'])
    val_dataset = SentimentDataset(val_df, tokenizer, config['max_len'])
    test_dataset = SentimentDataset(test_df, tokenizer, config['max_len'])
    ood_dataset = SentimentDataset(ood_df, tokenizer, config['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, ood_loader


def count_parameters(model):
    """统计可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(model_name, model_config, train_loader, val_loader, 
                   test_loader, ood_loader, config):
    """运行单个模型实验"""
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {model_config['description']}")
    print(f"{'='*60}")
    
    set_seed(config['seed'])
    
    # 创建模型
    device = config['device']
    model = model_config['class'](freeze_bert_layers=config['freeze_bert_layers'])
    model = model.to(device)
    
    # 统计参数量
    total_params = count_parameters(model)
    print(f"可训练参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 训练
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        device=device,
        model_name=model_name.replace(' ', '_')
    )
    
    # 加载最佳模型进行评估
    checkpoint_path = f"{config['output_dir']}/checkpoints/{model_name.replace(' ', '_')}_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 评估
    print(f"\n评估 {model_name}...")
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    ood_acc, ood_f1 = evaluate_model(model, ood_loader, device)
    
    # 获取训练历史中的最佳验证指标
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    final_train_acc = history['train_acc'][-1]
    
    results = {
        'model_name': model_name,
        'description': model_config['description'],
        'parameters': total_params,
        'train_acc': final_train_acc,
        'best_val_acc': best_val_acc,
        'best_val_epoch': best_val_epoch,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'ood_acc': ood_acc,
        'ood_f1': ood_f1,
        'history': history,
        'color': model_config['color']
    }
    
    print(f"\n{model_name} 最终结果:")
    print(f"  Train Acc: {final_train_acc:.4f}")
    print(f"  Val Acc: {best_val_acc:.4f} (Epoch {best_val_epoch})")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  OOD Acc: {ood_acc:.4f} ⭐")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  OOD F1: {ood_f1:.4f}")
    
    return results


def plot_training_curves(all_results, output_dir):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 训练准确率
    ax = axes[0, 0]
    for result in all_results:
        epochs = range(1, len(result['history']['train_acc']) + 1)
        ax.plot(epochs, result['history']['train_acc'], 
                label=result['model_name'], color=result['color'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 验证准确率
    ax = axes[0, 1]
    for result in all_results:
        epochs = range(1, len(result['history']['val_acc']) + 1)
        ax.plot(epochs, result['history']['val_acc'], 
                label=result['model_name'], color=result['color'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 训练损失
    ax = axes[1, 0]
    for result in all_results:
        epochs = range(1, len(result['history']['train_loss']) + 1)
        ax.plot(epochs, result['history']['train_loss'], 
                label=result['model_name'], color=result['color'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 验证损失
    ax = axes[1, 1]
    for result in all_results:
        epochs = range(1, len(result['history']['val_loss']) + 1)
        ax.plot(epochs, result['history']['val_loss'], 
                label=result['model_name'], color=result['color'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存: {output_dir}/training_curves.png")
    plt.close()


def plot_ood_comparison(all_results, output_dir):
    """绘制OOD性能对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = [r['model_name'] for r in all_results]
    test_accs = [r['test_acc'] * 100 for r in all_results]
    ood_accs = [r['ood_acc'] * 100 for r in all_results]
    colors = [r['color'] for r in all_results]
    
    # 1. Test vs OOD 对比
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test (In-Distribution)', 
                   color=colors, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ood_accs, width, label='OOD (Out-of-Distribution)', 
                   color=colors, alpha=0.5, edgecolor='black', hatch='//')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test vs OOD Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 2. OOD准确率排序
    ax = axes[1]
    sorted_results = sorted(all_results, key=lambda x: x['ood_acc'], reverse=True)
    sorted_names = [r['model_name'] for r in sorted_results]
    sorted_ood = [r['ood_acc'] * 100 for r in sorted_results]
    sorted_colors = [r['color'] for r in sorted_results]
    
    bars = ax.barh(sorted_names, sorted_ood, color=sorted_colors, edgecolor='black')
    ax.set_xlabel('OOD Accuracy (%)')
    ax.set_title('OOD Performance Ranking')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, sorted_ood)):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ood_comparison.png', dpi=300, bbox_inches='tight')
    print(f"OOD对比图已保存: {output_dir}/ood_comparison.png")
    plt.close()


def save_results_table(all_results, output_dir):
    """保存结果表格"""
    # 创建DataFrame
    data = []
    for r in all_results:
        data.append({
            'Model': r['model_name'],
            'Description': r['description'],
            'Parameters': f"{r['parameters']:,}",
            'Train Acc': f"{r['train_acc']:.4f}",
            'Val Acc': f"{r['best_val_acc']:.4f}",
            'Test Acc': f"{r['test_acc']:.4f}",
            'Test F1': f"{r['test_f1']:.4f}",
            'OOD Acc': f"{r['ood_acc']:.4f}",
            'OOD F1': f"{r['ood_f1']:.4f}",
            'OOD Gap': f"{(r['test_acc'] - r['ood_acc']):.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # 保存CSV
    csv_path = f'{output_dir}/model_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果表格已保存: {csv_path}")
    
    # 打印表格
    print("\n" + "="*100)
    print("模型对比结果汇总")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


def main():
    """主函数"""
    config = EXPERIMENT_CONFIG
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(f"{config['output_dir']}/checkpoints", exist_ok=True)
    
    print("="*60)
    print("学术预警情感分析 - 模型架构对比实验")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {config['device']}")
    print(f"随机种子: {config['seed']}")
    print(f"Epochs: {config['epochs']}, Batch: {config['batch_size']}, LR: {config['lr']}")
    
    # 加载数据
    print("\n加载数据集...")
    train_df, val_df, test_df, ood_df, tokenizer = load_datasets(config)
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        train_df, val_df, test_df, ood_df, tokenizer, config
    )
    
    # 运行所有实验
    all_results = []
    for model_name, model_config in MODELS.items():
        result = run_experiment(
            model_name, model_config, 
            train_loader, val_loader, test_loader, ood_loader,
            config
        )
        all_results.append(result)
        
        # 保存中间结果
        with open(f"{config['output_dir']}/results.json", 'w', encoding='utf-8') as f:
            # 不保存history以减小文件大小
            save_results = [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
            json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    # 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表...")
    plot_training_curves(all_results, config['output_dir'])
    plot_ood_comparison(all_results, config['output_dir'])
    
    # 保存结果表格
    df = save_results_table(all_results, config['output_dir'])
    
    # 总结
    print("\n" + "="*60)
    print("实验完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n最佳OOD性能: {max(all_results, key=lambda x: x['ood_acc'])['model_name']}")
    print(f"最佳Test性能: {max(all_results, key=lambda x: x['test_acc'])['model_name']}")
    print(f"\n结果保存在: {config['output_dir']}/")
    print("="*60)


if __name__ == '__main__':
    main()
