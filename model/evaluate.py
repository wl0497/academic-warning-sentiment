"""
评估脚本 — 生成完整评估报告
=============================
输出：Accuracy, Precision, Recall, F1, 混淆矩阵, 分类报告
"""
import torch
import numpy as np
from transformers import BertTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from config import Config
from bert_bilstm_cnn import BertBiLSTMCNN
from utils import load_data, split_data, create_data_loader


def evaluate(model, data_loader, device, label_names=None):
    """
    完整模型评估
    
    Returns:
        dict: 包含所有评估指标
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    if label_names is None:
        label_names = ['正面', '中性', '负面']
    
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    return results


def print_evaluation_report(results):
    """打印格式化评估报告"""
    print("=" * 70)
    print("       学业预警情感分析模型 — 评估报告")
    print("=" * 70)
    print(f"\n📊 核心指标（申报书目标: Acc≥88.4%, F1≥0.86）")
    print(f"   Accuracy:  {results['accuracy']:.4f}  {'✅' if results['accuracy'] >= 0.884 else '❌'}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1']:.4f}  {'✅' if results['f1'] >= 0.86 else '❌'}")
    
    print(f"\n📋 详细分类报告:")
    print(results['classification_report'])
    
    print(f"\n🔢 混淆矩阵:")
    print(results['confusion_matrix'])
    
    # 达标判定
    print(f"\n{'='*70}")
    acc_pass = results['accuracy'] >= 0.884
    f1_pass = results['f1'] >= 0.86
    if acc_pass and f1_pass:
        print("🎉 全部指标达标！模型满足申报书要求。")
    else:
        print("⚠️ 部分指标未达标，需要进一步调优。")
        if not acc_pass:
            print(f"   - Accuracy差值: {0.884 - results['accuracy']:.4f}")
        if not f1_pass:
            print(f"   - F1差值: {0.86 - results['f1']:.4f}")


def main():
    config = Config()
    device = config.DEVICE
    
    # 加载数据
    df = load_data(config.DATA_PATH)
    _, _, test_df = split_data(df)
    
    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = BertBiLSTMCNN(config).to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    test_loader = create_data_loader(
        test_df, tokenizer, config.MAX_LEN, config.BATCH_SIZE, shuffle=False
    )
    results = evaluate(model, test_loader, device)
    print_evaluation_report(results)
    
    return results


if __name__ == "__main__":
    main()
