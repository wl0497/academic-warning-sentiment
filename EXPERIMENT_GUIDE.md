# 学术预警情感分析 - 模型对比实验方案

## 实验目标
对比4种模型架构的性能，证明各组件的贡献：
1. **BERT-Only**: 仅使用BERT输出层
2. **BERT-CNN**: BERT + CNN特征提取
3. **BERT-BiLSTM**: BERT + BiLSTM序列建模
4. **BERT-BiLSTM-CNN**: 完整模型（BERT + BiLSTM + CNN融合）

## 实验设计

### 数据集
- 训练集: 30,000条（V5数据集）
- 同分布测试集: 5,001条
- OOD测试集: 1,800条（独立生成，零重叠）

### 评估指标
| 指标 | 说明 |
|------|------|
| Train Acc | 训练集准确率 |
| Val Acc | 验证集准确率 |
| Test Acc | 同分布测试集准确率 |
| OOD Acc | 分布外测试集准确率（关键指标）|
| F1-Score | 宏平均F1 |
| 参数量 | 可训练参数数量 |

### 公平对比原则
- 相同BERT基础模型（bert-base-chinese）
- 相同训练超参数（lr=2e-5, epochs=15, batch=16）
- 相同数据增强策略
- 相同随机种子（确保可复现）

## 预期结果
| 模型 | 预期OOD Acc | 说明 |
|------|-------------|------|
| BERT-Only | ~85% | 基线模型 |
| BERT-CNN | ~88% | CNN捕获局部特征 |
| BERT-BiLSTM | ~90% | BiLSTM捕获序列依赖 |
| BERT-BiLSTM-CNN | ~95%+ | 融合两种特征提取方式 |

## 运行步骤

### 1. 准备环境
```bash
cd academic-warning-sentiment
# 确保PyTorch已安装
python -c "import torch; print(torch.__version__)"
```

### 2. 运行对比实验
```bash
python model/compare_models.py
```

### 3. 查看结果
实验完成后，结果保存在：
- `model/comparison_results/results.json` - 详细指标
- `model/comparison_results/training_curves.png` - 训练曲线对比
- `model/comparison_results/ood_comparison.png` - OOD性能对比
- `model/comparison_results/model_comparison.csv` - 汇总表格

### 4. 生成论文图表
```bash
python model/generate_paper_figures.py
```
生成学术风格的图表，可直接用于论文。

## 文件说明

| 文件 | 用途 |
|------|------|
| `compare_models.py` | 主实验脚本，顺序训练4个模型 |
| `model_configs.py` | 4种模型架构定义 |
| `train_utils.py` | 统一训练流程（确保公平对比）|
| `generate_paper_figures.py` | 生成论文用图表 |

## 注意事项
1. **显存要求**: RTX 4060 8GB可以运行，但batch size保持16
2. **训练时间**: 4个模型 × 15 epochs ≈ 4-6小时
3. **中断恢复**: 支持从断点继续（检查checkpoint文件）
4. **结果复现**: 固定随机种子，相同环境结果一致
