# 模型权重与数据集

由于模型权重（~400MB）和训练数据集（~2MB）超过 GitHub 100MB 限制，
需要使用 Git LFS 拉取。

## 安装 Git LFS

```bash
# Windows (使用 Chocolatey)
choco install git-lfs

# 或下载安装包：https://git-lfs.github.com
```

## 拉取大文件

```bash
cd academic-warning-sentiment
git lfs install
git lfs pull
```

## 手动下载模型权重

如果 LFS 拉取失败，从以下地址下载并放入对应目录：

- `model/checkpoints/best_model_v5.pt` → 最佳模型（部署用）
- `model/comprehensive_results/checkpoints/BERT_BiLSTM_CNN_best.pt` → 对比实验最佳模型

推荐使用网盘或 HuggingFace 托管。
