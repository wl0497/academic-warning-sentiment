"""
模型参数配置 v2 — 方案A简化版（防止过拟合）
==========================================
主要改动：
- LSTM隐藏层: 256→64, 层数: 2→1
- CNN滤波器: 256→64
- Dropout: 0.3→0.5
- 冻结BERT底层6层
"""
import torch

class Config:
    # ============ 数据参数 ============
    DATA_PATH = "C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/data/dataset/sentiment_dataset_final.csv"
    MAX_LEN = 128
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    NUM_CLASSES = 3

    # ============ BERT参数 ============
    BERT_MODEL_NAME = "bert-base-chinese"
    BERT_HIDDEN_SIZE = 768
    BERT_DROPOUT = 0.15  # 0.1→0.15

    # ============ BiLSTM参数（方案A: 简化） ============
    LSTM_HIDDEN_SIZE = 64     # 256→64 大幅减少
    LSTM_NUM_LAYERS = 1       # 2→1 减少层数
    LSTM_DROPOUT = 0.5        # 0.3→0.5 增强防过拟合

    # ============ CNN参数（方案A: 简化） ============
    CNN_NUM_FILTERS = 64      # 256→64 大幅减少
    CNN_FILTER_SIZES = [2, 3, 4]
    CNN_DROPOUT = 0.5         # 0.3→0.5

    # ============ 融合层参数（方案A: 简化） ============
    FUSION_HIDDEN_SIZE = 128  # 256→128
    FUSION_DROPOUT = 0.5      # 0.4→0.5

    # ============ 训练参数 ============
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 30
    EARLY_STOP_PATIENCE = 7   # 5→7 给更多机会
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    # ============ 设备 ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 路径 ============
    CHECKPOINT_DIR = "model/checkpoints"
    BEST_MODEL_PATH = "model/checkpoints/best_model.pt"

    # ============ 标签映射 ============
    LABEL_MAP = {"正面": 0, "中性": 1, "负面": 2}
    LABEL_MAP_INV = {0: "正面", 1: "中性", 2: "负面"}
