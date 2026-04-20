"""
模型参数配置 — 严格匹配中期检查报告指标
"""
import torch

class Config:
    # ============ 数据参数 ============
    DATA_PATH = "C:/Users/29258/.qclaw/workspace-agent-66459c61/academic-warning-sentiment/data/dataset/sentiment_dataset_final.csv"  # 新数据集 (10200条, 0%重复)
    MAX_LEN = 128                    # 文本最大长度
    TRAIN_RATIO = 0.8                # 训练集比例
    VAL_RATIO = 0.1                  # 验证集比例
    TEST_RATIO = 0.1                 # 测试集比例
    NUM_CLASSES = 3                  # 情感分类数：正面/中性/负面

    # ============ BERT参数 ============
    BERT_MODEL_NAME = "bert-base-chinese"
    BERT_HIDDEN_SIZE = 768           # bert-base-chinese输出维度
    BERT_DROPOUT = 0.1

    # ============ BiLSTM参数 ============
    LSTM_HIDDEN_SIZE = 256           # BiLSTM隐层维度
    LSTM_NUM_LAYERS = 2              # BiLSTM层数
    LSTM_DROPOUT = 0.3               # Dropout防过拟合

    # ============ CNN参数 ============
    CNN_NUM_FILTERS = 256            # CNN卷积核数量
    CNN_FILTER_SIZES = [2, 3, 4]    # 卷积核尺寸（提取2-gram, 3-gram, 4-gram）
    CNN_DROPOUT = 0.3

    # ============ 融合层参数 ============
    FUSION_HIDDEN_SIZE = 256         # 融合全连接层维度
    FUSION_DROPOUT = 0.4             # 融合层Dropout

    # ============ 训练参数（严格匹配申报书） ============
    BATCH_SIZE = 32                  # 申报书明确指定
    LEARNING_RATE = 2e-5             # 申报书明确指定
    EPOCHS = 30                      # 最大训练轮数
    EARLY_STOP_PATIENCE = 5          # 早停耐心值（申报书要求EarlyStopping）
    WEIGHT_DECAY = 0.01              # 权重衰减
    WARMUP_RATIO = 0.1               # 预热比例

    # ============ 设备 ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 路径 ============
    CHECKPOINT_DIR = "model/checkpoints"
    BEST_MODEL_PATH = "model/checkpoints/best_model.pt"

    # ============ 标签映射 ============
    LABEL_MAP = {"正面": 0, "中性": 1, "负面": 2}
    LABEL_MAP_INV = {0: "正面", 1: "中性", 2: "负面"}
