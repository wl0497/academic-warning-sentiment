# 学术预警系统 - 情感分类 AI 全栈项目

基于 Bert-BiLSTM-CNN 的学术预警情感分类系统，集成 Flask 推理服务 + SpringBoot 后端 + Vue 前端。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Vue](https://img.shields.io/badge/Vue-3-green)
![SpringBoot](https://img.shields.io/badge/SpringBoot-3.x-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      Vue 前端 (3003)                      │
│    Dashboard / 批量分析 / 情感预测 / 预警管理 / 模型演示    │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP API
┌──────────────────────▼──────────────────────────────────┐
│               SpringBoot 后端 (8080)                      │
│     DashboardAPI / BatchAPI / PredictAPI / WarningAPI     │
│                 ↓ HTTP 请求                                │
│              Flask 推理服务 (5000)                        │
└──────────────────────┬──────────────────────────────────┘
                       │ PyTorch 模型
┌──────────────────────▼──────────────────────────────────┐
│             Bert-BiLSTM-CNN 模型                          │
│  BERT(freeze8) + BiLSTM(64) + CNN(64×3) + Fusion(128)     │
│  Test Acc=97.66%  OOD=23.32%  鲁棒性=96.80%             │
└─────────────────────────────────────────────────────────┘
```

## 核心功能

- **情感分类预测**：单条文本情感分析（正面/中性/负面）+ 置信度
- **批量分析**：支持 CSV/Excel 批量上传，自动生成预警记录
- **预警管理**：预警列表查询、处理状态更新、单条/批量删除
- **数据概览**：情感分布饼图、预警趋势图、学院分布热力图
- **模型架构演示**：动态可视化 BERT-BiLSTM-CNN 数据流

## 快速启动

### 方式一：一键启动（需先安装依赖）

```bash
# 安装后端依赖（需 JDK 8+）
cd backend
mvn spring-boot:run

# 新开终端，安装前端依赖（需 Node.js）
cd frontend
npm install
npm run dev

# 新开终端，启动推理服务（需 Python 3.10 + PyTorch）
cd model
pip install -r requirements.txt
python flask_app.py
```

### 方式二：Docker 部署

```bash
cd docker
docker-compose up -d
```

## 项目结构

```
academic-warning-sentiment/
├── model/                      # 深度学习模型
│   ├── flask_app.py            # Flask 推理服务（端口 5000）
│   ├── bert_bilstm_cnn_comparison.py  # 模型架构定义
│   ├── train_v5.py             # 最终训练脚本（V5，最佳性能）
│   ├── config.py               # 模型配置
│   ├── train_utils.py          # 训练工具函数
│   ├── compare_models.py       # 模型对比实验脚本
│   ├── ablation_study.py       # 消融实验脚本
│   ├── generate_paper_figures.py  # 论文图表生成
│   └── checkpoints/            # 模型权重（需单独下载）
│       └── BERT_BiLSTM_CNN_best.pt
├── backend/                    # SpringBoot 后端（端口 8080）
│   ├── pom.xml
│   └── src/main/java/com/warning/
│       ├── controller/         # REST API 控制器
│       ├── service/           # 业务逻辑
│       ├── entity/            # 数据实体
│       └── mapper/            # 数据库映射
├── frontend/                   # Vue 3 前端（端口 3003）
│   ├── src/
│   │   ├── views/             # 页面组件
│   │   ├── components/        # 可复用组件
│   │   ├── api/               # API 请求封装
│   │   └── router/            # 路由配置
│   └── package.json
├── data/                       # 数据集
│   └── dataset/
│       ├── sentiment_dataset_v5_train.csv   # 训练集（30000条）
│       └── sentiment_dataset_v5_test.csv   # 测试集（5001条）
├── docker/                     # Docker 部署配置
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.python
│   ├── nginx.conf
│   └── README.md
└── docs/                       # 项目文档
    ├── ARCHITECTURE.md         # 系统架构文档
    └── EXECUTION_PLAN.md       # 执行计划
```

## 模型性能

| 模型 | 测试集 Acc | OOD Acc | 鲁棒性 |
|------|-----------|---------|--------|
| BERT-Only | 99.76% | 20.22% | 99.00% |
| BERT-CNN | 95.60% | 10.06% | 93.40% |
| BERT-BiLSTM | 97.72% | 15.12% | 95.00% |
| **BERT-BiLSTM-CNN** | **97.66%** | **23.32%** | **96.80%** |

> OOD（Out-of-Distribution）反映模型在真实场景下的泛化能力，是最重要的指标

## 技术栈

**后端**：SpringBoot 3.x | MyBatis-Plus | H2（开发）/ MySQL（生产）| fastjson2

**前端**：Vue 3 | Element Plus | ECharts | Axios | Vite

**模型**：PyTorch 2.0 | Transformers (BERT) | Numpy | Pandas

**部署**：Docker | Nginx | CUDA (GPU推理)

## License

MIT License
