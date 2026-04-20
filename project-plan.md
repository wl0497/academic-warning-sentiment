# 基于Bert-BiLSTM-CNN的高校大学生学业预警情感分析 — 全流程项目计划

## 📋 项目元信息

| 项目 | 内容 |
|------|------|
| 项目名称 | 基于Bert-BiLSTM-CNN模型的高校大学生学业预警情感分析 |
| 学校 | 淮南师范学院 |
| 负责人 | 王亮（计算机学院，学号2308010227） |
| 指导教师 | 李晓燕（讲师，研究方向NLP） |
| 团队成员 | 刘子杰(模型优化)、洪明(数据采集)、邹红亮(后端开发)、刘雨蕾(文档撰写) |
| 研究周期 | 2025.05 — 2026.09 |
| 当前进度 | 65%（中期已过，系统开发与集成阶段） |

## 🎯 核心指标（来自申报书+中期检查报告，必须达标）

| 指标 | 目标值 | 来源 |
|------|--------|------|
| 模型准确率(Accuracy) | ≥ 88.4% | 中期报告已达成 |
| F1值 | ≥ 0.86 | 中期报告已达成 |
| 问卷有效样本 | 3200+份 | 中期报告已达成 |
| 社交平台文本 | 15000+条 | 中期报告已达成 |
| 标注语料 | ~5000条 | 中期报告已达成 |
| 结构化数据集 | ~40MB CSV | 中期报告已达成 |
| 训练参数 | batch=32, lr=2e-5 | 中期报告 |
| 防过拟合 | Dropout + EarlyStopping | 中期报告 |

## 📁 项目目录结构

```
academic-warning-sentiment/
├── README.md                       # 项目总览
├── docs/                           # 文档
│   ├── 申报书要点提取.md
│   ├── 中期检查报告要点提取.md
│   ├── 实验报告.md
│   └── 部署文档.md
│
├── data/                           # 数据模块
│   ├── raw/                        # 原始数据
│   │   ├── questionnaire/          # 问卷数据
│   │   └── social_media/           # 社交平台爬取数据
│   ├── processed/                  # 预处理后数据
│   │   ├── segmented/              # 分词后数据
│   │   ├── cleaned/                # 清洗后数据
│   │   └── labeled/                # 标注后数据
│   ├── sentiment_dict/             # 学业领域情感词典
│   ├── stopwords/                  # 停用词表
│   └── dataset/                    # 最终数据集(CSV~40MB)
│
├── model/                          # 模型模块
│   ├── bert_bilstm_cnn.py          # Bert-BiLSTM-CNN融合模型
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   ├── predict.py                  # 推理脚本
│   ├── config.py                   # 模型参数配置
│   ├── utils.py                    # 工具函数
│   ├── checkpoints/                # 模型权重保存
│   └── ablation/                   # 消融实验
│       ├── bert_only.py
│       ├── bilstm_only.py
│       ├── cnn_only.py
│       └── bert_bilstm.py
│
├── backend/                        # 后端模块(SpringBoot)
│   ├── pom.xml
│   ├── src/main/java/com/warning/
│   │   ├── Application.java
│   │   ├── controller/             # RESTful API
│   │   │   ├── PredictController.java
│   │   │   ├── BatchController.java
│   │   │   └── WarningController.java
│   │   ├── service/                # 业务逻辑
│   │   │   ├── PredictService.java
│   │   │   └── WarningService.java
│   │   ├── entity/                 # 实体类
│   │   ├── mapper/                 # MyBatis映射
│   │   └── config/                 # 配置类
│   └── src/main/resources/
│       ├── application.yml
│       └── mapper/
│
├── frontend/                       # 前端模块(Vue+ECharts)
│   ├── package.json
│   ├── src/
│   │   ├── App.vue
│   │   ├── views/
│   │   │   ├── Dashboard.vue       # 预警大屏
│   │   │   ├── Predict.vue         # 文本预测
│   │   │   ├── BatchAnalysis.vue   # 批量分析
│   │   │   └── WarningList.vue     # 预警列表
│   │   ├── components/
│   │   │   ├── HeatMap.vue         # 学业情感热力图
│   │   │   ├── EmotionCurve.vue    # 情绪波动曲线
│   │   │   ├── SentimentPie.vue    # 情感分布饼图
│   │   │   └── WarningTable.vue    # 预警数据表
│   │   ├── api/                    # API调用
│   │   └── utils/
│   └── public/
│
├── docker/                         # Docker部署
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.model
│   └── nginx.conf
│
├── scripts/                        # 工具脚本
│   ├── data_collection.py          # 数据采集(问卷+爬虫)
│   ├── preprocess.py               # 数据预处理
│   ├── build_dict.py               # 情感词典构建
│   └── stress_test.py              # 压力测试
│
└── requirements.txt                # Python依赖
```

## 🔄 阶段任务与执行顺序

### 阶段1：数据采集与预处理 ✅ (已完成，需复现)
1. 问卷调研3200+份 → 问卷星导出CSV
2. Scrapy爬虫采集微博/知乎/贴吧 → 15000+条
3. Jieba分词 + 哈工大停用词表 → 去停用词
4. 数据去重、去噪（广告、乱码）
5. 人工标注5000条 → 三分类（正面/中性/负面）
6. 构建学业领域情感词典
7. 输出40MB结构化CSV数据集

### 阶段2：模型构建与训练 ✅ (已完成，需复现)
1. Bert预训练模型加载（bert-base-chinese）
2. BiLSTM时序特征提取层
3. CNN局部关键特征提取层
4. 融合层设计与输出
5. 训练参数：batch=32, lr=2e-5
6. Dropout + EarlyStopping防过拟合
7. 消融实验（纯Bert/BiLSTM/CNN对比）
8. 目标：Acc≥88.4%, F1≥0.86

### 阶段3：全栈系统开发 🔧 (当前攻坚阶段)
1. SpringBoot后端搭建
   - MySQL数据库设计（用户表、预警记录表、预测结果表）
   - RESTful API：文本预测/批量分析/预警推送/结果查询
   - 模型推理服务封装（Python Flask API → Java调用）
2. Vue前端开发
   - 预警大屏：情感热力图 + 情绪波动曲线 + 情感分布
   - 文本预测页：单条输入 → 实时预测
   - 批量分析页：文件上传 → 批量处理
   - 预警列表页：预警记录管理

### 阶段4：部署上线
1. Docker容器化（3个容器：model-service, backend, frontend）
2. docker-compose编排
3. Nginx反向代理
4. 云服务器部署
5. 压力测试与性能优化

### 阶段5：试点与结题
1. 计算机学院2-3个班级试点
2. 收集反馈，校准模型参数
3. 结题报告 + 答辩PPT
4. 论文投稿 + 软著申请

## ⚙️ 技术栈

| 层次 | 技术 |
|------|------|
| 深度学习 | PyTorch, Transformers(HuggingFace), bert-base-chinese |
| NLP预处理 | Jieba, 哈工大停用词表 |
| 后端 | SpringBoot 2.7+, MyBatis-Plus, MySQL 8.0 |
| 模型服务 | Flask + Gunicorn (Python推理API) |
| 前端 | Vue 3, ECharts 5, Axios, Element Plus |
| 部署 | Docker, docker-compose, Nginx |
| 数据采集 | Scrapy, 问卷星导出 |

## 📌 时间线（下一阶段2026.01-2026.06）

| 时间 | 任务 |
|------|------|
| 2026.01-2026.03 | 系统开发与集成（前端大屏+后端API+模型部署） |
| 2026.03-2026.04 | 试点运行与反馈收集 |
| 2026.04-2026.06 | 结题报告、论文、软著、答辩 |
