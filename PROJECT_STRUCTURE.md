# ===== 完整项目文件清单 =====
# 更新时间: 2026-04-18
# 项目: 基于Bert-BiLSTM-CNN的高校大学生学业预警情感分析系统

academic-warning-sentiment/
│
├── project-plan.md                          # 项目全流程方案
├── requirements.txt                         # Python依赖
│
├── model/                                   # 深度学习模型模块
│   ├── config.py                            # 模型配置（超参数、标签映射）
│   ├── bert_bilstm_cnn.py                   # Bert-BiLSTM-CNN融合模型架构
│   ├── utils.py                             # 工具函数（数据加载、评估指标）
│   ├── train.py                             # 训练脚本（含EarlyStopping）
│   ├── evaluate.py                          # 评估脚本（混淆矩阵、分类报告）
│   ├── predict.py                           # 推理脚本（单条/批量预测）
│   └── saved/                               # 训练好的模型权重（需训练后生成）
│
├── scripts/                                 # 数据处理脚本
│   ├── preprocess.py                        # 数据预处理流水线（清洗+分词+去停用词）
│   └── build_dict.py                        # 学业领域情感词典构建
│
├── data/                                    # 数据目录
│   ├── raw/
│   │   ├── questionnaire/                   # 问卷原始数据（3200+条）
│   │   └── social_media/                    # 社交文本爬取数据（15000+条）
│   ├── processed/
│   │   ├── cleaned/                         # 清洗后数据
│   │   └── labeled/                         # 标注后数据（5000条）
│   └── stopwords/                           # 停用词表
│
├── model_service.py                         # Flask模型推理API（端口5000）
│
├── backend/                                 # SpringBoot后端
│   ├── pom.xml                              # Maven依赖配置
│   └── src/main/
│       ├── java/com/warning/
│       │   ├── Application.java             # 启动类
│       │   ├── config/
│       │   │   └── MyBatisPlusConfig.java   # MyBatis-Plus分页配置
│       │   ├── entity/
│       │   │   ├── WarningRecord.java       # 预警记录实体
│       │   │   └── PredictResult.java       # 预测结果实体
│       │   ├── mapper/
│       │   │   ├── WarningRecordMapper.java
│       │   │   └── PredictResultMapper.java
│       │   ├── service/
│       │   │   └── PredictService.java      # 预测服务（调用Python API）
│       │   └── controller/
│       │       ├── PredictController.java   # 预测REST API
│       │       └── WarningController.java   # 预警管理REST API
│       └── resources/
│           ├── application.yml              # 配置文件
│           └── schema.sql                   # 数据库初始化脚本
│
├── frontend/                                # Vue前端
│   ├── package.json                         # 依赖配置
│   ├── vite.config.js                       # Vite构建配置（含API代理）
│   ├── index.html                           # 入口HTML
│   └── src/
│       ├── main.js                          # Vue入口
│       ├── App.vue                          # 根组件（布局+导航）
│       ├── router/index.js                  # 路由配置
│       ├── api/index.js                     # Axios API封装
│       └── views/
│           ├── Dashboard.vue                # 数据概览（4个ECharts图表）
│           ├── Predict.vue                  # 文本情感分析页
│           ├── WarningList.vue              # 预警管理列表页
│           └── ModelInfo.vue                # 模型信息展示页
│
├── deployment/                              # Docker部署
│   ├── docker-compose.yml                   # 一键编排（MySQL+模型+后端+前端）
│   ├── Dockerfile.python                    # Python模型服务镜像
│   ├── Dockerfile.backend                   # SpringBoot后端镜像
│   ├── Dockerfile.frontend                  # Vue前端Nginx镜像
│   └── nginx.conf                           # Nginx反向代理配置
│
└── .dockerignore                            # Docker忽略文件
