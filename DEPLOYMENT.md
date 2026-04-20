# 学术预警系统 - 部署指南

## 系统架构

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Vue前端       │ ───▶ │  SpringBoot     │ ───▶ │  Flask推理      │
│  (端口3000)     │      │  (端口8080)     │      │  (端口5000)     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                │
                                ▼
                         ┌─────────────────┐
                         │  MySQL/H2 DB    │
                         └─────────────────┘
```

---

## 一、启动Flask推理服务

```powershell
# 进入模型目录
cd C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\model

# 激活虚拟环境（如有）
# .\venv\Scripts\Activate

# 启动Flask服务
python flask_app.py
```

**验证：** 浏览器访问 http://localhost:5000/health
应返回: `{"status": "ok", "model": "BERT-BiLSTM-CNN v5"}`

---

## 二、启动SpringBoot后端

### 方式A：使用Maven Wrapper（推荐）

```powershell
cd C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\backend

# 开发模式（H2数据库，无需MySQL）
.\mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=dev

# 生产模式（需MySQL）
.\mvnw.cmd spring-boot:run
```

### 方式B：使用IDEA

1. 打开 `backend` 项目
2. 运行 `Application.java`
3. VM options: `-Dspring.profiles.active=dev`（开发模式）

**验证：** 浏览器访问 http://localhost:8080/api/predict/statistics

---

## 三、启动Vue前端

```powershell
cd C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\frontend

# 安装依赖（首次）
npm install

# 启动开发服务器
npm run dev
```

**访问：** 浏览器打开 http://localhost:3000

---

## 四、API接口说明

### Flask推理服务 (端口5000)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/predict` | POST | 单条预测 `{text: "..."}` |
| `/predict/batch` | POST | 批量预测 `{texts: ["...", "..."]}` |
| `/analyze` | POST | 分析+预警建议 |

### SpringBoot后端 (端口8080)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/predict` | POST | 单条预测 |
| `/api/predict/batch` | POST | 批量预测 |
| `/api/predict/statistics` | GET | 统计数据 |
| `/api/warning` | GET | 预警列表 |
| `/api/warning` | POST | 创建预警 |
| `/api/warning/{id}/handle` | POST | 处理预警 |
| `/api/dashboard/stats` | GET | 仪表盘统计 |

---

## 五、数据集位置

- **训练集：** `data/dataset/sentiment_dataset_v5_train.csv` (30000条)
- **测试集：** `data/dataset/sentiment_dataset_v5_test.csv` (5001条，OOD)
- **V5模型：** `model/checkpoints/best_model_v5.pt`

---

## 六、模型性能

| 版本 | Train Acc | Val Acc | OOD Acc | 说明 |
|------|-----------|---------|---------|------|
| V5 | 99.5% | 100% | **97.5%** | 最终版本 |
| V4 | 97.5% | 100% | 87.4% | 消融实验 |
| V3 | 99.5% | 100% | N/A | 数据泄漏 |

---

## 七、常见问题

### 1. Flask启动失败
```powershell
# 检查依赖
pip install flask flask-cors torch transformers

# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. SpringBoot启动失败
```powershell
# 清理并重新编译
.\mvnw.cmd clean compile

# 检查JDK版本（需要JDK 8+）
java -version
```

### 3. Vue空白页
```powershell
# 检查router配置
# 确保 main.js 包含 import router 和 app.use(router)
```

### 4. 跨域问题
- Flask已配置CORS
- SpringBoot已配置@CrossOrigin

---

## 八、一键启动脚本（Windows）

创建 `start_all.bat`:

```batch
@echo off
echo Starting Academic Warning System...

start "Flask" cmd /k "cd C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\model && python flask_app.py"
timeout /t 10

start "SpringBoot" cmd /k "cd C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\backend && .\mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=dev"
timeout /t 30

start "Vue" cmd /k "cd C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment\frontend && npm run dev"

echo All services starting...
echo Flask: http://localhost:5000
echo SpringBoot: http://localhost:8080
echo Vue: http://localhost:3000
pause
```
