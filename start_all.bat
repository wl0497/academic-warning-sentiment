@echo off
chcp 65001 >nul
echo ================================================
echo   学术预警系统 - 一键启动
echo ================================================
echo.

set "BASE=C:\Users\29258\.qclaw\workspace-agent-66459c61\academic-warning-sentiment"

echo [1/3] 启动 Flask 推理服务 (端口5000)...
start "Flask-BestModel" cmd /k "cd /d "%BASE%\model" && python flask_app.py"
timeout /t 15 /nobreak >nul

echo [2/3] 启动 SpringBoot 后端 (端口8080)...
start "SpringBoot" cmd /k "cd /d "%BASE%\backend" && .\mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=dev"
timeout /t 40 /nobreak >nul

echo [3/3] 启动 Vue 前端 (端口3003)...
start "Vue-Frontend" cmd /k "cd /d "%BASE%\frontend" && npm run dev"

echo.
echo ================================================
echo   所有服务已启动
echo ================================================
echo.
echo   Flask推理:   http://localhost:5000
echo   SpringBoot:  http://localhost:8080
echo   Vue前端:     http://localhost:3003
echo.
echo   当前模型: BERT-BiLSTM-CNN (对比实验最佳)
echo   性能: Test=97.66%% OOD=23.32%% 鲁棒=96.80%%
echo ================================================
pause
