@echo off
chcp 65001 >nul
echo ========================================
echo   CARLA 全自动数据收集器
echo ========================================
echo.

REM 切换到脚本所在目录的父目录
cd /d "%~dp0..\.."

REM 激活 conda 环境
echo 正在激活 conda 环境 (study)...
call conda activate study
if errorlevel 1 (
    echo [警告] conda activate 失败，尝试继续...
)

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

REM 运行收集脚本
echo 正在启动数据收集...
echo.
python -m collect_data_new.scripts.run_auto_collection --config collect_data_new/config/auto_collection_config.json %*

echo.
echo ========================================
echo   收集完成
echo ========================================
pause
