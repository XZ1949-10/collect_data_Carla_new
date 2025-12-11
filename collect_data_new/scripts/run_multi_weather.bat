@echo off
chcp 65001 >nul
echo ========================================
echo   CARLA 多天气数据收集器
echo ========================================
echo.

REM 切换到脚本所在目录的父目录
cd /d "%~dp0..\.."

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

echo 可用的天气预设:
echo   basic      - 基础组合（4种）
echo   all_noon   - 所有正午天气（7种）
echo   all_sunset - 所有日落天气（7种）
echo   all_night  - 所有夜晚天气（7种）
echo   clear_all  - 所有晴朗天气（3种）
echo   rain_all   - 所有雨天（9种）
echo   full       - 完整组合（13种）
echo   complete   - 所有天气（22种）
echo.

set /p PRESET="请输入天气预设名称 [默认: basic]: "
if "%PRESET%"=="" set PRESET=basic

set /p SAVE_PATH="请输入保存路径 [默认: ./multi_weather_data]: "
if "%SAVE_PATH%"=="" set SAVE_PATH=./multi_weather_data

echo.
echo 正在启动多天气数据收集...
echo 天气预设: %PRESET%
echo 保存路径: %SAVE_PATH%
echo.

python -m collect_data_new.scripts.run_auto_collection --multi-weather %PRESET% --save-path %SAVE_PATH%

echo.
echo ========================================
echo   多天气收集完成
echo ========================================
pause
