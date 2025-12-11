@echo off
REM 数据验证脚本
REM 使用方法: run_verify.bat [数据路径]

set DATA_PATH=%1
if "%DATA_PATH%"=="" set DATA_PATH=E:\datasets\ClearSunset

echo ========================================
echo 数据验证工具
echo ========================================
echo 数据路径: %DATA_PATH%
echo.

cd /d %~dp0\..\..
python -m collect_data_new.scripts.verify_data --data-path %DATA_PATH%

pause
