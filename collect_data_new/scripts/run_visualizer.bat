@echo off
REM H5数据可视化脚本
REM 使用方法: run_visualizer.bat [数据路径]

set DATA_PATH=%1
if "%DATA_PATH%"=="" set DATA_PATH=.\carla_data

echo ========================================
echo H5数据可视化工具
echo ========================================
echo 数据路径: %DATA_PATH%
echo.

cd /d %~dp0\..\..
python -m collect_data_new.scripts.visualize_data --dir %DATA_PATH%

pause
