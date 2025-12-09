@echo off
chcp 65001 >nul
title 测试转弯性能

echo ========================================
echo [测试转弯性能]
echo ========================================
echo.
echo 此脚本将使用优化的参数测试车辆转弯性能
echo.
echo 优化参数:
echo   * 速度: 6 km/h (防止转弯失控)
echo   * 帧率: 20 FPS (更平滑的控制)
echo   * 路线: 短距离测试 (100-300米)
echo   * 帧数: 200帧 (快速验证)
echo.
echo ========================================
echo.

pause

echo 开始测试...
python auto_full_town_collection.py --target-speed 6.0 --fps 20 --strategy smart --min-distance 100 --max-distance 300 --frames-per-route 200

echo.
echo ========================================
echo [测试完成]
echo ========================================
echo.
echo 请检查:
echo   1. 车辆是否在转弯时保持在车道内
echo   2. 是否没有冲上马路牙子
echo   3. 转向是否平滑
echo.
echo 如果仍有问题，可以进一步降低速度到 5 km/h:
echo   python auto_full_town_collection.py --target-speed 5.0 --fps 20
echo.
pause
