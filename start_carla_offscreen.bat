@echo off
chcp 65001 >nul
REM ============================================
REM CARLA 低负载启动脚本
REM 使用离屏渲染 + 低画质，减少 GPU 占用
REM ============================================

REM 设置 CARLA 安装路径（请根据实际情况修改）
set CARLA_PATH=D://CARLA_0.9.16

echo ============================================
echo 正在以低负载模式启动 CARLA...
echo - 离屏渲染 (RenderOffScreen)
echo - 低画质 (quality-level=Low)
echo ============================================

cd /d %CARLA_PATH%
CarlaUE4.exe -RenderOffScreen -quality-level=Low

pause
