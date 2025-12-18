#!/bin/bash
# ============================================================
# TurtleBot 数据收集启动脚本
# 在独立终端窗口中启动所有 ROS 节点
# ============================================================

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       TurtleBot 数据收集环境启动${NC}"
echo -e "${BLUE}============================================================${NC}"

# ROS 环境
ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
WS_SETUP="$HOME/catkin_ws/devel/setup.bash"

SOURCE_CMD="source $ROS_SETUP"
[ -f "$WS_SETUP" ] && SOURCE_CMD="$SOURCE_CMD && source $WS_SETUP"

# 1. TurtleBot 底盘
echo -e "${YELLOW}[1/3] 启动 TurtleBot 底盘...${NC}"
gnome-terminal --title="[1] TurtleBot Base" -- bash -c "$SOURCE_CMD && roslaunch turtlebot_bringup minimal.launch; exec bash"
sleep 3
echo -e "${GREEN}✓ 底盘终端已打开${NC}"

# 2. Kinect 摄像头
echo -e "${YELLOW}[2/3] 启动 Kinect 摄像头...${NC}"
gnome-terminal --title="[2] Kinect Camera" -- bash -c "$SOURCE_CMD && roslaunch freenect_launch freenect.launch; exec bash"
sleep 2
echo -e "${GREEN}✓ Kinect 终端已打开${NC}"

# 3. 手柄
echo -e "${YELLOW}[3/3] 启动手柄驱动...${NC}"
gnome-terminal --title="[3] Joystick" -- bash -c "$SOURCE_CMD && rosrun joy joy_node; exec bash"
sleep 1
echo -e "${GREEN}✓ 手柄终端已打开${NC}"

echo ""
echo -e "${GREEN}✓ 所有终端已启动！${NC}"
echo ""
echo "现在可以运行: python collector.py"
