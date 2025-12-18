#!/bin/bash
# ============================================================
# TurtleBot 数据收集启动脚本
# ============================================================
#
# 功能: 一键启动数据收集所需的所有 ROS 节点
#
# 使用方法:
#   ./start_collect.sh              # 默认启动所有组件
#   ./start_collect.sh --no-joy     # 不启动手柄
#   ./start_collect.sh --no-kinect  # 不启动 Kinect
#   ./start_collect.sh --help       # 显示帮助
#
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
ENABLE_JOY=true
ENABLE_KINECT=true
JOY_DEV="/dev/input/js0"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-joy)
            ENABLE_JOY=false
            shift
            ;;
        --no-kinect)
            ENABLE_KINECT=false
            shift
            ;;
        --joy-dev)
            JOY_DEV="$2"
            shift 2
            ;;
        --help|-h)
            echo "TurtleBot 数据收集启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --no-joy        不启动手柄驱动"
            echo "  --no-kinect     不启动 Kinect 驱动"
            echo "  --joy-dev DEV   指定手柄设备 (默认: /dev/input/js0)"
            echo "  --help, -h      显示此帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 打印启动信息
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       TurtleBot 数据收集环境启动脚本${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# 检查 ROS 环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}错误: ROS 环境未初始化${NC}"
    echo "请先运行: source /opt/ros/<distro>/setup.bash"
    exit 1
fi
echo -e "${GREEN}✓ ROS 环境: $ROS_DISTRO${NC}"

# 检查 roscore
if ! rostopic list &>/dev/null; then
    echo -e "${YELLOW}启动 roscore...${NC}"
    roscore &
    sleep 2
fi
echo -e "${GREEN}✓ roscore 运行中${NC}"

# 启动 TurtleBot 底盘
echo ""
echo -e "${YELLOW}[1/3] 启动 TurtleBot 底盘...${NC}"
roslaunch turtlebot_bringup minimal.launch &
PIDS="$!"
sleep 3
echo -e "${GREEN}✓ TurtleBot 底盘已启动${NC}"

# 启动 Kinect
if [ "$ENABLE_KINECT" = true ]; then
    echo ""
    echo -e "${YELLOW}[2/3] 启动 Kinect 摄像头...${NC}"
    roslaunch freenect_launch freenect.launch &
    PIDS="$PIDS $!"
    sleep 3
    echo -e "${GREEN}✓ Kinect 已启动${NC}"
else
    echo ""
    echo -e "${YELLOW}[2/3] 跳过 Kinect (--no-kinect)${NC}"
fi

# 启动手柄
if [ "$ENABLE_JOY" = true ]; then
    echo ""
    echo -e "${YELLOW}[3/3] 启动手柄驱动...${NC}"
    
    # 检查手柄设备
    if [ ! -e "$JOY_DEV" ]; then
        echo -e "${RED}警告: 手柄设备 $JOY_DEV 不存在${NC}"
        echo "请检查手柄是否连接，或使用 --joy-dev 指定正确的设备"
    else
        rosrun joy joy_node _dev:="$JOY_DEV" &
        PIDS="$PIDS $!"
        sleep 1
        echo -e "${GREEN}✓ 手柄已启动 ($JOY_DEV)${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}[3/3] 跳过手柄 (--no-joy)${NC}"
fi

# 完成
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✓ 数据收集环境启动完成！${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "可用话题:"
echo "  图像: /camera/rgb/image_color"
echo "  控制: /mobile_base/commands/velocity"
echo "  里程计: /odom"
echo "  手柄: /joy"
echo ""
echo "现在可以运行数据收集器:"
echo -e "  ${GREEN}python collector.py${NC}"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止所有节点${NC}"

# 等待退出
trap "echo ''; echo '正在停止所有节点...'; kill $PIDS 2>/dev/null; exit 0" SIGINT SIGTERM
wait
