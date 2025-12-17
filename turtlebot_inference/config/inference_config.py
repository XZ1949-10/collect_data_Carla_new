#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 推理配置参数
'''

# ==================== 图像参数 ====================
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 88

# 图像裁剪参数（与数据收集保持一致）
CROP_TOP_RATIO = 0.3
CROP_BOTTOM_RATIO = 0.0

# ==================== 速度参数 ====================
# 与 CARLA 训练保持一致
SPEED_NORMALIZATION_KMH = 25.0

# ==================== TurtleBot 参数 ====================
TURTLEBOT_PARAMS = {
    'burger': {'max_linear': 0.22, 'max_angular': 2.84},
    'waffle': {'max_linear': 0.26, 'max_angular': 1.82},
    'waffle_pi': {'max_linear': 0.26, 'max_angular': 1.82},
    'turtlebot1': {'max_linear': 0.5, 'max_angular': 1.0},
    'turtlebot2': {'max_linear': 0.7, 'max_angular': 1.0},
    'kobuki': {'max_linear': 0.7, 'max_angular': 1.0},
}

# ==================== 导航命令映射 ====================
COMMAND_FOLLOW = 2
COMMAND_LEFT = 3
COMMAND_RIGHT = 4
COMMAND_STRAIGHT = 5

COMMAND_NAMES = {
    2: 'Follow',
    3: 'Left',
    4: 'Right',
    5: 'Straight'
}

# ==================== 控制参数 ====================
DEFAULT_COMMAND = COMMAND_FOLLOW  # 默认跟车命令
CONTROL_RATE_HZ = 10              # 控制频率
