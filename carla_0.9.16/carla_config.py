#!/usr/bin/env python
# coding=utf-8
'''
CARLA 推理配置模块
包含所有常量和配置参数
'''

# ==================== 可解释性仪表板参数 ====================
# 第一行热力图使用的卷积层索引
# -1: 最后一层（分辨率最低，语义最高）
# -3: 倒数第3层（推荐，平衡分辨率和语义）
# -5: 倒数第5层（更高分辨率）
DASHBOARD_ROW1_LAYER_INDEX = -3

# 第二行多层级热力图使用的卷积层索引列表
# 默认: 所有8个卷积层 [-8, -7, -6, -5, -4, -3, -2, -1] 分别对应 Layer 1-8
# 从浅层（高分辨率）到深层（高语义）
DASHBOARD_ROW2_LAYER_INDICES = [-8, -7, -6, -5, -4, -3, -2, -1]

# 积分梯度 (Integrated Gradients) 的积分步数
# 步数越大精度越高，但计算时间越长
# 推荐值: 30-50
DASHBOARD_IG_STEPS = 30

# 历史记录最大帧数，None表示记录所有帧（无限制）
DASHBOARD_HISTORY_MAX_FRAMES = None

# ==================== 图像参数 ====================
# 模型输入尺寸
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 88

# 摄像头采集分辨率（高分辨率，用于裁剪）
CAMERA_RAW_WIDTH = 800
CAMERA_RAW_HEIGHT = 600
CAMERA_FOV = 90  # 与训练时保持一致

# 摄像头位置（相对于车辆）
CAMERA_LOCATION_X = 2.0
CAMERA_LOCATION_Z = 1.4
CAMERA_PITCH = -15

# ==================== 速度参数 ====================
# 注意：必须与训练时保持一致！
# 这个其实就是 25km/h
SPEED_NORMALIZATION_MPS = 25.0
MAX_SPEED_KMH = SPEED_NORMALIZATION_MPS   # 用于显示

# 最高速度限制（后处理器使用）
MAX_SPEED_LIMIT_MPS = 25.0  # 最高速度限制（km/h），
MAX_SPEED_LIMIT_KMH = MAX_SPEED_LIMIT_MPS   # 用于显示，约 36 km/h

# ==================== 后处理器默认配置 ====================
POST_PROCESSOR_DEFAULT_CONFIG = {
    'enable_brake_denoising': True,        # 启用刹车去噪
    'enable_throttle_brake_mutex': True,   # 启用油门刹车互斥
    'enable_speed_limit': True,            # 启用速度限制
    'enable_turning_slowdown': True,       # 启用转弯减速
    'enable_avoid_stopping': False          # 启用避免停车
}

# ==================== 同步模式参数 ====================
# 每次 tick 推进的模拟器时间（秒）
# 0.05 = 20 FPS，时间流速接近现实
SYNC_MODE_DELTA_SECONDS = 0.05  # 推荐值，与现实时间同步

# ==================== 导航命令映射 ====================
# 与训练数据一致：2-5编码
COMMAND_NAMES_CN = {
    2: '跟车',
    3: '左转',
    4: '右转',
    5: '直行'
}

COMMAND_NAMES_EN = {
    2: 'Follow',
    3: 'Left',
    4: 'Right',
    5: 'Straight'
}

# ==================== 车辆生成参数 ====================
MAX_SPAWN_ATTEMPTS = 10           # 最大生成尝试次数
SPAWN_STABILIZE_TICKS = 3         # 生成后等待稳定的tick数
SPAWN_STABILIZE_DELAY = 0.05      # 每个tick之间的延迟（秒）

# ==================== 路线规划参数 ====================
ROUTE_SAMPLING_RESOLUTION = 2.0   # 路径采样分辨率（米）
ROUTE_COMPLETED_THRESHOLD = 5.0   # 到达目的地阈值（米）

# ==================== 显示参数 ====================
VISUALIZATION_WIDTH = 400
VISUALIZATION_HEIGHT = 220
PRINT_INTERVAL_FRAMES = 10        # 每多少帧打印一次信息

