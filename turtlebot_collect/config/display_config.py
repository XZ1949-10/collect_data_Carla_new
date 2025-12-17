#!/usr/bin/env python
# coding=utf-8
'''
显示配置

本文件定义了数据收集过程中可视化界面的参数，
包括窗口尺寸、颜色方案和字体设置。

界面布局 (美化版 v2 - 中文界面):
    ┌────────────────────────────────────────────────────────────────────┐
    │  ┌─────────────────────────────────────┬────────────────────────┐  │
    │  │                                     │   ┌──────────────────┐ │  │
    │  │                                     │   │   运行状态       │ │  │
    │  │         [摄像头画面]                 │   │  ● 录制中        │ │  │
    │  │        (200x88 等比放大)             │   │  ● 同步正常      │ │  │
    │  │                                     │   ├──────────────────┤ │  │
    │  │                                     │   │   导航命令       │ │  │
    │  │                                     │   │   [ 跟随 ]       │ │  │
    │  │                                     │   ├──────────────────┤ │  │
    │  │                                     │   │   速度信息       │ │  │
    │  │                                     │   │  速度: 2.5 km/h  │ │  │
    │  └─────────────────────────────────────┴────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  控制信号                                                    │  │
    │  │  转向  [◀━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━▶]  -0.350   │  │
    │  │  油门  [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░]   0.750   │  │
    │  │  刹车  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.000   │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────────────┘

使用方法:
    from config import DisplayConfig
    
    window = DisplayConfig.WINDOW_NAME
    color = DisplayConfig.COLOR_RECORDING
'''


class DisplayConfig:
    """
    显示参数配置 (美化版 v2 - 中文界面)
    
    定义了:
    1. 窗口属性
    2. 颜色方案 (BGR 格式) - 现代深色主题
    3. 字体设置
    4. 中文状态文字
    """
    
    # ============ 窗口配置 ============
    
    # 窗口标题
    WINDOW_NAME = 'TurtleBot 数据收集器'
    
    # 显示窗口尺寸 (像素)
    # 基于 200x88 图像放大 3 倍 = 600x264
    # 加上右侧面板 200px 和边距
    DISPLAY_WIDTH = 830     # 显示宽度
    DISPLAY_HEIGHT = 404    # 显示高度
    
    # 图像放大倍数
    IMAGE_SCALE_FACTOR = 3.0
    
    # ============ 颜色配置 - 现代深色主题 ============
    # OpenCV 使用 BGR 格式
    
    # --- 背景色 ---
    COLOR_BG_DARK = (32, 28, 24)        # 主背景 - 深色
    COLOR_BG_PANEL = (48, 44, 40)       # 面板背景
    COLOR_BG_CARD = (58, 54, 50)        # 卡片背景
    COLOR_BG_HEADER = (42, 38, 34)      # 标题栏背景
    
    # --- 边框和分隔线 ---
    COLOR_BORDER = (80, 75, 70)         # 边框
    COLOR_BORDER_ACCENT = (120, 100, 80) # 强调边框
    COLOR_DIVIDER = (65, 60, 55)        # 分隔线
    
    # --- 文字颜色 ---
    COLOR_TEXT_PRIMARY = (255, 255, 255)    # 主文字 - 白色
    COLOR_TEXT_SECONDARY = (170, 165, 160)  # 次要文字
    COLOR_TEXT_MUTED = (120, 115, 110)      # 弱化文字
    COLOR_TITLE = (240, 220, 180)           # 标题 - 暖金色
    COLOR_TITLE_CN = (200, 220, 255)        # 中文标题 - 淡蓝
    
    # --- 状态颜色 ---
    COLOR_RECORDING = (80, 80, 255)     # 录制中 - 红色
    COLOR_RECORDING_GLOW = (40, 40, 160) # 录制光晕
    COLOR_STANDBY = (80, 200, 100)      # 待机 - 绿色
    COLOR_SYNC_OK = (80, 180, 80)       # 同步正常
    COLOR_SYNC_FAIL = (80, 80, 200)     # 同步失败
    COLOR_WARNING = (50, 150, 255)      # 警告 - 橙色
    
    # --- 命令颜色 ---
    COLOR_COMMAND = (255, 200, 100)     # 默认命令颜色
    COLOR_CMD_FOLLOW = (255, 200, 100)  # 跟随 - 金色
    COLOR_CMD_LEFT = (100, 200, 255)    # 左转 - 橙色
    COLOR_CMD_RIGHT = (255, 160, 120)   # 右转 - 蓝色
    COLOR_CMD_STRAIGHT = (120, 230, 120) # 直行 - 绿色
    
    # --- 控制信号颜色 (CARLA) ---
    COLOR_STEER = (255, 200, 80)        # 转向 - 青黄色
    COLOR_STEER_BG = (80, 65, 45)       # 转向背景
    COLOR_THROTTLE = (80, 220, 120)     # 油门 - 绿色
    COLOR_THROTTLE_BG = (40, 70, 50)    # 油门背景
    COLOR_BRAKE = (100, 100, 255)       # 刹车 - 红色
    COLOR_BRAKE_BG = (60, 45, 55)       # 刹车背景
    
    # --- TurtleBot 速度颜色 ---
    COLOR_LINEAR = (255, 180, 100)      # 线速度 - 橙色
    COLOR_ANGULAR = (150, 200, 255)     # 角速度 - 淡蓝
    COLOR_SPEED = (100, 255, 200)       # 速度 - 青绿
    
    # --- 标题颜色 (扩展) ---
    COLOR_TITLE_CARLA = (180, 220, 255)     # CARLA标题 - 淡蓝
    COLOR_TITLE_TURTLE = (180, 255, 200)    # TurtleBot标题 - 淡绿
    
    # --- 进度条颜色 ---
    COLOR_BAR_BG = (45, 42, 38)         # 进度条背景
    COLOR_BAR_BORDER = (70, 65, 60)     # 进度条边框
    COLOR_BAR_TICK = (55, 52, 48)       # 刻度线
    
    # 兼容旧代码
    COLOR_INFO = (255, 255, 255)
    
    # ============ 字体配置 ============
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.45
    FONT_SCALE_SMALL = 0.38
    FONT_SCALE_TITLE = 0.48
    FONT_THICKNESS = 1
    
    # ============ 中文状态文字 ============
    STATUS_RECORDING = '● 录制中'
    STATUS_STANDBY = '○ 待机'
    STATUS_SYNC_OK = '● 同步正常'
    STATUS_SYNC_FAIL = '● 同步异常'
    
    # ============ 中文命令名称 ============
    COMMAND_NAMES_CN = {
        2.0: '跟随',
        3.0: '左转',
        4.0: '右转',
        5.0: '直行',
    }
    
    # ============ 中文标签 ============
    LABEL_STATUS = '运行状态'
    LABEL_COMMAND = '导航命令'
    LABEL_VELOCITY = '速度信息'
    LABEL_STATS = '数据统计'
    LABEL_CONTROL_CARLA = 'CARLA 控制信号'
    LABEL_CONTROL_TURTLE = 'TurtleBot 速度'
    LABEL_STEER = '转向'
    LABEL_THROTTLE = '油门'
    LABEL_BRAKE = '刹车'
    LABEL_SPEED = '速度'
    LABEL_LINEAR = '线速度'
    LABEL_ANGULAR = '角速度'
    LABEL_EPISODE = '片段'
    LABEL_FRAMES = '帧数'
    
    # ============ 进度条归一化参数 ============
    # 用于将速度值归一化到 0~1 范围显示在进度条上
    
    # 线速度最大值 (m/s) - 用于进度条归一化
    # 设为 1.0 可以覆盖大多数 TurtleBot 型号
    NORM_MAX_LINEAR_VEL = 1.0
    
    # 角速度最大值 (rad/s) - 用于进度条归一化
    # 设为 3.0 可以覆盖 TurtleBot 3 Burger 的 2.84 rad/s
    NORM_MAX_ANGULAR_VEL = 3.0
    
    # 速度最大值 (km/h) - 用于进度条归一化
    # 设为 10.0 可以覆盖大多数室内机器人的速度范围
    NORM_MAX_SPEED_KMH = 10.0
