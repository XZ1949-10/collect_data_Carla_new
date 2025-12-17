#!/usr/bin/env python
# coding=utf-8
'''
导航命令配置

本文件定义了导航命令的值和 targets 向量的格式。
这些定义与 CARLA 训练数据格式完全一致，确保数据兼容性。

导航命令说明:
    - Follow (2): 跟随前车/保持车道
    - Left (3): 在路口左转
    - Right (4): 在路口右转
    - Straight (5): 在路口直行

targets 向量说明:
    targets 是一个 25 维的向量，存储控制信号和元数据。
    这个格式与 CARLA 条件模仿学习 (CIL) 的数据格式一致。

训练模式说明:
    1. CARLA 兼容模式: 使用 targets[0:3] (steer, throttle, brake)
       - 适用于 carla_train/carla_net_ori.py
       - 输出 3 维控制信号
       
    2. TurtleBot 直接模式: 使用 targets[20:22] (linear_vel, angular_vel)
       - 适用于 turtlebot_train/turtlebot_net.py
       - 输出 2 维控制信号
       - 无需格式转换，直接预测速度

使用方法:
    from config import CommandConfig
    
    # 获取命令值
    cmd = CommandConfig.CMD_LEFT  # 3.0
    
    # 构建 targets 向量
    targets = np.zeros(CommandConfig.TARGETS_DIM)
    targets[CommandConfig.TARGETS_STEER_IDX] = steer
    targets[CommandConfig.TARGETS_COMMAND_IDX] = cmd
'''


class CommandConfig:
    """
    导航命令配置
    
    定义了:
    1. 导航命令的数值 (与 CARLA 一致)
    2. 命令名称映射 (用于显示和日志)
    3. targets 向量的索引定义 (用于数据存储)
    """
    
    # ============ 命令值定义 ============
    # 这些值与 CARLA 条件模仿学习 (CIL) 的命令定义一致
    # 注意: 值从 2 开始，0 和 1 在 CARLA 中有其他用途
    
    CMD_FOLLOW = 2.0    # 跟随/保持车道 - 无特定转向指令，保持当前行驶状态
    CMD_LEFT = 3.0      # 左转 - 在下一个路口左转
    CMD_RIGHT = 4.0     # 右转 - 在下一个路口右转
    CMD_STRAIGHT = 5.0  # 直行 - 在下一个路口直行
    
    # 默认命令 (启动时使用)
    # Follow 是最安全的默认选择，机器人会保持当前状态
    DEFAULT_COMMAND = CMD_FOLLOW
    
    # ============ 命令名称映射 ============
    # 用于日志输出和调试显示
    COMMAND_NAMES = {
        CMD_FOLLOW: 'Follow',      # 跟随
        CMD_LEFT: 'Left',          # 左转
        CMD_RIGHT: 'Right',        # 右转
        CMD_STRAIGHT: 'Straight',  # 直行
    }
    
    # 简短名称 (用于界面显示，节省空间)
    COMMAND_SHORT_NAMES = {
        CMD_FOLLOW: 'Follow',    # Follow
        CMD_LEFT: 'Left',      # Left
        CMD_RIGHT: 'Right',     # Right
        CMD_STRAIGHT: 'Straight',  # Straight
    }
    
    # ============ targets 向量索引定义 ============
    # targets 是一个 25 维向量，以下定义各字段的索引位置
    # 这个格式与 CARLA CIL 训练数据完全一致
    
    # --- 控制信号 (索引 0-2) ---
    # 
    # ⚠️ TurtleBot 特殊说明:
    # TurtleBot 是差速驱动机器人，没有真正的"刹车"概念。
    # 为了与 CARLA 格式兼容，我们使用以下映射:
    #
    #   前进 (linear_vel > 0):
    #     - throttle = 前进强度 (0~1)
    #     - brake = 0 (互斥)
    #
    #   后退 (linear_vel < 0):
    #     - throttle = 0 (互斥)
    #     - brake = 后退强度 (0~1)
    #
    #   停止 (linear_vel = 0):
    #     - throttle = 0
    #     - brake = 0
    #
    # ⚠️ 油门和刹车互斥，不会同时大于 0！
    # ⚠️ TurtleBot 数据中的 brake 表示"后退强度"，不是"减速"！
    # 原始的 linear_vel 和 angular_vel 存储在索引 20-21
    
    TARGETS_STEER_IDX = 0       # 转向角 (-1.0 ~ 1.0)
                                 # -1.0 = 最大左转, +1.0 = 最大右转
    
    TARGETS_THROTTLE_IDX = 1    # 前进强度 (0.0 ~ 1.0)
                                 # TurtleBot: 前进强度，与 brake 互斥
                                 # CARLA: 油门强度
    
    TARGETS_BRAKE_IDX = 2       # 后退强度 (TurtleBot) / 刹车 (CARLA)
                                 # TurtleBot: 后退强度 (0~1)，与 throttle 互斥
                                 # CARLA: 刹车强度 (0~1)
    
    # --- 速度信息 (索引 10) ---
    TARGETS_SPEED_IDX = 10      # 当前速度 (km/h)
                                 # 用于速度条件输入
    
    # --- TurtleBot 扩展字段 (索引 20-21) ---
    # 这些字段是 TurtleBot 特有的，用于存储原始控制信号
    TARGETS_LINEAR_VEL_IDX = 20   # 原始线速度 (m/s)
    TARGETS_ANGULAR_VEL_IDX = 21  # 原始角速度 (rad/s)
    
    # --- 导航命令 (索引 24) ---
    TARGETS_COMMAND_IDX = 24    # 导航命令 (2.0/3.0/4.0/5.0)
                                 # 用于条件分支选择
    
    # ============ 向量维度 ============
    # targets 向量的总维度
    TARGETS_DIM = 25
