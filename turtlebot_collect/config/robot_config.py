#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 机器人参数配置

本文件定义了不同型号 TurtleBot 机器人的运动学参数，
这些参数用于控制信号转换（将 TurtleBot 的线速度/角速度转换为 CARLA 格式）。

使用方法:
    from config import RobotConfig
    
    # 获取特定型号的参数
    params = RobotConfig.TURTLEBOT_PARAMS['burger']
    max_linear = params['max_linear']   # 0.22 m/s
    max_angular = params['max_angular'] # 2.84 rad/s
    
    # 或使用默认型号
    model = RobotConfig.DEFAULT_MODEL   # 'turtlebot2'
'''


class RobotConfig:
    """
    TurtleBot 不同型号的运动参数配置
    
    参数说明:
        max_linear: 最大线速度 (m/s)，机器人前进/后退的最大速度
        max_angular: 最大角速度 (rad/s)，机器人原地旋转的最大速度
    
    这些参数用于:
        1. 限制发送给机器人的速度命令，防止超出硬件能力
        2. 将速度命令归一化到 [-1, 1] 范围，与 CARLA 格式兼容
    """
    
    # ============ 各型号运动参数 ============
    # 格式: 'model_name': {'max_linear': m/s, 'max_angular': rad/s}
    TURTLEBOT_PARAMS = {
        
        # ---------- TurtleBot 1 ----------
        # 基于 iRobot Create 底盘，较老的型号
        # 特点: 速度较慢，但稳定性好
        'turtlebot1': {
            'max_linear': 0.5,     # 最大线速度 0.5 m/s (约 1.8 km/h)
            'max_angular': 1.0,    # 最大角速度 1.0 rad/s (约 57 deg/s)
        },
        
        # ---------- TurtleBot 2 / Kobuki ----------
        # 基于 Kobuki 底盘，最常用的型号
        # 特点: 性价比高，社区支持好
        'turtlebot2': {
            'max_linear': 0.7,     # 最大线速度 0.7 m/s (约 2.5 km/h)
                                   # Kobuki 硬件最大支持 0.7 m/s
            'max_angular': 1.0,    # 最大角速度 1.0 rad/s (约 57 deg/s)
                                   # 安全值，Kobuki 硬件最大约 3.14 rad/s (180 deg/s)
                                   # 使用较低值以保证控制稳定性
        },
        
        # Kobuki 底盘 (与 turtlebot2 相同)
        'kobuki': {
            'max_linear': 0.7,     # m/s
            'max_angular': 1.0,    # rad/s
        },
        
        # ---------- TurtleBot 3 Burger ----------
        # 小型版本，适合桌面实验
        # 特点: 体积小，价格低，但速度和负载能力有限
        'burger': {
            'max_linear': 0.22,    # 最大线速度 0.22 m/s (约 0.8 km/h)
            'max_angular': 2.84,   # 最大角速度 2.84 rad/s (约 163 deg/s)
                                   # 角速度较高，转向灵活
        },
        
        # ---------- TurtleBot 3 Waffle ----------
        # 大型版本，适合实际应用
        # 特点: 负载能力强，可搭载更多传感器
        'waffle': {
            'max_linear': 0.26,    # 最大线速度 0.26 m/s (约 0.9 km/h)
            'max_angular': 1.82,   # 最大角速度 1.82 rad/s (约 104 deg/s)
        },
        
        # TurtleBot 3 Waffle Pi (带树莓派版本)
        'waffle_pi': {
            'max_linear': 0.26,    # m/s
            'max_angular': 1.82,   # rad/s
        },
    }
    
    # ============ 默认配置 ============
    
    # 默认使用的机器人型号
    # 可选值: 'turtlebot1', 'turtlebot2', 'kobuki', 'burger', 'waffle', 'waffle_pi'
    DEFAULT_MODEL = 'turtlebot2'
    
    # 当指定的型号不在 TURTLEBOT_PARAMS 中时，使用以下默认参数
    # 这里使用 TurtleBot 3 Burger 的参数作为保守默认值
    DEFAULT_MAX_LINEAR = 0.22   # 默认最大线速度 (m/s)
    DEFAULT_MAX_ANGULAR = 2.84  # 默认最大角速度 (rad/s)
