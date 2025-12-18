#!/usr/bin/env python
# coding=utf-8
'''
控制信号转换模块
将 TurtleBot 的控制信号转换为 CARLA 格式
'''

import numpy as np

from config import RobotConfig, CommandConfig


class ControlConverter:
    """
    控制信号转换器
    
    TurtleBot 使用: linear_vel (m/s), angular_vel (rad/s)
    CARLA 使用: steer (-1~1), throttle (0~1), brake (0~1), speed (km/h)
    """
    
    def __init__(self, 
                 max_linear=None, 
                 max_angular=None,
                 model=None):
        """
        初始化转换器
        
        参数:
            max_linear (float): 最大线速度 (m/s)，None 使用型号默认值
            max_angular (float): 最大角速度 (rad/s)，None 使用型号默认值
            model (str): TurtleBot 型号，None 使用默认型号
        """
        model = model or RobotConfig.DEFAULT_MODEL
        
        if model in RobotConfig.TURTLEBOT_PARAMS:
            params = RobotConfig.TURTLEBOT_PARAMS[model]
            self.max_linear = max_linear or params['max_linear']
            self.max_angular = max_angular or params['max_angular']
        else:
            self.max_linear = max_linear or RobotConfig.DEFAULT_MAX_LINEAR
            self.max_angular = max_angular or RobotConfig.DEFAULT_MAX_ANGULAR
            
        self.model = model
        
    def to_carla_format(self, linear_vel, angular_vel, actual_speed=None):
        """
        将 TurtleBot 控制信号转换为 CARLA 格式
        
        参数:
            linear_vel (float): 线速度命令 (m/s)，正值前进，负值后退
            angular_vel (float): 角速度命令 (rad/s)，正值左转，负值右转
            actual_speed (float): 实际速度 (m/s)，如果为 None 则使用 |linear_vel|
            
        返回:
            dict: {
                'steer': float,      # -1.0 ~ 1.0 (负=左转, 正=右转)
                'throttle': float,   # 0.0 ~ 1.0 (前进强度)
                'brake': float,      # 0.0 ~ 1.0 (后退强度)
                'speed_kmh': float,  # km/h (实际速度)
            }
            
        转换说明:
            TurtleBot 是差速驱动机器人，没有真正的"刹车"概念。
            为了与 CARLA 格式兼容，我们使用以下映射：
            
            前进 (linear_vel > 0):
                - throttle = 前进强度 (0~1)
                - brake = 0 (互斥)
                
            后退 (linear_vel < 0):
                - throttle = 0 (互斥)
                - brake = 后退强度 (0~1)
                
            停止 (linear_vel = 0):
                - throttle = 0
                - brake = 0
                
            ⚠️ 注意: TurtleBot 数据中的 brake 表示"后退强度"，不是"减速"！
            油门和刹车互斥，不会同时大于 0。
        """
        # 方向盘: angular_vel 转换为 steer
        # TurtleBot: 左转为正 (逆时针)
        # CARLA: 左转为负
        # 所以需要取反
        steer = np.clip(-angular_vel / self.max_angular, -1.0, 1.0)
        
        # 油门和刹车 (互斥)
        if linear_vel >= 0:
            # 前进或停止
            throttle = np.clip(linear_vel / self.max_linear, 0.0, 1.0)
            brake = 0.0
        else:
            # 后退
            throttle = 0.0
            brake = np.clip(-linear_vel / self.max_linear, 0.0, 1.0)
        
        # 速度转换为 km/h
        speed = actual_speed if actual_speed is not None else abs(linear_vel)
        speed_kmh = speed * 3.6
        
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'speed_kmh': speed_kmh,
        }
    
    def from_carla_format(self, steer, throttle, brake):
        """
        将 CARLA 格式转换为 TurtleBot 控制信号
        
        参数:
            steer (float): 方向盘 (-1.0 ~ 1.0)，负=左转，正=右转
            throttle (float): 前进强度 (0.0 ~ 1.0)
            brake (float): 后退强度 (0.0 ~ 1.0)
            
        返回:
            dict: {
                'linear_vel': float,   # m/s，正=前进，负=后退
                'angular_vel': float,  # rad/s，正=左转，负=右转
            }
            
        转换说明:
            与 to_carla_format 对应，油门和刹车互斥：
            - throttle > 0, brake = 0: 前进，linear_vel = +throttle * max_linear
            - throttle = 0, brake > 0: 后退，linear_vel = -brake * max_linear
            - throttle = 0, brake = 0: 停止，linear_vel = 0
        """
        # 角速度: steer 转换为 angular_vel (取反)
        # CARLA: 左转为负 → TurtleBot: 左转为正
        angular_vel = -steer * self.max_angular
        
        # 线速度: 油门和刹车互斥
        if brake > 0.01:
            # 后退 (brake 表示后退强度)
            linear_vel = -brake * self.max_linear
        else:
            # 前进或停止
            linear_vel = throttle * self.max_linear
        
        return {
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
        }
    
    def build_targets_vector(self, linear_vel, angular_vel, actual_speed, command):
        """
        构建与 CARLA 兼容的 targets 向量
        
        参数:
            linear_vel (float): 线速度 (m/s)
            angular_vel (float): 角速度 (rad/s)
            actual_speed (float): 实际速度 (m/s)
            command (float): 导航命令 (2=Follow, 3=Left, 4=Right, 5=Straight)
            
        返回:
            np.ndarray: targets 向量
        """
        carla_ctrl = self.to_carla_format(linear_vel, angular_vel, actual_speed)
        
        targets = np.zeros(CommandConfig.TARGETS_DIM, dtype=np.float32)
        targets[CommandConfig.TARGETS_STEER_IDX] = carla_ctrl['steer']
        targets[CommandConfig.TARGETS_THROTTLE_IDX] = carla_ctrl['throttle']
        targets[CommandConfig.TARGETS_BRAKE_IDX] = carla_ctrl['brake']
        targets[CommandConfig.TARGETS_SPEED_IDX] = carla_ctrl['speed_kmh']
        targets[CommandConfig.TARGETS_COMMAND_IDX] = command
        
        # 额外存储原始 TurtleBot 数据
        targets[CommandConfig.TARGETS_LINEAR_VEL_IDX] = linear_vel
        targets[CommandConfig.TARGETS_ANGULAR_VEL_IDX] = angular_vel
        
        return targets
