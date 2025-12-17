#!/usr/bin/env python
# coding=utf-8
'''
控制信号转换模块
CARLA 格式 (steer, throttle, brake) <-> TurtleBot 格式 (linear_vel, angular_vel)
'''

from ..config.inference_config import TURTLEBOT_PARAMS, SPEED_NORMALIZATION_KMH


class ControlConverter:
    """控制信号转换器"""
    
    def __init__(self, model='burger', max_linear=None, max_angular=None):
        """
        初始化转换器
        
        参数:
            model (str): TurtleBot 型号
            max_linear (float): 最大线速度 (m/s)，None 则使用型号默认值
            max_angular (float): 最大角速度 (rad/s)，None 则使用型号默认值
        """
        if model in TURTLEBOT_PARAMS:
            params = TURTLEBOT_PARAMS[model]
            self.max_linear = max_linear or params['max_linear']
            self.max_angular = max_angular or params['max_angular']
        else:
            self.max_linear = max_linear or 0.22
            self.max_angular = max_angular or 2.84
            
        self.model = model
        
    def from_carla_format(self, steer, throttle, brake):
        """
        将 CARLA 格式转换为 TurtleBot 控制信号
        
        参数:
            steer (float): 方向盘 (-1.0 ~ 1.0)
            throttle (float): 油门 (0.0 ~ 1.0)
            brake (float): 刹车 (0.0 ~ 1.0)
            
        返回:
            dict: {
                'linear_vel': float,   # m/s
                'angular_vel': float,  # rad/s
            }
        """
        # 角速度: steer 转换为 angular_vel (取反)
        angular_vel = -steer * self.max_angular
        
        # 线速度
        if brake > 0.1:
            # 刹车时减速或后退
            linear_vel = -brake * self.max_linear * 0.5  # 后退速度减半
        else:
            linear_vel = throttle * self.max_linear
        
        return {
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
        }
    
    def to_carla_format(self, linear_vel, angular_vel, actual_speed=None):
        """
        将 TurtleBot 控制信号转换为 CARLA 格式
        
        参数:
            linear_vel (float): 线速度 (m/s)
            angular_vel (float): 角速度 (rad/s)
            actual_speed (float): 实际速度 (m/s)，可选
            
        返回:
            dict: {
                'steer': float,
                'throttle': float,
                'brake': float,
                'speed_kmh': float,
                'speed_normalized': float,
            }
        """
        # 方向: angular_vel 转换为 steer (取反)
        steer = -angular_vel / self.max_angular
        steer = max(-1.0, min(1.0, steer))
        
        # 油门/刹车
        if linear_vel >= 0:
            throttle = linear_vel / self.max_linear
            brake = 0.0
        else:
            throttle = 0.0
            brake = -linear_vel / self.max_linear
        
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        
        # 速度
        speed = actual_speed if actual_speed is not None else abs(linear_vel)
        speed_kmh = speed * 3.6
        speed_normalized = speed_kmh / SPEED_NORMALIZATION_KMH
        
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'speed_kmh': speed_kmh,
            'speed_normalized': speed_normalized,
        }
