#!/usr/bin/env python
# coding=utf-8
'''
车辆控制模块
负责将控制信号应用到车辆
'''

import carla
from carla_config import SPEED_NORMALIZATION_MPS


class VehicleController:
    """车辆控制器"""
    
    @staticmethod
    def apply_control(vehicle, steer, throttle, brake):
        """
        应用控制信号到车辆
        
        参数:
            vehicle: carla.Vehicle 对象
            steer (float): 方向盘角度 [-1.0, 1.0]
            throttle (float): 油门 [0.0, 1.0]
            brake (float): 刹车 [0.0, 1.0]
        """
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False
        vehicle.apply_control(control)
    
    @staticmethod
    def get_speed_normalized(vehicle, speed_normalization_mps=None):
        """
        获取车辆归一化速度
        
        使用 Carla 0.9.x 原生 API get_velocity() 获取速度向量（m/s）
        
        参数:
            vehicle: carla.Vehicle 对象
            speed_normalization_mps (float): 归一化速度上限（m/s，默认与训练一致）
            
        返回:
            float: 归一化速度值 [0, 1]
        """
        if speed_normalization_mps is None:
            speed_normalization_mps = SPEED_NORMALIZATION_MPS
        
        # 使用 Carla API 获取速度向量（单位：m/s）
        velocity = vehicle.get_velocity()  # 返回 carla.Vector3D
        
        # 计算速度大小（欧几里得距离）
        speed_mps = 3.6*(velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        # 归一化
        return speed_mps / speed_normalization_mps
