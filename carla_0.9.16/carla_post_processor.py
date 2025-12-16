#!/usr/bin/env python
# coding=utf-8
'''
模型输出后处理模块
对模型的原始输出进行启发式规则优化
基于 Carla_cil/Cil/imitation_learning_pytorch.py 的后处理逻辑
'''

import numpy as np
from carla_config import SPEED_NORMALIZATION_MPS, MAX_SPEED_LIMIT_MPS


class PostProcessor:
    """
    模型输出后处理器
    
    应用启发式规则优化模型的原始控制输出，提高驾驶安全性和稳定性
    """
    
    def __init__(self, 
                 enable_brake_denoising=True,
                 enable_throttle_brake_mutex=True,
                 enable_speed_limit=True,
                 enable_turning_slowdown=True,
                 enable_avoid_stopping=True):
        """
        初始化后处理器
        
        参数:
            enable_brake_denoising (bool): 启用刹车去噪
            enable_throttle_brake_mutex (bool): 启用油门刹车互斥
            enable_speed_limit (bool): 启用速度限制
            enable_turning_slowdown (bool): 启用转弯减速
            enable_avoid_stopping (bool): 启用避免停车逻辑
        """
        self.enable_brake_denoising = enable_brake_denoising
        self.enable_throttle_brake_mutex = enable_throttle_brake_mutex
        self.enable_speed_limit = enable_speed_limit
        self.enable_turning_slowdown = enable_turning_slowdown
        self.enable_avoid_stopping = enable_avoid_stopping
        
        # 后处理参数（基于原始实现）
        self.brake_noise_threshold = 0.1      # 刹车噪声阈值
        self.max_speed_limit_mps = MAX_SPEED_LIMIT_MPS  # 最高速度限制（m/s），从配置文件读取
        self.turning_steer_threshold = 0.1   # 转弯方向盘阈值
        self.turning_throttle_scale = 0.4     # 转弯时油门缩放因子
        
        # 避免停车参数（基于原始实现，使用配置文件中的速度归一化因子）
        self.avoid_stopping_min_speed = 5.0   # 当前速度阈值 (km/h)
        self.avoid_stopping_pred_speed = 5.0  # 预测速度阈值 (km/h)
        self.avoid_stopping_target_speed = 5.6 # 目标启动速度 (km/h)
        self.turning_steer_scale = 1.5
        self.speed_normalization = SPEED_NORMALIZATION_MPS  # 使用配置文件中的值
    
    def process(self, steer, throttle, brake, speed_normalized, pred_speed_normalized=None, current_command=None):
        """
        对模型输出进行后处理
        
        参数:
            steer (float): 方向盘角度 [-1.0, 1.0]
            throttle (float): 油门 [0.0, 1.0]
            brake (float): 刹车 [0.0, 1.0]
            speed_normalized (float): 归一化的当前速度（除以25.0）
            pred_speed_normalized (float): 归一化的预测速度（可选，用于避免停车逻辑）
            current_command (int): 当前导航命令 (2=跟车, 3=左转, 4=右转, 5=直行)
            
        返回:
            tuple: (steer, throttle, brake) 处理后的控制信号
        """
        # 规则1: 刹车去噪 - 避免误刹车
        if self.enable_brake_denoising:
            if brake < self.brake_noise_threshold:
                brake = 0.0
        
        # 规则1.1: 刹车去噪 - 避免误刹车
        # if self.enable_brake_denoising:
        #     throttle=throttle*0.85
        
        # 规则2: 油门刹车互斥 - 如果油门更大，则不刹车
        if self.enable_throttle_brake_mutex:
            if throttle > brake:
                brake = 0.0
        
        # # 规则3: 速度限制 - 当速度超过配置的最高速度时，关闭油门
        if self.enable_speed_limit:
            if speed_normalized * self.speed_normalization > self.max_speed_limit_mps:
                throttle = throttle * 0.6
        
        # 规则4: 转弯减速 - 当命令为左转(3)或右转(4)时降低油门
        if self.enable_turning_slowdown and current_command is not None:
            if current_command in [3, 4]:  # 3=左转, 4=右转
                throttle = throttle * 1.0
        
        # 规则5: 避免停车逻辑
        if self.enable_avoid_stopping and pred_speed_normalized is not None:
            throttle, brake = self._apply_avoid_stopping(
                throttle, brake, speed_normalized, pred_speed_normalized
            )
        
        return steer, throttle, brake
    
    def _apply_avoid_stopping(self, throttle, brake, speed_normalized, pred_speed_normalized):
        """
        应用避免停车逻辑
        
        当车辆速度很慢但模型预测应该有速度时，增加油门避免误停
        
        参数:
            throttle (float): 油门
            brake (float): 刹车
            speed_normalized (float): 归一化的当前速度
            pred_speed_normalized (float): 归一化的预测速度
            
        返回:
            tuple: (throttle, brake) 调整后的油门和刹车
        """
        # 转换为实际速度 (km/h)
        real_speed = speed_normalized * self.speed_normalization
        real_predicted_speed = pred_speed_normalized * self.speed_normalization
        
        # 如果当前速度很慢但预测速度较高，说明可能误停
        if real_speed < self.avoid_stopping_min_speed and \
           real_predicted_speed > self.avoid_stopping_pred_speed:
            
            # P控制器：增加油门以达到目标速度
            target_speed_normalized = self.avoid_stopping_target_speed / self.speed_normalization
            throttle = 1.0 * (target_speed_normalized - speed_normalized) + throttle
            
            # 限制油门范围
            throttle = np.clip(throttle, 0.0, 1.0)
            
            # 不刹车
            brake = 0.0
        
        return throttle, brake
    
