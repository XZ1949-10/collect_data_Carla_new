#!/usr/bin/env python
# coding=utf-8
'''
模型输出后处理模块
对模型的原始输出进行启发式规则优化
基于 Carla_cil/Cil/imitation_learning_pytorch.py 的后处理逻辑
'''

import numpy as np
from collections import deque
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
        self.speed_normalization = SPEED_NORMALIZATION_MPS  # 使用配置文件中的值
        
        # 避免停车/停车判断阈值
        self.high_brake_threshold = 0.3       # 高刹车阈值
        self.low_brake_threshold = 0.1        # 低刹车阈值
        self.high_throttle_threshold = 0.3    # 高油门阈值
        self.low_pred_speed_threshold = 5.0   # 低预测速度阈值 (km/h)
        self.high_pred_speed_threshold = 15.0 # 高预测速度阈值 (km/h)
        self.speed_drop_threshold = 5.0       # 速度下降阈值 (km/h)
        
        # 历史记录（用于判断速度趋势）
        self.history_size = 5  # 记录最近10帧
        self.pred_speed_history = deque(maxlen=self.history_size)
        self.real_speed_history = deque(maxlen=self.history_size)
        self.speed_drop_ratio_threshold = 0.3  # 速度下降比例阈值（30%）
    
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
                throttle = throttle * 0.6

        
        # 规则5: 避免停车逻辑
        if self.enable_avoid_stopping and pred_speed_normalized is not None:
            throttle, brake = self._apply_avoid_stopping(
                throttle, brake, speed_normalized, pred_speed_normalized
            )
        
        return steer, throttle, brake
    
    def _apply_avoid_stopping(self, throttle, brake, speed_normalized, pred_speed_normalized):
        """
        应用避免停车/停车逻辑
        
        根据 brake、throttle、预测速度、真实速度的组合判断应该停车还是继续行驶
        使用投票机制：停车票数 vs 行驶票数，最终综合决定
        
        停车条件:
            1. brake高 + 预测速度持续下降 → +1 停车票
            2. brake高 + 预测速度高 → +1 停车票（紧急制动）
            3. brake低 + 预测速度低 → +1 停车票（自然减速）
            4. brake高 + 预测速度低 → +1 停车票
            
        行驶条件:
            5. throttle高 + 预测速度 > 真实速度 → +1 行驶票
        """
        real_speed = speed_normalized * self.speed_normalization
        real_pred_speed = pred_speed_normalized * self.speed_normalization
        
        # 更新历史记录
        self.pred_speed_history.append(real_pred_speed)
        self.real_speed_history.append(real_speed)
        
        # 计算速度趋势
        is_speed_dropping = self._is_speed_dropping()
        is_speed_drop_ratio_high = self._is_speed_drop_ratio_high()
        
        high_brake = brake > self.high_brake_threshold
        low_brake = brake <= self.low_brake_threshold
        high_throttle = throttle > self.high_throttle_threshold
        low_pred_speed = real_pred_speed < self.low_pred_speed_threshold
        high_pred_speed = real_pred_speed > self.high_pred_speed_threshold
        pred_greater_than_real = real_pred_speed > real_speed
        
        # 投票计数
        stop_votes = 0
        go_votes = 0
        
        # 规则1: brake高 + 预测速度持续下降 → 停车票
        if high_brake and is_speed_dropping:
            stop_votes += 1
        
        # 规则2: brake高 + 预测速度高 → 停车票（紧急制动）
        if high_brake and high_pred_speed:
            stop_votes += 1
        
        # 规则3: brake低 + 预测速度低 → 停车票（自然减速到停止）
        if low_brake and low_pred_speed:
            stop_votes += 1
        
        # 规则4: brake高 + 预测速度低 → 停车票
        if high_brake and low_pred_speed:
            stop_votes += 1
        
        # 规则5: throttle高 + 预测速度 > 真实速度 → 行驶票
        if high_throttle and pred_greater_than_real:
            go_votes += 1
        
        # 规则6: 真实速度和预测速度下降比例都很大 → 停车票
        if is_speed_drop_ratio_high:
            stop_votes += 1
        
        # 综合决策
        if stop_votes > go_votes:
            # 停车：清零油门，保留刹车
            throttle = 0.0
            if low_brake and low_pred_speed:
                brake = 0.0  # 自然减速场景不需要刹车
        elif go_votes > stop_votes:
            # 行驶：追踪预测速度
            # speed_error = pred_speed_normalized - speed_normalized
            # throttle = throttle + speed_error
            throttle = np.clip(throttle, 0.0, 1.0)
            brake = 0.0
        else:
            # 票数相等：默认追踪预测速度（如果预测速度更高）
            if pred_greater_than_real:
                # speed_error = pred_speed_normalized - speed_normalized
                # throttle = throttle + speed_error
                throttle = np.clip(throttle, 0.0, 1.0)
                brake = 0.0
        
        return throttle, brake
    
    def _is_speed_dropping(self):
        """
        判断预测速度是否在持续下降
        
        返回:
            bool: 如果最近一段时间速度下降超过阈值，返回 True
        """
        if len(self.pred_speed_history) < 3:
            return False
        
        # 比较最早和最新的速度
        oldest = self.pred_speed_history[0]
        newest = self.pred_speed_history[-1]
        drop = oldest - newest
        
        return drop > self.speed_drop_threshold
    
    def _is_speed_drop_ratio_high(self):
        """
        判断真实速度和预测速度的下降比例是否都很大
        
        返回:
            bool: 如果两者下降比例都超过阈值，返回 True
        """
        if len(self.pred_speed_history) < 3 or len(self.real_speed_history) < 3:
            return False
        
        # 预测速度下降比例
        pred_oldest = self.pred_speed_history[0]
        pred_newest = self.pred_speed_history[-1]
        if pred_oldest > 1.0:  # 避免除零，且只在有一定速度时判断
            pred_drop_ratio = (pred_oldest - pred_newest) / pred_oldest
        else:
            pred_drop_ratio = 0.0
        
        # 真实速度下降比例
        real_oldest = self.real_speed_history[0]
        real_newest = self.real_speed_history[-1]
        if real_oldest > 1.0:
            real_drop_ratio = (real_oldest - real_newest) / real_oldest
        else:
            real_drop_ratio = 0.0
        
        # 两者下降比例都超过阈值
        return (pred_drop_ratio > self.speed_drop_ratio_threshold and 
                real_drop_ratio > self.speed_drop_ratio_threshold)
    
