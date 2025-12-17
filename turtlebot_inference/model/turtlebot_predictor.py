#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 模型预测模块
直接预测 linear_vel 和 angular_vel

与 CARLA 版本的区别:
- 输出 2 个控制值 (linear_vel, angular_vel) 而不是 3 个 (steer, throttle, brake)
- 不需要 CARLA 格式转换
'''

import time
import numpy as np
import torch

from ..config.inference_config import SPEED_NORMALIZATION_KMH, TURTLEBOT_PARAMS


class TurtleBotPredictor:
    """
    TurtleBot 模型预测器
    
    直接输出 linear_vel 和 angular_vel
    """
    
    def __init__(self, model, device, turtlebot_model='burger'):
        """
        初始化预测器
        
        参数:
            model: PyTorch 模型
            device: torch.device 对象
            turtlebot_model (str): TurtleBot 型号
        """
        self.model = model
        self.device = device
        self.all_branch_predictions = None
        
        # TurtleBot 参数
        if turtlebot_model in TURTLEBOT_PARAMS:
            params = TURTLEBOT_PARAMS[turtlebot_model]
            self.max_linear = params['max_linear']
            self.max_angular = params['max_angular']
        else:
            self.max_linear = 0.22
            self.max_angular = 2.84
        
        self.turtlebot_model = turtlebot_model
        
    def predict(self, img_tensor, speed_normalized, current_command):
        """
        使用模型预测控制信号
        
        参数:
            img_tensor: torch.Tensor，预处理后的图像 (1, 3, 88, 200)
            speed_normalized: float，归一化的速度值 (speed_kmh / 25.0)
            current_command: int，当前导航命令 (2-5)
            
        返回:
            dict: {
                'linear_vel': float,   # 线速度 (m/s)
                'angular_vel': float,  # 角速度 (rad/s)
                'linear_vel_norm': float,  # 归一化线速度 (-1 ~ 1)
                'angular_vel_norm': float, # 归一化角速度 (-1 ~ 1)
                'pred_speed': float,   # 预测速度 (km/h)
                'inference_time': float,
            }
        """
        speed_tensor = torch.FloatTensor([[speed_normalized]]).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(img_tensor, speed_tensor)
            inference_time = time.time() - start_time
        
        # 处理不同的网络结构输出
        if len(outputs) == 4:
            pred_control, pred_speed, log_var_control, log_var_speed = outputs
        else:
            pred_control, pred_speed = outputs
        
        # 提取当前命令对应的控制信号
        # 命令编码: 2=Follow, 3=Left, 4=Right, 5=Straight
        # 转换为分支索引: 0, 1, 2, 3
        pred_control = pred_control.cpu().numpy()[0]
        self.all_branch_predictions = pred_control.copy()
        
        branch_idx = current_command - 2
        start_idx = branch_idx * 2  # 每个分支 2 个值
        control_values = pred_control[start_idx:start_idx + 2]
        
        # 提取归一化的控制值
        linear_vel_norm = float(np.clip(control_values[0], -1.0, 1.0))
        angular_vel_norm = float(np.clip(control_values[1], -1.0, 1.0))
        
        # 反归一化得到实际控制值
        linear_vel = linear_vel_norm * self.max_linear
        angular_vel = angular_vel_norm * self.max_angular
        
        # 预测速度
        predicted_speed = pred_speed.cpu().numpy()[0][0]
        
        return {
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
            'linear_vel_norm': linear_vel_norm,
            'angular_vel_norm': angular_vel_norm,
            'pred_speed': predicted_speed * SPEED_NORMALIZATION_KMH,
            'inference_time': inference_time
        }
    
    def get_all_branch_predictions(self):
        """
        获取所有分支的预测值（用于调试）
        
        返回:
            np.ndarray: (8,) 数组，4个分支 × 2个控制值
        """
        return self.all_branch_predictions
    
    def get_all_branch_velocities(self):
        """
        获取所有分支的速度预测（反归一化后）
        
        返回:
            list: 4个分支的 (linear_vel, angular_vel) 元组列表
        """
        if self.all_branch_predictions is None:
            return None
        
        velocities = []
        for i in range(4):
            start_idx = i * 2
            linear_norm = self.all_branch_predictions[start_idx]
            angular_norm = self.all_branch_predictions[start_idx + 1]
            
            linear_vel = linear_norm * self.max_linear
            angular_vel = angular_norm * self.max_angular
            
            velocities.append((linear_vel, angular_vel))
        
        return velocities
