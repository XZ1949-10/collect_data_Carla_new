#!/usr/bin/env python
# coding=utf-8
'''
模型预测模块
'''

import time
import numpy as np
import torch

from ..config.inference_config import SPEED_NORMALIZATION_KMH


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model, device):
        """
        初始化预测器
        
        参数:
            model: PyTorch 模型
            device: torch.device 对象
        """
        self.model = model
        self.device = device
        self.all_branch_predictions = None
        
    def predict(self, img_tensor, speed_normalized, current_command):
        """
        使用模型预测控制信号
        
        参数:
            img_tensor: torch.Tensor，预处理后的图像 (1, 3, 88, 200)
            speed_normalized: float，归一化的速度值 (speed_kmh / 25.0)
            current_command: int，当前导航命令 (2-5)
            
        返回:
            dict: {
                'steer': float,      # 方向 (-1.0 ~ 1.0)
                'throttle': float,   # 油门 (0.0 ~ 1.0)
                'brake': float,      # 刹车 (0.0 ~ 1.0)
                'pred_speed': float, # 预测速度 (km/h)
                'inference_time': float,
            }
        """
        speed_tensor = torch.FloatTensor([[speed_normalized]]).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            pred_control, pred_speed, log_var_control, log_var_speed = \
                self.model(img_tensor, speed_tensor)
            inference_time = time.time() - start_time
        
        # 提取当前命令对应的控制信号
        # 命令编码: 2=Follow, 3=Left, 4=Right, 5=Straight
        # 转换为分支索引: 0, 1, 2, 3
        pred_control = pred_control.cpu().numpy()[0]
        self.all_branch_predictions = pred_control.copy()
        
        branch_idx = current_command - 2
        start_idx = branch_idx * 3
        control_values = pred_control[start_idx:start_idx+3]
        
        # 提取控制值
        steer = float(np.clip(control_values[0], -1.0, 1.0))
        throttle = float(np.clip(control_values[1], 0.0, 1.0))
        brake = float(np.clip(control_values[2], 0.0, 1.0))
        
        # 预测速度
        predicted_speed = pred_speed.cpu().numpy()[0][0]
        
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'pred_speed': predicted_speed * SPEED_NORMALIZATION_KMH,
            'inference_time': inference_time
        }
    
    def get_all_branch_predictions(self):
        """获取所有分支的预测值（用于调试）"""
        return self.all_branch_predictions
