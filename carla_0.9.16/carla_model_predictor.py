#!/usr/bin/env python
# coding=utf-8
'''
模型预测模块
负责使用模型进行推理预测
'''

import time
import numpy as np
import torch
from carla_config import MAX_SPEED_KMH, POST_PROCESSOR_DEFAULT_CONFIG
from carla_post_processor import PostProcessor


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model, device, enable_post_processing=False, post_processor_config=None):
        """
        初始化预测器
        
        参数:
            model: PyTorch模型
            device: torch.device 对象
            enable_post_processing (bool): 是否启用后处理
            post_processor_config (dict): 后处理器配置参数
        """
        self.model = model
        self.device = device
        self.all_branch_predictions = None
        
        # 初始化后处理器
        if enable_post_processing:
            if post_processor_config is None:
                # 使用配置文件中的默认配置
                self.post_processor = PostProcessor(**POST_PROCESSOR_DEFAULT_CONFIG)
            else:
                # 使用自定义配置
                self.post_processor = PostProcessor(**post_processor_config)
            
            print("✅ 后处理器已启用")
            # self.post_processor.print_config()
        else:
            self.post_processor = None
            print("⚠️  后处理器已禁用 - 模型输出将直接作为控制信号")
        
    def predict(self, img_tensor, speed, current_command):
        """
        使用模型预测控制信号（可选后处理）
        
        参数:
            img_tensor: torch.Tensor，预处理后的图像
            speed: float，归一化的速度值（除以25.0 KM/H，与训练时一致）
            current_command: int，当前导航命令 (2-5)
            
        返回:
            dict: 包含控制信号和不确定性的字典
        """
        # 模型推理
        speed_tensor = torch.FloatTensor([[speed]]).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            pred_control, pred_speed, log_var_control, log_var_speed = \
                self.model(img_tensor, speed_tensor)
            inference_time = time.time() - start_time
        
        # 提取当前命令对应的控制信号
        # 命令编码: 2=跟车, 3=左转, 4=右转, 5=直行
        # 转换为分支索引: 0, 1, 2, 3
        pred_control = pred_control.cpu().numpy()[0]
        
        # 保存所有分支的输出（用于调试）
        self.all_branch_predictions = pred_control.copy()
        
        branch_idx = current_command - 2  # 2->0, 3->1, 4->2, 5->3
        start_idx = branch_idx * 3
        control_values = pred_control[start_idx:start_idx+3]
        
        # 提取原始模型输出（不做clip，与训练时一致）
        steer = float(control_values[0])
        throttle = float(control_values[1])
        brake = float(control_values[2])
        
        # 获取预测速度和不确定性
        predicted_speed = pred_speed.cpu().numpy()[0][0]

        
        # 应用后处理（如果启用）
        if self.post_processor is not None:
            # 注意：系统现在统一使用25.KM/H 作为速度归一化因子，与训练时一致
            steer, throttle, brake = self.post_processor.process(
                steer, throttle, brake,
                speed,
                predicted_speed,
                current_command  # 传入当前命令用于转弯减速判断
            )
        
        # 最后做clip，确保控制值在合法范围内
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'pred_speed': predicted_speed * MAX_SPEED_KMH,
            'pred_speed_normalized': predicted_speed,  # 添加归一化的预测速度
            'inference_time': inference_time
        }
    
    def get_all_branch_predictions(self):
        """获取所有分支的预测值（用于调试）"""
        return self.all_branch_predictions
