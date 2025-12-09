#!/usr/bin/env python
# coding=utf-8
'''
日志和调试输出模块
负责打印调试信息、状态信息和统计信息
'''

import time
import numpy as np
from carla_config import COMMAND_NAMES_EN, SPEED_NORMALIZATION_MPS


class CarlaLogger:
    """Carla推理日志记录器"""
    
    def __init__(self):
        """初始化日志记录器"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = None
        
    def set_start_time(self, start_time):
        """设置开始时间"""
        self.start_time = start_time
        
    def increment_frame(self):
        """增加帧计数"""
        self.frame_count += 1
        
    def add_inference_time(self, inference_time):
        """累计推理时间"""
        self.total_inference_time += inference_time
        
    def debug_print_all_branches(self, model_predictor, current_command, control_result=None):
        """
        调试：打印所有分支的预测值，并对比后处理后的实际值
        
        参数:
            model_predictor: ModelPredictor 对象
            current_command: 当前导航命令 (2-5)
            control_result: 后处理后的控制结果（可选）
        """
        all_predictions = model_predictor.get_all_branch_predictions()
        if all_predictions is None:
            return
            
        print(f"\n{'='*80}")
        print(f"[调试] 所有分支预浌值 (帧 {self.frame_count})")
        print(f"{'='*80}")
        print(f"当前命令: {current_command} ({COMMAND_NAMES_EN.get(current_command, 'Unknown')})")
        print(f"当前分支索引: {current_command - 2}")
        print(f"\n{'分支':<12} {'命令':<10} {'Steer':<10} {'Throttle':<10} {'Brake':<10} {'使用?'}")
        print(f"{'-'*80}")
        
        branch_names = ['Follow', 'Left', 'Right', 'Straight']
        for i, name in enumerate(branch_names):
            start_idx = i * 3
            steer = all_predictions[start_idx]
            throttle = all_predictions[start_idx + 1]
            brake = all_predictions[start_idx + 2]
            
            is_current = '>>> YES' if (i == current_command - 2) else ''
            
            print(f"Branch {i:<4} {name:<10} {steer:+.3f}     {throttle:.3f}      {brake:.3f}      {is_current}")
        
        # 如果提供了后处理结果，显示对比
        if control_result is not None:
            branch_idx = current_command - 2
            start_idx = branch_idx * 3
            
            raw_steer = all_predictions[start_idx]
            raw_throttle = all_predictions[start_idx + 1]
            raw_brake = all_predictions[start_idx + 2]
            
            actual_steer = control_result['steer']
            actual_throttle = control_result['throttle']
            actual_brake = control_result['brake']
            
            print(f"\n{'='*80}")
            print(f"当前分支 (Branch {branch_idx}) 对比:")
            print(f"{'-'*80}")
            print(f"{'':20} {'Steer':>12} {'Throttle':>12} {'Brake':>12}")
            print(f"{'-'*80}")
            print(f"{'[模型原始输出]':<20} {raw_steer:>+12.3f} {raw_throttle:>12.3f} {raw_brake:>12.3f}")
            print(f"{'[后处理后]':<20} {actual_steer:>+12.3f} {actual_throttle:>12.3f} {actual_brake:>12.3f}")
            
            # 计算差异
            diff_steer = actual_steer - raw_steer
            diff_throttle = actual_throttle - raw_throttle
            diff_brake = actual_brake - raw_brake
            
            print(f"{'[差异]':<20} {diff_steer:>+12.3f} {diff_throttle:>+12.3f} {diff_brake:>+12.3f}")
        
        print(f"{'='*80}\n")
    
    def print_status(self, current_speed, control_result, route_info):
        """
        打印状态信息
        
        参数:
            current_speed: 当前速度（归一化值）
            control_result: 控制结果字典
            route_info: 路线信息字典
        """
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # 计算实际速度（km/h）
        actual_speed = current_speed * SPEED_NORMALIZATION_MPS * 3.6
        command_en = COMMAND_NAMES_EN.get(route_info['current_command'], 'Unknown')
        
        print(f"[{elapsed:.1f}s] "
              f"Cmd: {command_en:8s} | "
              f"Prog: {route_info['progress']:5.1f}% | "
              f"Dist: {route_info['remaining_distance']:4.0f}m | "
              f"Spd: {actual_speed:4.1f} | "
              f"Str: {control_result['steer']:+.3f} | "
              f"Thr: {control_result['throttle']:.3f} | "
              f"Brk: {control_result['brake']:.3f} | "
              f"FPS: {fps:.1f}")
              
    def print_statistics(self):
        """打印统计信息"""
        if self.frame_count == 0:
            return
            
        print(f"\n{'='*60}")
        print("推理统计信息")
        print(f"{'='*60}")
        print(f"总帧数: {self.frame_count}")
        print(f"平均推理时间: {self.total_inference_time/self.frame_count*1000:.2f} ms")
        print(f"{'='*60}\n")
    
    def reset(self):
        """重置统计信息"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = None
