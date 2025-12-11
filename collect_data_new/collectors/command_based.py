#!/usr/bin/env python
# coding=utf-8
"""
基于命令分段的数据收集器

当导航命令变化时暂停，询问用户是否保存该段数据。
"""

import os
import time
import numpy as np
import cv2
from typing import Optional

from ..config import CollectorConfig, COMMAND_NAMES
from ..core import BaseDataCollector
from ..utils import DataSaver, FrameVisualizer


class CommandBasedCollector(BaseDataCollector):
    """
    基于命令分段的数据收集器
    
    特点：
    1. 检测导航命令变化
    2. 命令变化时暂停并询问是否保存
    3. 每段数据按200条切片保存
    """
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        super().__init__(config)
        
        self._data_saver: Optional[DataSaver] = None
        self._visualizer: Optional[FrameVisualizer] = None
    
    def _ask_user_save_segment(self, command: float, current_image: Optional[np.ndarray] = None,
                                speed: float = 0.0, current_frame: int = 0, 
                                total_frames: int = 0) -> Optional[bool]:
        """
        询问用户是否保存当前数据段
        
        返回:
            bool: True=保存, False=丢弃, None=停止收集
        """
        if self._visualizer and current_image is not None:
            # 获取可视化信息（低耦合方式）
            vis_info = self.get_visualization_info()
            self._visualizer.visualize_frame(
                current_image, speed, int(command),
                current_frame, total_frames,
                self.segment_count, paused=True, is_collecting=True,
                noise_info=vis_info.to_noise_info(),
                control_info=vis_info.to_control_info(),
                expert_control=vis_info.to_expert_control()
            )
        
        print("\n" + "="*70)
        print(f"⏸️  车辆已暂停 - 检测到命令: {COMMAND_NAMES.get(int(command), 'Unknown')}")
        print("="*70)
        print(f"\n请选择操作:")
        print(f"  ✅ '保存' 或 's' → 收集200帧 → 自动保存")
        print(f"  ❌ '跳过' 或 'n' → 跳过此命令段")
        print(f"  ⏹️  '停止' 或 'q' → 停止收集")
        
        while True:
            try:
                choice = input(f"\n👉 你的选择: ").strip().lower()
                
                if choice in ['保存', 'save', 's', 'y', 'yes']:
                    print(f"✅ 将保存这段数据")
                    return True
                elif choice in ['跳过', 'skip', 'n', 'no']:
                    print(f"❌ 将丢弃这段数据")
                    return False
                elif choice in ['停止', 'stop', 'q', 'quit']:
                    print(f"⏹️  停止收集")
                    return None
                else:
                    print(f"❌ 无效选择！")
            except KeyboardInterrupt:
                return None
    
    def collect_data_interactive(self, max_frames: int = 50000, 
                                  save_path: str = './carla_data',
                                  visualize: bool = True):
        """
        交互式数据收集
        
        参数:
            max_frames: 最大帧数
            save_path: 保存路径
            visualize: 是否可视化
        """
        self.config.enable_visualization = visualize
        
        print("\n" + "="*70)
        print("📊 基于命令的交互式数据收集")
        print("="*70)
        
        os.makedirs(save_path, exist_ok=True)
        self._data_saver = DataSaver(save_path, self.config.segment_size)
        
        if visualize:
            self._visualizer = FrameVisualizer()
        
        self.wait_for_first_frame()
        
        collected_frames = 0
        self.current_segment_data = {'rgb': [], 'targets': []}
        self.segment_count = 0
        
        self.current_command = self.get_navigation_command()
        
        # 预热
        for _ in range(10):
            self.step_simulation()
            time.sleep(0.05)
        
        print("\n开始数据收集循环...")
        
        try:
            while collected_frames < max_frames:
                self.current_command = self.get_navigation_command()
                
                current_image = self.image_buffer[-1] if len(self.image_buffer) > 0 else None
                current_speed = self.get_vehicle_speed()
                
                # 询问用户
                user_choice = self._ask_user_save_segment(
                    command=self.current_command,
                    current_image=current_image,
                    speed=current_speed,
                    current_frame=collected_frames,
                    total_frames=max_frames
                )
                
                if user_choice is None:
                    break
                
                if not user_choice:
                    collected_frames = self._skip_until_command_change(collected_frames, max_frames)
                    continue
                
                # 收集200帧
                save_command = self.current_command
                print(f"✅ 开始收集 {COMMAND_NAMES[int(save_command)]} 命令段...")
                
                self.current_segment_data = {'rgb': [], 'targets': []}
                self.segment_count = 0
                self.reset_collision_state()
                
                while self.segment_count < 200 and collected_frames < max_frames:
                    self.step_simulation()
                    
                    if self.collision_detected:
                        print(f"💥 碰撞！丢弃当前数据")
                        self.current_segment_data = {'rgb': [], 'targets': []}
                        self.segment_count = 0
                        break
                    
                    if self.is_route_completed():
                        print(f"\n🎯 已到达目的地！")
                        break
                    
                    if len(self.image_buffer) == 0:
                        continue
                    
                    current_image = self.image_buffer[-1].copy()
                    speed_kmh = self.get_vehicle_speed()
                    current_cmd = self.get_navigation_command()
                    
                    if current_image.mean() < 5 or speed_kmh > 150:
                        continue
                    
                    targets = self.build_targets(speed_kmh, current_cmd)
                    
                    self.current_segment_data['rgb'].append(current_image)
                    self.current_segment_data['targets'].append(targets)
                    self.segment_count += 1
                    collected_frames += 1
                    
                    if self._visualizer:
                        # 获取可视化信息（低耦合方式）
                        vis_info = self.get_visualization_info()
                        self._visualizer.visualize_frame(
                            current_image, speed_kmh, int(current_cmd),
                            collected_frames, max_frames, self.segment_count,
                            is_collecting=True,
                            noise_info=vis_info.to_noise_info(),
                            control_info=vis_info.to_control_info(),
                            expert_control=vis_info.to_expert_control()
                        )
                
                # 保存
                if self.segment_count > 0:
                    self._data_saver.save_segment_chunked(
                        self.current_segment_data['rgb'],
                        self.current_segment_data['targets'],
                        save_command
                    )
                
                if self.is_route_completed():
                    break
            
            self._print_summary(collected_frames)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断...")
        finally:
            if self._visualizer:
                self._visualizer.close()
    
    def _skip_until_command_change(self, collected_frames: int, max_frames: int) -> int:
        """跳过直到命令变化"""
        print("🔄 等待命令变化...")
        skip_frames = 0
        
        while skip_frames < 500:
            self.step_simulation()
            
            if self.is_route_completed():
                return collected_frames
            
            new_command = self.get_navigation_command()
            if new_command != self.current_command:
                print(f"✅ 命令已变化")
                break
            
            skip_frames += 1
            collected_frames += 1
            
            if self._visualizer and len(self.image_buffer) > 0:
                # 获取可视化信息（低耦合方式）
                vis_info = self.get_visualization_info()
                self._visualizer.visualize_frame(
                    self.image_buffer[-1], self.get_vehicle_speed(),
                    int(new_command), collected_frames, max_frames,
                    is_collecting=False,
                    noise_info=vis_info.to_noise_info(),
                    control_info=vis_info.to_control_info(),
                    expert_control=vis_info.to_expert_control()
                )
        
        return collected_frames
    
    def _print_summary(self, collected_frames: int):
        """打印收集总结"""
        print(f"\n{'='*70}")
        print(f"✅ 数据收集完成！")
        print(f"{'='*70}")
        print(f"总收集帧数: {collected_frames}")
        if self._data_saver:
            stats = self._data_saver.get_statistics()
            print(f"总保存帧数: {stats['total_frames']}")
            print(f"保存段数: {stats['total_segments']}")
