#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant  
日期: 2025-11-03
说明: 基于命令分段的交互式数据收集
      当导航命令变化时暂停，询问用户是否保存该段数据
      每段数据按200条切片保存
'''

import os
import sys
import time
import numpy as np
import cv2

# 导入基类
from base_collector import BaseDataCollector, AGENTS_AVAILABLE


class CommandBasedDataCollector(BaseDataCollector):
    """
    基于命令分段的数据收集器
    
    特点：
    1. 检测导航命令变化
    2. 命令变化时暂停并询问是否保存
    3. 每段数据按200条切片保存
    4. 支持跳过不需要的命令段
    """
    
    def __init__(self, host='localhost', port=2000, town='Town01',
                 ignore_traffic_lights=True, ignore_signs=True, 
                 ignore_vehicles_percentage=80, target_speed=10.0, simulation_fps=20,
                 noise_enabled=False, lateral_noise=True, longitudinal_noise=False,
                 noise_ratio=0.4, max_steer_offset=0.35, max_throttle_offset=0.2,
                 noise_modes=None):
        super().__init__(host, port, town, ignore_traffic_lights, ignore_signs,
                        ignore_vehicles_percentage, target_speed, simulation_fps)
        
        # 配置噪声参数
        self.configure_noise(
            enabled=noise_enabled,
            lateral_enabled=lateral_noise,
            longitudinal_enabled=longitudinal_noise,
            noise_ratio=noise_ratio,
            max_steer_offset=max_steer_offset,
            max_throttle_offset=max_throttle_offset,
            noise_modes=noise_modes
        )
    
    def _ask_user_save_segment(self, command, show_visualization=False, 
                                current_image=None, speed=0.0, current_frame=0, total_frames=0):
        """
        询问用户是否保存当前数据段
        
        返回:
            bool: True=保存, False=丢弃, None=停止收集
        """
        if show_visualization and current_image is not None:
            self._visualize_frame(current_image, speed, command, current_frame, total_frames, 
                                paused=True, is_collecting=True)
        
        print("\n" + "="*70)
        print(f"⏸️  车辆已暂停 - 检测到命令: {self.COMMAND_NAMES.get(int(command), 'Unknown')} (命令{command})")
        print("="*70)
        print(f"\n💡 提示：车辆已停止，等待你的指令")
        print(f"请选择操作:")
        print(f"  ✅ '保存' 或 's' → 收集200帧 → 自动保存")
        print(f"  ❌ '跳过' 或 'n' → 跳过此命令段，等待命令变化")
        print(f"  ⏹️  '停止' 或 'q' → 停止收集并退出")
        
        while True:
            try:
                choice = input(f"\n👉 你的选择: ").strip().lower()
                
                if choice in ['保存', 'save', 's', 'y', 'yes']:
                    print(f"✅ 将保存这段数据")
                    print(f"▶️  车辆继续行驶...\n")
                    return True
                elif choice in ['跳过', 'skip', 'n', 'no']:
                    print(f"❌ 将丢弃这段数据")
                    print(f"▶️  车辆继续行驶...\n")
                    return False
                elif choice in ['停止', 'stop', 'q', 'quit']:
                    print(f"⏹️  停止收集")
                    return None
                else:
                    print(f"❌ 无效选择！请输入 '保存' (s)、'跳过' (n) 或 '停止' (q)")
                    
            except KeyboardInterrupt:
                print("\n⏹️  收到中断信号")
                return None
    
    def _save_segment(self, save_path, command):
        """保存当前数据段（按200条切片）"""
        if len(self.current_segment_data['rgb']) == 0:
            print("当前段无数据，跳过保存")
            return
        
        print(f"\n正在保存数据段...")
        
        rgb_array = np.array(self.current_segment_data['rgb'], dtype=np.uint8)
        targets_array = np.array(self.current_segment_data['targets'], dtype=np.float32)
        
        total_samples = rgb_array.shape[0]
        num_chunks = (total_samples + 199) // 200
        print(f"  总样本数: {total_samples}, 分割成 {num_chunks} 个文件")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        command_name = self.COMMAND_NAMES.get(int(command), 'Unknown')
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * 200
            end_idx = min((chunk_idx + 1) * 200, total_samples)
            
            chunk_rgb = rgb_array[start_idx:end_idx]
            chunk_targets = targets_array[start_idx:end_idx]
            
            self._save_data_to_h5(
                chunk_rgb.tolist(), chunk_targets.tolist(),
                save_path, command, f"_part{chunk_idx+1:03d}"
            )
        
        print(f"✅ 数据段保存完成！")
    
    def collect_data_interactive(self, max_frames=50000, save_path='./carla_data', visualize=True):
        """
        交互式数据收集
        
        工作流程：
        1. 询问是否收集当前命令段
        2. 如果选择"保存"→ 收集200帧 → 自动保存
        3. 自动保存后 → 继续询问下一段
        """
        self.enable_visualization = visualize
        
        print("\n" + "="*70)
        print("📊 基于命令的交互式数据收集")
        print("="*70)
        print(f"最大帧数: {max_frames}")
        print(f"保存路径: {save_path}")
        print(f"可视化: {'开启' if visualize else '关闭'}")
        print("="*70)
        
        os.makedirs(save_path, exist_ok=True)
        self.wait_for_first_frame()
        
        collected_frames = 0
        self.current_segment_data = {'rgb': [], 'targets': []}
        self.segment_count = 0
        
        self.current_command = self._get_navigation_command()
        
        # 获取初始画面
        for _ in range(10):
            self.step_simulation()
            time.sleep(0.05)
        
        print("\n开始数据收集循环...")
        
        try:
            while collected_frames < max_frames:
                self.current_command = self._get_navigation_command()
                
                current_image = self.image_buffer[-1] if len(self.image_buffer) > 0 else None
                current_speed = self._get_vehicle_speed()
                
                # 询问用户
                user_choice = self._ask_user_save_segment(
                    command=self.current_command,
                    show_visualization=self.enable_visualization,
                    current_image=current_image,
                    speed=current_speed,
                    current_frame=collected_frames,
                    total_frames=max_frames
                )
                
                if user_choice is None:
                    print("✅ 用户选择停止收集")
                    break
                
                if not user_choice:
                    # 跳过模式：等待命令变化
                    print(f"❌ 跳过 {self.COMMAND_NAMES[int(self.current_command)]} 命令段")
                    collected_frames = self._skip_until_command_change(collected_frames, max_frames)
                    continue
                
                # 收集200帧
                save_command = self.current_command
                print(f"✅ 开始收集 {self.COMMAND_NAMES[int(save_command)]} 命令段（目标：200帧）...")
                
                self.current_segment_data = {'rgb': [], 'targets': []}
                self.segment_count = 0
                collision_occurred = False  # 碰撞标记
                
                # 重置碰撞状态
                self.reset_collision_state()
                
                while self.segment_count < 200 and collected_frames < max_frames:
                    self.step_simulation()
                    
                    # 检测碰撞
                    if self.collision_detected:
                        print(f"💥 碰撞发生！丢弃当前segment数据（{self.segment_count}帧）")
                        collision_occurred = True
                        self.current_segment_data = {'rgb': [], 'targets': []}
                        self.segment_count = 0
                        break
                    
                    if self._is_route_completed():
                        print(f"\n🎯 已到达目的地！")
                        break
                    
                    if len(self.image_buffer) == 0:
                        continue
                    
                    current_image = self.image_buffer[-1].copy()
                    speed_kmh = self._get_vehicle_speed()
                    current_cmd = self._get_navigation_command()
                    
                    # 数据质量检查
                    if current_image.mean() < 5 or speed_kmh > 150:
                        continue
                    
                    targets = self._build_targets(speed_kmh, current_cmd)
                    
                    self.current_segment_data['rgb'].append(current_image)
                    self.current_segment_data['targets'].append(targets)
                    self.segment_count += 1
                    collected_frames += 1
                    
                    if self.enable_visualization:
                        self._visualize_frame(current_image, speed_kmh, current_cmd,
                                            collected_frames, max_frames, is_collecting=True)
                    
                    if self.segment_count % 50 == 0:
                        print(f"  [收集中] 进度: {self.segment_count}/200 帧")
                
                # 自动保存（如果没有碰撞）
                if self.segment_count > 0 and not collision_occurred:
                    print(f"\n💾 自动保存数据段（{self.segment_count} 帧）...")
                    self._save_segment(save_path, save_command)
                elif collision_occurred:
                    print(f"⚠️  因碰撞跳过保存，等待下一个命令段...")
                
                if self._is_route_completed():
                    break
            
            self._print_summary(collected_frames)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断收集...")
            if self.segment_count > 0:
                save_final = input(f"\n当前段有 {self.segment_count} 帧，是否保存？(y/n): ").strip().lower()
                if save_final in ['y', 'yes', '保存']:
                    self._save_segment(save_path, self.current_command)
        
        finally:
            if self.enable_visualization:
                cv2.destroyAllWindows()
    
    def _skip_until_command_change(self, collected_frames, max_frames):
        """跳过直到命令变化"""
        print("🔄 等待命令变化...")
        skip_frames = 0
        
        while skip_frames < 500:
            self.step_simulation()
            
            if self._is_route_completed():
                print(f"\n🎯 已到达目的地！")
                return collected_frames
            
            new_command = self._get_navigation_command()
            if new_command != self.current_command:
                print(f"✅ 命令已变化: {self.COMMAND_NAMES.get(int(self.current_command), 'Unknown')} → "
                      f"{self.COMMAND_NAMES.get(int(new_command), 'Unknown')}\n")
                break
            
            skip_frames += 1
            collected_frames += 1
            
            if self.enable_visualization and len(self.image_buffer) > 0:
                self._visualize_frame(self.image_buffer[-1], self._get_vehicle_speed(),
                                    new_command, collected_frames, max_frames, is_collecting=False)
        
        return collected_frames
    
    def _print_summary(self, collected_frames):
        """打印收集总结"""
        print(f"\n{'='*70}")
        print(f"✅ 数据收集完成！")
        print(f"{'='*70}")
        print(f"总收集帧数: {collected_frames}")
        print(f"总保存帧数: {self.total_saved_frames}")
        print(f"保存段数: {self.total_saved_segments}")
        print(f"{'='*70}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于命令的交互式数据收集')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--spawn-index', type=int, required=True)
    parser.add_argument('--dest-index', type=int, required=True)
    parser.add_argument('--max-frames', type=int, default=50000)
    parser.add_argument('--save-path', type=str, default='./carla_data')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    collector = CommandBasedDataCollector(args.host, args.port, args.town)
    
    try:
        collector.connect()
        
        if not collector.spawn_vehicle(args.spawn_index, args.dest_index):
            print("无法生成车辆！")
            return
        
        collector.setup_camera()
        collector.setup_collision_sensor()  # 设置碰撞传感器
        time.sleep(1.0)
        
        collector.collect_data_interactive(
            max_frames=args.max_frames,
            save_path=args.save_path,
            visualize=args.visualize
        )
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        collector.cleanup()


if __name__ == '__main__':
    main()
