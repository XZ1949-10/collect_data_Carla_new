#!/usr/bin/env python
# coding=utf-8
"""
可视化工具

提供数据收集过程中的实时可视化和H5数据查看功能。
"""

import os
import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple

from ..config import COMMAND_NAMES, COMMAND_COLORS


class FrameVisualizer:
    """帧可视化器（用于数据收集过程）"""
    
    def __init__(self, window_name: str = "Data Collection", target_speed: float = 10.0):
        self.window_name = window_name
        self.target_speed = target_speed
        self._window_created = False
        
        # 统计信息
        self.total_saved_frames = 0
        self.total_saved_segments = 0
    
    def update_statistics(self, saved_frames: int, saved_segments: int):
        """更新统计信息"""
        self.total_saved_frames = saved_frames
        self.total_saved_segments = saved_segments
    
    def visualize_frame(self, image: np.ndarray, speed: float, command: int,
                        current_frame: int, total_frames: int,
                        segment_count: int = 0,
                        paused: bool = False, is_collecting: bool = True,
                        noise_info: Optional[Dict] = None,
                        control_info: Optional[Dict] = None,
                        expert_control: Optional[Dict] = None):
        """
        可视化当前帧
        
        参数:
            image: RGB图像
            speed: 当前速度 (km/h)
            command: 导航命令
            current_frame: 当前帧号
            total_frames: 总帧数
            segment_count: 当前段帧数
            paused: 是否暂停
            is_collecting: 是否正在收集
            noise_info: 噪声信息 {enabled, lateral_enabled, longitudinal_enabled, 
                                  lateral_active, longitudinal_active,
                                  steer_noise, throttle_noise}
            control_info: 实际控制信息 {steer, throttle, brake}
            expert_control: 专家控制信息 {steer, throttle, brake}
        """
        command = int(command)
        
        # 放大图像
        display_image = cv2.resize(image, (800, 600))
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        
        if paused:
            overlay = display_image.copy()
            cv2.rectangle(overlay, (0, 0), (800, 600), (0, 0, 0), -1)
            display_image = cv2.addWeighted(display_image, 0.6, overlay, 0.4, 0)
        
        # 创建信息面板
        info_panel = self._create_info_panel(
            speed, command, current_frame, total_frames, segment_count,
            paused, is_collecting, noise_info, control_info, expert_control
        )
        
        combined = np.hstack([display_image, info_panel])
        
        if paused:
            cv2.putText(combined, "PAUSED", (300, 300), 
                       cv2.FONT_HERSHEY_DUPLEX, 2, (0, 165, 255), 4)
            cv2.putText(combined, "Waiting for your command...", (150, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
        
        cv2.imshow(self.window_name, combined)
        cv2.waitKey(1)
    
    def _create_info_panel(self, speed: float, command: int,
                           current_frame: int, total_frames: int,
                           segment_count: int,
                           paused: bool, is_collecting: bool,
                           noise_info: Optional[Dict],
                           control_info: Optional[Dict],
                           expert_control: Optional[Dict]) -> np.ndarray:
        """创建信息面板"""
        panel_width = 320
        panel_height = 600
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 25
        
        # 标题
        cv2.putText(panel, "Data Collection", (10, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 25
        
        # 状态
        if paused:
            cv2.putText(panel, "*** PAUSED ***", (10, y_pos), font, 0.7, (0, 165, 255), 2)
        else:
            status_text = "SAVING" if is_collecting else "SKIPPING"
            status_color = (100, 255, 100) if is_collecting else (100, 100, 255)
            cv2.putText(panel, f"*** {status_text} ***", (10, y_pos), font, 0.6, status_color, 2)
        y_pos += 25
        
        # 进度
        cv2.putText(panel, f"Progress: {current_frame}/{total_frames}", (10, y_pos), 
                   font, 0.45, (200, 200, 200), 1)
        y_pos += 20
        cv2.putText(panel, f"Segment: {segment_count} frames", (10, y_pos), 
                   font, 0.45, (200, 200, 200), 1)
        y_pos += 28
        
        # 命令
        cmd_name = COMMAND_NAMES.get(command, 'Unknown')
        cmd_color = COMMAND_COLORS.get(command, (255, 255, 255))
        cv2.putText(panel, f"Command: {cmd_name}", (10, y_pos), font, 0.6, cmd_color, 2)
        y_pos += 28
        
        # 速度
        speed_color = (100, 255, 100) if speed < 60 else (255, 200, 100)
        cv2.putText(panel, f"Speed: {speed:.1f} km/h", (10, y_pos), font, 0.5, speed_color, 1)
        y_pos += 20
        cv2.putText(panel, f"Target: {self.target_speed:.1f} km/h", (10, y_pos), 
                   font, 0.4, (150, 150, 150), 1)
        y_pos += 25
        
        # === 专家控制（标签值）===
        if expert_control:
            cv2.putText(panel, "=== Label (Dataset) ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
            y_pos += 22
            cv2.putText(panel, f"Steer: {expert_control.get('steer', 0):+.3f}", (10, y_pos), 
                       font, 0.5, (100, 200, 255), 1)
            y_pos += 20
            cv2.putText(panel, f"Throttle: {expert_control.get('throttle', 0):.3f}", (10, y_pos), 
                       font, 0.5, (100, 255, 100), 1)
            y_pos += 20
            cv2.putText(panel, f"Brake: {expert_control.get('brake', 0):.3f}", (10, y_pos), 
                       font, 0.5, (150, 150, 150), 1)
            y_pos += 25
        
        # === 噪声信息 ===
        cv2.putText(panel, "=== Noise ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        
        if noise_info and noise_info.get('enabled', False):
            # 横向噪声状态
            lat_enabled = noise_info.get('lateral_enabled', False)
            lat_active = noise_info.get('lateral_active', False)
            lat_status = "ON" if lat_enabled else "OFF"
            lat_color = (0, 165, 255) if lat_active else ((100, 255, 100) if lat_enabled else (150, 150, 150))
            lat_indicator = " [ACTIVE]" if lat_active else ""
            cv2.putText(panel, f"Lateral: {lat_status}{lat_indicator}", (10, y_pos), font, 0.4, lat_color, 1)
            y_pos += 18
            
            # 纵向噪声状态
            lon_enabled = noise_info.get('longitudinal_enabled', False)
            lon_active = noise_info.get('longitudinal_active', False)
            lon_status = "ON" if lon_enabled else "OFF"
            lon_color = (0, 165, 255) if lon_active else ((100, 255, 100) if lon_enabled else (150, 150, 150))
            lon_indicator = " [ACTIVE]" if lon_active else ""
            cv2.putText(panel, f"Longitudinal: {lon_status}{lon_indicator}", (10, y_pos), font, 0.4, lon_color, 1)
            y_pos += 22
            
            # 噪声公式显示
            if expert_control and control_info:
                expert_steer = expert_control.get('steer', 0)
                actual_steer = control_info.get('steer', 0)
                steer_noise = actual_steer - expert_steer
                
                expert_throttle = expert_control.get('throttle', 0)
                actual_throttle = control_info.get('throttle', 0)
                throttle_noise = actual_throttle - expert_throttle
                
                # 转向噪声公式
                cv2.putText(panel, "--- Steer ---", (10, y_pos), font, 0.35, (180, 180, 180), 1)
                y_pos += 16
                steer_color = (0, 165, 255) if abs(steer_noise) > 0.01 else (150, 150, 150)
                cv2.putText(panel, f"{expert_steer:+.2f} + ({steer_noise:+.2f}) = {actual_steer:+.2f}", 
                           (10, y_pos), font, 0.4, steer_color, 1)
                y_pos += 18
                
                # 油门噪声公式
                cv2.putText(panel, "--- Throttle ---", (10, y_pos), font, 0.35, (180, 180, 180), 1)
                y_pos += 16
                throttle_color = (0, 165, 255) if abs(throttle_noise) > 0.01 else (150, 150, 150)
                cv2.putText(panel, f"{expert_throttle:.2f} + ({throttle_noise:+.2f}) = {actual_throttle:.2f}", 
                           (10, y_pos), font, 0.4, throttle_color, 1)
                y_pos += 20
        else:
            cv2.putText(panel, "Noise: OFF", (10, y_pos), font, 0.45, (150, 150, 150), 1)
            y_pos += 20
        
        y_pos += 8
        
        # === 实际控制（车辆执行的值）===
        if control_info:
            cv2.putText(panel, "=== Actual Control ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
            y_pos += 22
            
            actual_steer = control_info.get('steer', 0)
            actual_throttle = control_info.get('throttle', 0)
            actual_brake = control_info.get('brake', 0)
            
            # 判断是否有噪声偏移
            has_noise = noise_info and noise_info.get('enabled', False)
            expert_steer = expert_control.get('steer', 0) if expert_control else actual_steer
            expert_throttle = expert_control.get('throttle', 0) if expert_control else actual_throttle
            
            steer_color = (100, 100, 255) if (has_noise and abs(actual_steer - expert_steer) > 0.01) else (100, 200, 255)
            cv2.putText(panel, f"Steer: {actual_steer:+.3f}", (10, y_pos), font, 0.45, steer_color, 1)
            y_pos += 18
            
            throttle_color = (100, 100, 255) if (has_noise and abs(actual_throttle - expert_throttle) > 0.01) else (100, 255, 100)
            cv2.putText(panel, f"Throttle: {actual_throttle:.3f}", (10, y_pos), font, 0.45, throttle_color, 1)
            y_pos += 18
            
            cv2.putText(panel, f"Brake: {actual_brake:.3f}", (10, y_pos), font, 0.45, (150, 150, 150), 1)
            y_pos += 22
        
        # === 统计信息 ===
        cv2.putText(panel, "=== Statistics ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        cv2.putText(panel, f"Saved: {self.total_saved_frames}", (10, y_pos), font, 0.45, (100, 255, 100), 1)
        y_pos += 18
        cv2.putText(panel, f"Segments: {self.total_saved_segments}", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        
        return panel
    
    def close(self):
        """关闭窗口"""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class H5DataVisualizer:
    """H5数据可视化器"""
    
    def __init__(self, h5_file_path: str, auto_start: bool = False):
        """
        初始化可视化器
        
        参数:
            h5_file_path: H5文件路径
            auto_start: 是否自动开始播放（不需要按空格）
        """
        self.h5_file_path = h5_file_path
        self.rgb_data = None
        self.targets_data = None
        self.current_frame = 0
        self.total_frames = 0
        self.playing = auto_start
        self.play_speed = 20  # 毫秒/帧
        self.auto_next = False  # 播放完后是否自动跳到下一个文件
    
    def load_data(self) -> bool:
        """加载H5数据"""
        import h5py
        
        if not os.path.exists(self.h5_file_path):
            print(f"❌ 文件不存在: {self.h5_file_path}")
            return False
        
        try:
            with h5py.File(self.h5_file_path, 'r') as hf:
                self.rgb_data = hf['rgb'][:]
                self.targets_data = hf['targets'][:]
            
            self.total_frames = self.rgb_data.shape[0]
            
            print(f"✅ 数据加载成功！")
            print(f"  • RGB shape: {self.rgb_data.shape}")
            print(f"  • Targets shape: {self.targets_data.shape}")
            print(f"  • 总帧数: {self.total_frames}")
            
            # 打印统计信息
            self._print_statistics()
            
            return True
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def _print_statistics(self):
        """打印数据统计信息"""
        print(f"\n📊 数据统计:")
        
        # 速度统计
        speeds = self.targets_data[:, 10]
        print(f"  • 速度范围: {speeds.min():.1f} - {speeds.max():.1f} km/h")
        print(f"  • 平均速度: {speeds.mean():.1f} km/h")
        
        # 命令分布
        commands = self.targets_data[:, 24]
        unique_commands = np.unique(commands)
        print(f"  • 命令分布:")
        for cmd in unique_commands:
            count = np.sum(commands == cmd)
            percentage = count / len(commands) * 100
            cmd_name = COMMAND_NAMES.get(cmd, f'Unknown({cmd})')
            print(f"    - {cmd_name}: {count} 帧 ({percentage:.1f}%)")
        
        # 控制信号统计
        steers = self.targets_data[:, 0]
        throttles = self.targets_data[:, 1]
        brakes = self.targets_data[:, 2]
        
        print(f"  • 方向盘: {steers.min():.3f} - {steers.max():.3f}")
        print(f"  • 油门: {throttles.min():.3f} - {throttles.max():.3f}")
        print(f"  • 刹车: {brakes.min():.3f} - {brakes.max():.3f}")
        print()
    
    def visualize(self) -> Optional[str]:
        """
        启动可视化窗口
        
        返回:
            str: 'quit', 'next', 'prev' 或 None
        """
        if self.rgb_data is None:
            print("❌ 请先加载数据！")
            return None
        
        print("\n🎬 启动可视化窗口...")
        print("操作说明:")
        print("  • 空格键: 播放/暂停")
        print("  • A/D键: 上一帧/下一帧")
        print("  • W/S键: 加速/减速")
        print("  • H键: 跳到第一帧")
        print("  • E键: 跳到最后一帧")
        print("  • N键: 下一个文件")
        print("  • P键: 上一个文件")
        print("  • Q或ESC: 退出\n")
        
        window_name = "H5 Data Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        result = None
        
        while True:
            rgb_frame = self.rgb_data[self.current_frame].copy()
            targets = self.targets_data[self.current_frame]
            
            # 在图像上显示帧号
            cv2.putText(rgb_frame, f"Frame: {self.current_frame}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # 放大图像
            display_image = cv2.resize(rgb_frame, (800, 600))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            
            # 创建信息面板
            info_panel = self._create_info_panel(targets)
            combined = np.hstack([display_image, info_panel])
            
            cv2.imshow(window_name, combined)
            
            wait_time = self.play_speed if self.playing else 1
            key = cv2.waitKey(wait_time) & 0xFF
            
            # 自动播放
            if self.playing and key == 255:
                self.current_frame += 1
                if self.current_frame >= self.total_frames:
                    if self.auto_next:
                        result = 'next'
                        break
                    else:
                        self.current_frame = 0  # 循环播放
                continue
            
            # 按键处理
            if key == 27 or key == ord('q') or key == ord('Q'):
                result = 'quit'
                break
            elif key == ord('n') or key == ord('N'):
                print("跳到下一个文件")
                result = 'next'
                break
            elif key == ord('p') or key == ord('P'):
                print("跳到上一个文件")
                result = 'prev'
                break
            elif key == 32:  # Space
                self.playing = not self.playing
                status = "播放" if self.playing else "暂停"
                print(f"状态: {status}")
            elif key == ord('a') or key == ord('A'):
                self.current_frame = max(0, self.current_frame - 1)
                self.playing = False
            elif key == ord('d') or key == ord('D'):
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                self.playing = False
            elif key == ord('w') or key == ord('W'):
                self.play_speed = max(10, self.play_speed - 10)
                print(f"播放速度: {1000/self.play_speed:.1f} FPS")
            elif key == ord('s') or key == ord('S'):
                self.play_speed = min(200, self.play_speed + 10)
                print(f"播放速度: {1000/self.play_speed:.1f} FPS")
            elif key == ord('h') or key == ord('H'):
                self.current_frame = 0
                self.playing = False
                print("跳到第一帧")
            elif key == ord('e') or key == ord('E'):
                self.current_frame = self.total_frames - 1
                self.playing = False
                print("跳到最后一帧")
        
        cv2.destroyAllWindows()
        return result
    
    def _create_info_panel(self, targets: np.ndarray) -> np.ndarray:
        """创建信息面板"""
        panel_width = 400
        panel_height = 600
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        
        steer = targets[0]
        throttle = targets[1]
        brake = targets[2]
        speed = targets[10]
        command = targets[24]
        
        # 标题
        cv2.putText(panel, "H5 Data Viewer", (10, y_pos), font, 0.8, (255, 255, 255), 2)
        y_pos += 40
        
        # 文件名
        filename = os.path.basename(self.h5_file_path)
        if len(filename) > 30:
            filename = filename[:27] + "..."
        cv2.putText(panel, filename, (10, y_pos), font, 0.4, (200, 200, 200), 1)
        y_pos += 30
        
        # 进度条
        progress = self.current_frame / max(self.total_frames - 1, 1)
        bar_width = 380
        bar_height = 20
        cv2.rectangle(panel, (10, y_pos), (10 + bar_width, y_pos + bar_height), (80, 80, 80), -1)
        fill_width = int(bar_width * progress)
        cv2.rectangle(panel, (10, y_pos), (10 + fill_width, y_pos + bar_height), (100, 200, 255), -1)
        
        progress_text = f"{self.current_frame + 1}/{self.total_frames}"
        cv2.putText(panel, progress_text, (10 + bar_width // 2 - 40, y_pos + 15), 
                   font, 0.5, (255, 255, 255), 1)
        y_pos += 40
        
        # 分隔线
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        
        # 命令
        cmd_name = COMMAND_NAMES.get(command, f'Unknown({command})')
        cmd_color = COMMAND_COLORS.get(command, (255, 255, 255))
        cv2.putText(panel, "Command:", (10, y_pos), font, 0.6, (200, 200, 200), 1)
        y_pos += 30
        cv2.putText(panel, cmd_name, (10, y_pos), font, 1.0, cmd_color, 2)
        y_pos += 50
        
        # 速度
        speed_color = (100, 255, 100) if speed < 60 else (255, 200, 100)
        cv2.putText(panel, "Speed:", (10, y_pos), font, 0.6, (200, 200, 200), 1)
        y_pos += 30
        cv2.putText(panel, f"{speed:.1f} km/h", (10, y_pos), font, 1.0, speed_color, 2)
        y_pos += 40
        
        # 分隔线
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        
        # 控制信号
        cv2.putText(panel, "Control Signals:", (10, y_pos), font, 0.6, (200, 200, 200), 1)
        y_pos += 35
        
        # 方向盘（带数值条）
        steer_color = (100, 255, 100) if abs(steer) < 0.3 else (255, 200, 100)
        cv2.putText(panel, f"Steer:    {steer:+.3f}", (10, y_pos), font, 0.6, steer_color, 1)
        self._draw_bar(panel, 200, y_pos - 15, 180, 15, steer, -1.0, 1.0, steer_color)
        y_pos += 35
        
        # 油门
        throttle_color = (100, 255, 100)
        cv2.putText(panel, f"Throttle: {throttle:.3f}", (10, y_pos), font, 0.6, throttle_color, 1)
        self._draw_bar(panel, 200, y_pos - 15, 180, 15, throttle, 0.0, 1.0, throttle_color)
        y_pos += 35
        
        # 刹车
        brake_color = (100, 100, 255) if brake > 0.1 else (100, 255, 100)
        cv2.putText(panel, f"Brake:    {brake:.3f}", (10, y_pos), font, 0.6, brake_color, 1)
        self._draw_bar(panel, 200, y_pos - 15, 180, 15, brake, 0.0, 1.0, brake_color)
        y_pos += 40
        
        # 分隔线
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        
        # 操作提示
        cv2.putText(panel, "Controls:", (10, y_pos), font, 0.6, (200, 200, 200), 1)
        y_pos += 30
        controls = [
            "Space - Play/Pause",
            "A/D - Prev/Next frame",
            "W/S - Speed +/-",
            "H - First frame",
            "E - Last frame",
            "N/P - Next/Prev file",
            "Q/ESC - Quit"
        ]
        for ctrl in controls:
            cv2.putText(panel, ctrl, (10, y_pos), font, 0.4, (150, 150, 150), 1)
            y_pos += 22
        
        # 播放状态和速度
        status = "[PLAYING]" if self.playing else "[PAUSED]"
        status_color = (100, 255, 100) if self.playing else (255, 200, 100)
        cv2.putText(panel, status, (10, panel_height - 40), font, 0.6, status_color, 2)
        
        fps_text = f"Speed: {1000/self.play_speed:.1f} FPS"
        cv2.putText(panel, fps_text, (10, panel_height - 15), font, 0.4, (150, 150, 150), 1)
        
        return panel
    
    def _draw_bar(self, image: np.ndarray, x: int, y: int, width: int, height: int,
                  value: float, min_val: float, max_val: float, color: Tuple[int, int, int]):
        """
        绘制数值条
        
        参数:
            image: 图像
            x, y: 起始位置
            width, height: 条的宽度和高度
            value: 当前值
            min_val, max_val: 值范围
            color: 颜色
        """
        # 背景
        cv2.rectangle(image, (x, y), (x + width, y + height), (80, 80, 80), -1)
        
        # 计算填充宽度
        if min_val < 0:  # 双向条（如方向盘）
            center_x = x + width // 2
            if value >= 0:
                fill_width = int((width // 2) * (value / max_val))
                cv2.rectangle(image, (center_x, y), 
                            (center_x + fill_width, y + height), color, -1)
            else:
                fill_width = int((width // 2) * (value / min_val))
                cv2.rectangle(image, (center_x - fill_width, y), 
                            (center_x, y + height), color, -1)
            # 中心线
            cv2.line(image, (center_x, y), (center_x, y + height), (200, 200, 200), 1)
        else:  # 单向条（如油门、刹车）
            normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
            fill_width = int(width * normalized)
            cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1)
