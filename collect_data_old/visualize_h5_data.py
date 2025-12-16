#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant
日期: 2025-12-01
说明: H5数据可视化工具
      在弹窗中查看收集到的CARLA数据（图像、速度、控制信号等）
'''

import os
import sys
import h5py
import numpy as np
import cv2
import argparse
from collections import defaultdict


class H5DataVisualizer:
    """H5数据可视化器"""
    
    def __init__(self, h5_file_path, auto_start=False):
        """
        初始化可视化器
        
        参数:
            h5_file_path (str): H5文件路径
            auto_start (bool): 是否自动开始播放（不需要按空格）
        """
        self.h5_file_path = h5_file_path
        self.data = None
        self.rgb_data = None
        self.targets_data = None
        self.current_frame = 0
        self.total_frames = 0
        self.playing = auto_start  # 如果auto_start为True，直接开始播放
        self.play_speed = 20  # 毫秒/帧
        self.auto_next = False  # 是否在播放完后自动跳到下一个文件
        
        # 命令名称映射（只有4个有效命令）
        self.command_names = {
            2.0: 'Follow',
            3.0: 'Left',
            4.0: 'Right',
            5.0: 'Straight'
        }
        
        # 命令颜色映射
        self.command_colors = {
            2.0: (100, 255, 100),  # 绿色
            3.0: (100, 100, 255),  # 蓝色
            4.0: (255, 100, 100),  # 红色
            5.0: (255, 255, 100)   # 黄色
        }
        
    def load_data(self):
        """加载H5数据"""
        print(f"\n正在加载数据: {self.h5_file_path}")
        
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
            
            # 统计信息
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
            cmd_name = self.command_names.get(cmd, f'Unknown({cmd})')
            print(f"    - {cmd_name}: {count} 帧 ({percentage:.1f}%)")
        
        # 控制信号统计
        steers = self.targets_data[:, 0]
        throttles = self.targets_data[:, 1]
        brakes = self.targets_data[:, 2]
        
        print(f"  • 方向盘: {steers.min():.3f} - {steers.max():.3f}")
        print(f"  • 油门: {throttles.min():.3f} - {throttles.max():.3f}")
        print(f"  • 刹车: {brakes.min():.3f} - {brakes.max():.3f}")
        print()
    
    def _create_info_panel(self, frame_idx):
        """
        创建信息面板
        
        参数:
            frame_idx (int): 当前帧索引
            
        返回:
            np.ndarray: 信息面板图像
        """
        panel_width = 400
        panel_height = 600
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        
        # 获取当前帧数据
        targets = self.targets_data[frame_idx]
        steer = targets[0]
        throttle = targets[1]
        brake = targets[2]
        speed = targets[10]
        command = targets[24]
        
        # 标题
        cv2.putText(panel, "H5 Data Viewer", (10, y_pos), 
                   font, 0.8, (255, 255, 255), 2)
        y_pos += 50
        
        # 文件名 - 支持两行显示
        filename = os.path.basename(self.h5_file_path)
        max_chars_per_line = 45
        if len(filename) <= max_chars_per_line:
            cv2.putText(panel, filename, (10, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 40
        else:
            # 第一行
            cv2.putText(panel, filename[:max_chars_per_line], (10, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 20
            # 第二行（剩余部分）
            remaining = filename[max_chars_per_line:]
            if len(remaining) > max_chars_per_line:
                remaining = remaining[:max_chars_per_line-3] + "..."
            cv2.putText(panel, remaining, (10, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 25
        
        # 进度条
        progress = frame_idx / max(self.total_frames - 1, 1)
        bar_width = 380
        bar_height = 20
        bar_x = 10
        bar_y = y_pos
        
        # 绘制进度条背景
        cv2.rectangle(panel, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (80, 80, 80), -1)
        
        # 绘制进度条填充
        fill_width = int(bar_width * progress)
        cv2.rectangle(panel, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), 
                     (100, 200, 255), -1)
        
        # 进度文字
        progress_text = f"{frame_idx + 1}/{self.total_frames}"
        cv2.putText(panel, progress_text, (bar_x + bar_width // 2 - 40, bar_y + 15), 
                   font, 0.5, (255, 255, 255), 1)
        
        y_pos += 50
        
        # 分隔线
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), 
                (100, 100, 100), 1)
        y_pos += 30
        
        # 命令信息
        cmd_name = self.command_names.get(command, f'Unknown({command})')
        cmd_color = self.command_colors.get(command, (255, 255, 255))
        
        cv2.putText(panel, "Command:", (10, y_pos), 
                   font, 0.6, (200, 200, 200), 1)
        y_pos += 30
        cv2.putText(panel, cmd_name, (10, y_pos), 
                   font, 1.0, cmd_color, 2)
        y_pos += 50
        
        # 速度信息
        cv2.putText(panel, "Speed:", (10, y_pos), 
                   font, 0.6, (200, 200, 200), 1)
        y_pos += 30
        speed_color = (100, 255, 100) if speed < 60 else (255, 200, 100)
        cv2.putText(panel, f"{speed:.1f} km/h", (10, y_pos), 
                   font, 1.0, speed_color, 2)
        y_pos += 50
        
        # 分隔线
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), 
                (100, 100, 100), 1)
        y_pos += 30
        
        # 控制信号
        cv2.putText(panel, "Control Signals:", (10, y_pos), 
                   font, 0.6, (200, 200, 200), 1)
        y_pos += 35
        
        # 方向盘
        steer_color = (100, 255, 100) if abs(steer) < 0.3 else (255, 200, 100)
        cv2.putText(panel, f"Steer:    {steer:+.3f}", (10, y_pos), 
                   font, 0.6, steer_color, 1)
        self._draw_bar(panel, 200, y_pos - 15, 180, 15, steer, -1.0, 1.0, steer_color)
        y_pos += 35
        
        # 油门
        throttle_color = (100, 255, 100)
        cv2.putText(panel, f"Throttle: {throttle:.3f}", (10, y_pos), 
                   font, 0.6, throttle_color, 1)
        self._draw_bar(panel, 200, y_pos - 15, 180, 15, throttle, 0.0, 1.0, throttle_color)
        y_pos += 35
        
        # 刹车
        brake_color = (100, 100, 255) if brake > 0.1 else (100, 255, 100)
        cv2.putText(panel, f"Brake:    {brake:.3f}", (10, y_pos), 
                   font, 0.6, brake_color, 1)
        self._draw_bar(panel, 200, y_pos - 15, 180, 15, brake, 0.0, 1.0, brake_color)
        y_pos += 50
        
        # 分隔线
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), 
                (100, 100, 100), 1)
        y_pos += 30
        
        # 操作提示
        cv2.putText(panel, "Controls:", (10, y_pos), 
                   font, 0.6, (200, 200, 200), 1)
        y_pos += 30
        
        controls = [
            "Space - Play/Pause",
            "A/D - Prev/Next frame",
            "W/S - Speed +/-",
            "H - First frame",
            "E - Last frame",
            "Q/ESC - Quit"
        ]
        
        for control in controls:
            cv2.putText(panel, control, (10, y_pos), 
                       font, 0.4, (150, 150, 150), 1)
            y_pos += 25
        
        # 播放状态
        if self.playing:
            cv2.putText(panel, "[PLAYING]", (10, panel_height - 20), 
                       font, 0.6, (100, 255, 100), 2)
        else:
            cv2.putText(panel, "[PAUSED]", (10, panel_height - 20), 
                       font, 0.6, (255, 200, 100), 2)
        
        return panel
    
    def _draw_bar(self, image, x, y, width, height, value, min_val, max_val, color):
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
        cv2.rectangle(image, (x, y), (x + width, y + height), 
                     (80, 80, 80), -1)
        
        # 计算填充宽度
        if min_val < 0:  # 双向条（如方向盘）
            center_x = x + width // 2
            if value >= 0:
                fill_width = int((width // 2) * (value / max_val))
                cv2.rectangle(image, (center_x, y), 
                            (center_x + fill_width, y + height), 
                            color, -1)
            else:
                fill_width = int((width // 2) * (value / min_val))
                cv2.rectangle(image, (center_x - fill_width, y), 
                            (center_x, y + height), 
                            color, -1)
            # 中心线
            cv2.line(image, (center_x, y), (center_x, y + height), 
                    (200, 200, 200), 1)
        else:  # 单向条（如油门、刹车）
            normalized = (value - min_val) / (max_val - min_val)
            fill_width = int(width * normalized)
            cv2.rectangle(image, (x, y), (x + fill_width, y + height), 
                         color, -1)
    
    def visualize(self):
        """
        启动可视化窗口
        
        返回:
            str: 'quit' 退出, 'next' 下一个文件, 'prev' 上一个文件, None 正常结束
        """
        if self.rgb_data is None or self.targets_data is None:
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
            # 获取当前帧
            rgb_frame = self.rgb_data[self.current_frame].copy()
            
            # 调试：在图像上显示帧号
            cv2.putText(rgb_frame, f"Frame: {self.current_frame}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # 放大图像
            display_image = cv2.resize(rgb_frame, (800, 600))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            
            # 创建信息面板
            info_panel = self._create_info_panel(self.current_frame)
            
            # 合并图像和信息面板
            combined = np.hstack([display_image, info_panel])
            
            # 显示
            cv2.imshow(window_name, combined)
            
            # 处理按键 - 播放时等待较短时间，暂停时等待按键
            wait_time = self.play_speed if self.playing else 1
            key = cv2.waitKey(wait_time) & 0xFF
            
            # 自动播放：在按键处理之前更新帧
            if self.playing and key == 255:  # 255表示没有按键
                self.current_frame += 1
                if self.current_frame >= self.total_frames:
                    if self.auto_next:
                        # 自动播放模式：播放完自动跳到下一个文件
                        result = 'next'
                        break
                    else:
                        self.current_frame = 0  # 循环播放
                continue  # 立即进入下一次循环显示新帧
            
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                print("退出可视化")
                result = 'quit'
                break
            elif key == ord('n') or key == ord('N'):  # N - 下一个文件
                print("跳到下一个文件")
                result = 'next'
                break
            elif key == ord('p') or key == ord('P'):  # P - 上一个文件
                print("跳到上一个文件")
                result = 'prev'
                break
            elif key == 32:  # Space
                self.playing = not self.playing
                status = "播放" if self.playing else "暂停"
                print(f"状态: {status}")
            elif key == ord('a') or key == ord('A'):  # A - 上一帧
                self.current_frame = max(0, self.current_frame - 1)
                self.playing = False
            elif key == ord('d') or key == ord('D'):  # D - 下一帧
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                self.playing = False
            elif key == ord('w') or key == ord('W'):  # W - 加速
                self.play_speed = max(10, self.play_speed - 10)
                print(f"播放速度: {1000/self.play_speed:.1f} FPS")
            elif key == ord('s') or key == ord('S'):  # S - 减速
                self.play_speed = min(200, self.play_speed + 10)
                print(f"播放速度: {1000/self.play_speed:.1f} FPS")
            elif key == ord('h') or key == ord('H'):  # H - 第一帧
                self.current_frame = 0
                self.playing = False
                print("跳到第一帧")
            elif key == ord('e') or key == ord('E'):  # E - 最后一帧
                self.current_frame = self.total_frames - 1
                self.playing = False
                print("跳到最后一帧")
        
        cv2.destroyAllWindows()
        return result


class H5DataBrowser:
    """H5数据浏览器（浏览目录中的所有H5文件）"""
    
    def __init__(self, data_dir, auto_play=False, auto_start=False):
        """
        初始化浏览器
        
        参数:
            data_dir (str): 数据目录
            auto_play (bool): 是否自动连续播放所有文件
            auto_start (bool): 是否自动开始播放（不需要按空格）
        """
        self.data_dir = data_dir
        self.h5_files = []
        self.current_file_idx = 0
        self.auto_play = auto_play
        self.auto_start = auto_start
        
    def scan_directory(self):
        """扫描目录中的H5文件"""
        print(f"\n正在扫描目录: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 目录不存在: {self.data_dir}")
            return False
        
        self.h5_files = sorted([
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.h5')
        ])
        
        if not self.h5_files:
            print(f"❌ 目录中没有找到H5文件")
            return False
        
        print(f"✅ 找到 {len(self.h5_files)} 个H5文件")
        return True
    
    def browse(self):
        """浏览所有H5文件"""
        if not self.h5_files:
            print("❌ 没有可浏览的文件")
            return
        
        print("\n📂 H5数据浏览器")
        print("="*70)
        
        if self.auto_play:
            print("🔄 自动连续播放模式 - 按N跳到下一个文件，按Q退出")
        
        while self.current_file_idx < len(self.h5_files):
            current_file = self.h5_files[self.current_file_idx]
            
            print(f"\n当前文件 ({self.current_file_idx + 1}/{len(self.h5_files)}):")
            print(f"  {os.path.basename(current_file)}")
            
            # 可视化当前文件
            visualizer = H5DataVisualizer(current_file, auto_start=self.auto_start)
            visualizer.auto_next = self.auto_play  # 传递自动播放标志
            if visualizer.load_data():
                result = visualizer.visualize()
                
                # 检查返回值决定下一步操作
                if result == 'quit':
                    print("退出浏览")
                    break
                elif result == 'next':
                    self.current_file_idx += 1
                    continue
                elif result == 'prev':
                    self.current_file_idx = max(0, self.current_file_idx - 1)
                    continue
            
            # 自动播放模式下自动进入下一个文件
            if self.auto_play:
                self.current_file_idx += 1
                continue
            
            # 手动模式：询问是否继续
            print("\n" + "="*70)
            choice = input("继续浏览下一个文件？(y/n/p=上一个): ").strip().lower()
            
            if choice in ['n', 'no', 'q', 'quit']:
                print("退出浏览")
                break
            elif choice in ['p', 'prev', 'previous']:
                self.current_file_idx = max(0, self.current_file_idx - 1)
            else:
                self.current_file_idx += 1
        
        print("\n✅ 浏览完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='H5数据可视化工具')
    parser.add_argument('--file', type=str, help='H5文件路径')
    parser.add_argument('--dir', type=str, help='数据目录路径（浏览模式）')
    parser.add_argument('--browse', action='store_true', 
                       help='浏览模式：逐个查看目录中的所有H5文件')
    parser.add_argument('--auto', action='store_true',
                       help='自动连续播放模式：播放完一个文件自动播放下一个')
    
    args = parser.parse_args()
    
    if args.browse or args.dir:
        # 浏览模式
        data_dir = args.dir if args.dir else './auto_collected_data'
        browser = H5DataBrowser(data_dir, auto_play=args.auto)
        if browser.scan_directory():
            browser.browse()
    elif args.file:
        # 单文件模式
        visualizer = H5DataVisualizer(args.file)
        if visualizer.load_data():
            visualizer.visualize()
    else:
        # 交互式选择
        print("\n" + "="*70)
        print("H5数据可视化工具")
        print("="*70)
        print("\n请选择模式:")
        print("  [1] 查看单个H5文件")
        print("  [2] 浏览目录中的所有H5文件（手动切换）")
        print("  [3] 自动连续播放目录中的所有H5文件（需按空格开始）")
        print("  [4] 自动连续播放目录中的所有H5文件（直接开始播放）")
        print("  [Q] 退出")
        
        choice = input("\n请输入选项 [1-4/Q]: ").strip()
        
        if choice == '1':
            file_path = input("请输入H5文件路径: ").strip()
            visualizer = H5DataVisualizer(file_path)
            if visualizer.load_data():
                visualizer.visualize()
        elif choice == '2':
            data_dir = input("请输入数据目录路径（默认: ./auto_collected_data）: ").strip()
            if not data_dir:
                data_dir = './auto_collected_data'
            browser = H5DataBrowser(data_dir, auto_play=False)
            if browser.scan_directory():
                browser.browse()
        elif choice == '3':
            data_dir = input("请输入数据目录路径（默认: ./auto_collected_data）: ").strip()
            if not data_dir:
                data_dir = './auto_collected_data'
            browser = H5DataBrowser(data_dir, auto_play=True, auto_start=False)
            if browser.scan_directory():
                browser.browse()
        elif choice == '4':
            data_dir = input("请输入数据目录路径（默认: ./auto_collected_data）: ").strip()
            if not data_dir:
                data_dir = './auto_collected_data'
            browser = H5DataBrowser(data_dir, auto_play=True, auto_start=True)
            if browser.scan_directory():
                browser.browse()
        else:
            print("退出")


if __name__ == '__main__':
    main()
