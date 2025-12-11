#!/usr/bin/env python
# coding=utf-8
"""
H5数据可视化脚本

使用方法:
    python -m collect_data_new.scripts.visualize_data --file data.h5
    python -m collect_data_new.scripts.visualize_data --dir ./carla_data
    python -m collect_data_new.scripts.visualize_data --dir ./carla_data --auto
    python -m collect_data_new.scripts.visualize_data --dir ./carla_data --auto --auto-start
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collect_data_new.utils import H5DataVisualizer, DataLoader


class H5DataBrowser:
    """H5数据浏览器（浏览目录中的所有H5文件）"""
    
    def __init__(self, data_dir: str, auto_play: bool = False, auto_start: bool = False):
        """
        初始化浏览器
        
        参数:
            data_dir: 数据目录
            auto_play: 是否自动连续播放所有文件
            auto_start: 是否自动开始播放（不需要按空格）
        """
        self.data_dir = data_dir
        self.auto_play = auto_play
        self.auto_start = auto_start
        self.loader = DataLoader(data_dir)
        self.h5_files = []
        self.current_idx = 0
    
    def scan(self) -> bool:
        """扫描目录"""
        self.h5_files = self.loader.find_h5_files()
        
        if not self.h5_files:
            print(f"❌ 目录中没有H5文件")
            return False
        
        print(f"✅ 找到 {len(self.h5_files)} 个文件")
        return True
    
    def browse(self):
        """浏览所有文件"""
        if not self.h5_files:
            return
        
        print("\n📂 H5数据浏览器")
        print("="*70)
        
        if self.auto_play:
            print("🔄 自动连续播放模式 - 按N跳到下一个文件，按Q退出")
        
        while self.current_idx < len(self.h5_files):
            filepath = self.h5_files[self.current_idx]
            
            print(f"\n当前文件 ({self.current_idx + 1}/{len(self.h5_files)}):")
            print(f"  {os.path.basename(filepath)}")
            
            # 创建可视化器
            visualizer = H5DataVisualizer(filepath, auto_start=self.auto_start)
            visualizer.auto_next = self.auto_play  # 传递自动播放标志
            
            if visualizer.load_data():
                result = visualizer.visualize()
                
                if result == 'quit':
                    print("退出浏览")
                    break
                elif result == 'next':
                    self.current_idx += 1
                    continue
                elif result == 'prev':
                    self.current_idx = max(0, self.current_idx - 1)
                    continue
            
            # 自动播放模式下自动进入下一个文件
            if self.auto_play:
                self.current_idx += 1
                continue
            
            # 手动模式：询问是否继续
            print("\n" + "="*70)
            choice = input("继续浏览下一个文件？(y/n/p=上一个): ").strip().lower()
            
            if choice in ['n', 'no', 'q', 'quit']:
                print("退出浏览")
                break
            elif choice in ['p', 'prev', 'previous']:
                self.current_idx = max(0, self.current_idx - 1)
            else:
                self.current_idx += 1
        
        print("\n✅ 浏览完成")


def main():
    parser = argparse.ArgumentParser(description='H5数据可视化工具')
    parser.add_argument('--file', type=str, help='单个H5文件路径')
    parser.add_argument('--dir', type=str, help='数据目录路径')
    parser.add_argument('--auto', action='store_true', 
                        help='自动连续播放模式：播放完一个文件自动播放下一个')
    parser.add_argument('--auto-start', action='store_true',
                        help='自动开始播放（不需要按空格）')
    
    args = parser.parse_args()
    
    if args.file:
        # 单文件模式
        visualizer = H5DataVisualizer(args.file, auto_start=args.auto_start)
        if visualizer.load_data():
            visualizer.visualize()
    elif args.dir:
        # 浏览模式
        browser = H5DataBrowser(args.dir, auto_play=args.auto, auto_start=args.auto_start)
        if browser.scan():
            browser.browse()
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
            data_dir = input("请输入数据目录路径（默认: ./carla_data）: ").strip()
            if not data_dir:
                data_dir = './carla_data'
            browser = H5DataBrowser(data_dir, auto_play=False)
            if browser.scan():
                browser.browse()
        elif choice == '3':
            data_dir = input("请输入数据目录路径（默认: ./carla_data）: ").strip()
            if not data_dir:
                data_dir = './carla_data'
            browser = H5DataBrowser(data_dir, auto_play=True, auto_start=False)
            if browser.scan():
                browser.browse()
        elif choice == '4':
            data_dir = input("请输入数据目录路径（默认: ./carla_data）: ").strip()
            if not data_dir:
                data_dir = './carla_data'
            browser = H5DataBrowser(data_dir, auto_play=True, auto_start=True)
            if browser.scan():
                browser.browse()
        else:
            print("退出")


if __name__ == '__main__':
    main()
