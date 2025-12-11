#!/usr/bin/env python
# coding=utf-8
"""
数据处理工具

提供数据保存、加载、验证等功能。
"""

import os
import time
import json
import numpy as np
import h5py
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..config import COMMAND_NAMES


class DataSaver:
    """数据保存器"""
    
    def __init__(self, save_path: str, segment_size: int = 200):
        """
        初始化数据保存器
        
        参数:
            save_path: 保存路径
            segment_size: 每段数据帧数
        """
        self.save_path = save_path
        self.segment_size = segment_size
        
        self.total_saved_segments = 0
        self.total_saved_frames = 0
        
        os.makedirs(save_path, exist_ok=True)
    
    def save_segment(self, rgb_list: List[np.ndarray], targets_list: List[np.ndarray],
                     command: float, suffix: str = '') -> Optional[str]:
        """
        保存数据段到H5文件
        
        参数:
            rgb_list: RGB图像列表
            targets_list: 目标数据列表
            command: 导航命令
            suffix: 文件名后缀
            
        返回:
            str: 保存的文件路径，失败返回None
        """
        if len(rgb_list) == 0:
            return None
        
        rgb_array = np.array(rgb_list, dtype=np.uint8)
        targets_array = np.array(targets_list, dtype=np.float32)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        command_name = COMMAND_NAMES.get(int(command), 'Unknown')
        filename = f"carla_cmd{int(command)}_{command_name}_{timestamp}{suffix}.h5"
        filepath = os.path.join(self.save_path, filename)
        
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('rgb', data=rgb_array, compression='gzip', compression_opts=4)
            hf.create_dataset('targets', data=targets_array, compression='gzip', compression_opts=4)
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  ✓ {filename} ({len(rgb_array)} 样本, {file_size_mb:.2f} MB)")
        
        self.total_saved_segments += 1
        self.total_saved_frames += len(rgb_array)
        
        return filepath
    
    def _save_segment_array(self, rgb_array: np.ndarray, targets_array: np.ndarray,
                            command: float, suffix: str = '') -> Optional[str]:
        """
        直接保存 numpy 数组到 H5 文件（内部方法，避免列表转换开销）
        
        参数:
            rgb_array: RGB图像数组 (N, H, W, C)
            targets_array: 目标数据数组 (N, 25)
            command: 导航命令
            suffix: 文件名后缀
            
        返回:
            str: 保存的文件路径，失败返回None
        """
        if rgb_array.shape[0] == 0:
            return None
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        command_name = COMMAND_NAMES.get(int(command), 'Unknown')
        filename = f"carla_cmd{int(command)}_{command_name}_{timestamp}{suffix}.h5"
        filepath = os.path.join(self.save_path, filename)
        
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('rgb', data=rgb_array, compression='gzip', compression_opts=4)
            hf.create_dataset('targets', data=targets_array, compression='gzip', compression_opts=4)
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  ✓ {filename} ({rgb_array.shape[0]} 样本, {file_size_mb:.2f} MB)")
        
        self.total_saved_segments += 1
        self.total_saved_frames += rgb_array.shape[0]
        
        return filepath
    
    def save_segment_chunked(self, rgb_list: List[np.ndarray], targets_list: List[np.ndarray],
                             command: float) -> List[str]:
        """
        按chunk大小分割保存数据段
        
        参数:
            rgb_list: RGB图像列表
            targets_list: 目标数据列表
            command: 导航命令
            
        返回:
            List[str]: 保存的文件路径列表
        """
        if len(rgb_list) == 0:
            return []
        
        rgb_array = np.array(rgb_list, dtype=np.uint8)
        targets_array = np.array(targets_list, dtype=np.float32)
        
        total_samples = rgb_array.shape[0]
        num_chunks = (total_samples + self.segment_size - 1) // self.segment_size
        
        saved_files = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.segment_size
            end_idx = min((chunk_idx + 1) * self.segment_size, total_samples)
            
            chunk_rgb = rgb_array[start_idx:end_idx]
            chunk_targets = targets_array[start_idx:end_idx]
            
            # 直接使用 numpy 数组切片，避免 tolist() 的性能开销
            filepath = self._save_segment_array(
                chunk_rgb, chunk_targets,
                command, f"_part{chunk_idx+1:03d}"
            )
            if filepath:
                saved_files.append(filepath)
        
        return saved_files
    
    def get_statistics(self) -> Dict[str, int]:
        """获取保存统计"""
        return {
            'total_segments': self.total_saved_segments,
            'total_frames': self.total_saved_frames
        }


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        参数:
            data_path: 数据目录路径
        """
        self.data_path = data_path
    
    def find_h5_files(self) -> List[str]:
        """递归查找所有H5文件"""
        h5_files = []
        for root, dirs, files in os.walk(self.data_path):
            for f in files:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(root, f))
        return sorted(h5_files)
    
    def load_file(self, filepath: str) -> Optional[Dict[str, np.ndarray]]:
        """
        加载单个H5文件
        
        参数:
            filepath: 文件路径
            
        返回:
            Dict: {'rgb': ndarray, 'targets': ndarray} 或 None
        """
        try:
            with h5py.File(filepath, 'r') as f:
                if 'rgb' not in f or 'targets' not in f:
                    return None
                return {
                    'rgb': f['rgb'][:],
                    'targets': f['targets'][:]
                }
        except Exception as e:
            print(f"❌ 加载文件失败 {filepath}: {e}")
            return None
    
    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        分析单个H5文件
        
        返回:
            Dict: 文件分析结果
        """
        result = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'valid': False,
            'total_frames': 0,
            'command_distribution': {},
            'speed_stats': {},
            'control_stats': {}
        }
        
        data = self.load_file(filepath)
        if data is None:
            return result
        
        rgb = data['rgb']
        targets = data['targets']
        
        result['valid'] = True
        result['total_frames'] = rgb.shape[0]
        
        # 命令分布
        commands = targets[:, 24]
        for cmd in np.unique(commands):
            cmd_name = COMMAND_NAMES.get(int(cmd), f'Unknown({int(cmd)})')
            result['command_distribution'][cmd_name] = int(np.sum(commands == cmd))
        
        # 速度统计
        speeds = targets[:, 10]
        result['speed_stats'] = {
            'mean': float(np.mean(speeds)),
            'min': float(np.min(speeds)),
            'max': float(np.max(speeds))
        }
        
        # 控制信号统计
        result['control_stats'] = {
            'steer': {'min': float(np.min(targets[:, 0])), 'max': float(np.max(targets[:, 0]))},
            'throttle': {'min': float(np.min(targets[:, 1])), 'max': float(np.max(targets[:, 1]))},
            'brake': {'min': float(np.min(targets[:, 2])), 'max': float(np.max(targets[:, 2]))}
        }
        
        return result
    
    def analyze_all(self) -> Dict[str, Any]:
        """分析所有文件"""
        h5_files = self.find_h5_files()
        
        total_frames = 0
        command_stats = defaultdict(int)
        speed_all = []
        valid_files = 0
        
        for filepath in h5_files:
            analysis = self.analyze_file(filepath)
            if analysis['valid']:
                valid_files += 1
                total_frames += analysis['total_frames']
                
                for cmd, count in analysis['command_distribution'].items():
                    command_stats[cmd] += count
                
                data = self.load_file(filepath)
                if data:
                    speed_all.extend(data['targets'][:, 10].tolist())
        
        return {
            'total_files': len(h5_files),
            'valid_files': valid_files,
            'total_frames': total_frames,
            'command_distribution': dict(command_stats),
            'speed_stats': {
                'mean': float(np.mean(speed_all)) if speed_all else 0,
                'min': float(np.min(speed_all)) if speed_all else 0,
                'max': float(np.max(speed_all)) if speed_all else 0
            }
        }


def build_targets(steer: float, throttle: float, brake: float,
                  speed_kmh: float, command: float) -> np.ndarray:
    """
    构建targets数组
    
    参数:
        steer: 方向盘值 (-1 to 1)
        throttle: 油门值 (0 to 1)
        brake: 刹车值 (0 to 1)
        speed_kmh: 速度 (km/h)
        command: 导航命令
        
    返回:
        np.ndarray: 25维targets数组
    """
    targets = np.zeros(25, dtype=np.float32)
    targets[0] = steer
    targets[1] = throttle
    targets[2] = brake
    targets[10] = speed_kmh
    targets[24] = command
    return targets
