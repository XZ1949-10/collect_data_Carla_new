#!/usr/bin/env python
# coding=utf-8
'''
数据存储模块
负责将收集的数据保存为 H5 格式
'''

import os
import h5py
import numpy as np
import rospy
from datetime import datetime

from ..config import StorageConfig


class DataSaver:
    """数据保存器"""
    
    def __init__(self, output_dir=None, prefix=None):
        """
        初始化数据保存器
        
        参数:
            output_dir (str): 输出目录，None 使用配置默认值
            prefix (str): 文件名前缀，None 使用配置默认值
        """
        self.output_dir = output_dir or StorageConfig.DEFAULT_OUTPUT_DIR
        self.prefix = prefix or StorageConfig.FILE_PREFIX
        self.episode_count = 0
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 扫描已有文件，确定起始 episode 编号
        self._scan_existing_files()
        
    def _scan_existing_files(self):
        """扫描已有文件，确定起始编号"""
        if not os.path.exists(self.output_dir):
            return
        
        max_episode = -1
        for f in os.listdir(self.output_dir):
            if f.startswith(self.prefix) and f.endswith(StorageConfig.FILE_EXTENSION):
                try:
                    # 尝试从文件名提取编号
                    parts = f.replace(self.prefix + '_', '').split('_')
                    episode_num = int(parts[0])
                    max_episode = max(max_episode, episode_num)
                except (ValueError, IndexError):
                    pass
        
        self.episode_count = max_episode + 1
    
    def set_output_dir(self, output_dir):
        """
        设置输出目录
        
        参数:
            output_dir (str): 新的输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._scan_existing_files()
        
    def save(self, rgb_data, targets_data, metadata=None):
        """
        保存数据到 H5 文件
        
        参数:
            rgb_data (list or np.ndarray): RGB 图像数据
            targets_data (list or np.ndarray): targets 数据
            metadata (dict): 额外的元数据
            
        返回:
            str: 保存的文件路径，如果失败返回 None
        """
        if len(rgb_data) == 0:
            return None
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{self.episode_count:04d}_{timestamp}{StorageConfig.FILE_EXTENSION}"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with h5py.File(filepath, 'w') as hf:
                # 保存图像数据
                hf.create_dataset(
                    StorageConfig.DATASET_RGB,
                    data=np.array(rgb_data, dtype=np.uint8),
                    compression=StorageConfig.COMPRESSION,
                    compression_opts=StorageConfig.COMPRESSION_LEVEL
                )
                
                # 保存 targets 数据
                hf.create_dataset(
                    StorageConfig.DATASET_TARGETS,
                    data=np.array(targets_data, dtype=np.float32),
                    compression=StorageConfig.COMPRESSION,
                    compression_opts=StorageConfig.COMPRESSION_LEVEL
                )
                
                # 保存元数据
                hf.attrs['platform'] = StorageConfig.PLATFORM_NAME
                hf.attrs['frames'] = len(rgb_data)
                hf.attrs['timestamp'] = datetime.now().isoformat()
                hf.attrs['episode'] = self.episode_count
                
                if metadata:
                    for key, value in metadata.items():
                        hf.attrs[key] = value
            
            self.episode_count += 1
            return filepath
            
        except Exception as e:
            rospy.logerr(f"保存数据失败: {e}")
            return None
    
    def get_episode_count(self):
        """获取当前 episode 编号"""
        return self.episode_count


class DataBuffer:
    """
    数据缓冲区
    支持自动分割保存
    """
    
    def __init__(self, frames_per_file=None, auto_save_callback=None):
        """
        初始化缓冲区
        
        参数:
            frames_per_file (int): 每个文件的最大帧数，None 使用配置默认值
            auto_save_callback (callable): 达到帧数上限时的回调函数
                回调签名: callback(rgb_data, targets_data) -> filepath
        """
        self.frames_per_file = frames_per_file or StorageConfig.FRAMES_PER_FILE
        self.auto_save_callback = auto_save_callback
        
        self.rgb_buffer = []
        self.targets_buffer = []
        self._total_frames = 0  # 本次录制的总帧数
        self._saved_files = []  # 本次录制保存的文件列表
        
    def add(self, image, targets):
        """
        添加一帧数据
        
        参数:
            image (np.ndarray): RGB 图像
            targets (np.ndarray): targets 向量
            
        返回:
            str or None: 如果触发自动保存，返回保存的文件路径
        """
        self.rgb_buffer.append(image)
        self.targets_buffer.append(targets)
        self._total_frames += 1
        
        # 检查是否需要自动保存
        if len(self.rgb_buffer) >= self.frames_per_file:
            return self._auto_save()
        
        return None
    
    def _auto_save(self):
        """自动保存当前缓冲区"""
        if self.auto_save_callback is None:
            return None
        
        if len(self.rgb_buffer) == 0:
            return None
        
        # 调用保存回调
        try:
            filepath = self.auto_save_callback(self.rgb_buffer, self.targets_buffer)
        except Exception as e:
            rospy.logerr(f"自动保存失败: {e}")
            # 保存失败时不清空缓冲区，下次继续尝试
            return None
        
        if filepath:
            self._saved_files.append(filepath)
        
        # 清空缓冲区 (但不重置总帧数)
        self.rgb_buffer.clear()
        self.targets_buffer.clear()
        
        return filepath
        
    def clear(self):
        """清空缓冲区并重置统计"""
        self.rgb_buffer.clear()
        self.targets_buffer.clear()
        self._total_frames = 0
        self._saved_files.clear()
        
    def get_data(self):
        """获取缓冲区数据"""
        return self.rgb_buffer, self.targets_buffer
    
    def get_total_frames(self):
        """获取本次录制的总帧数"""
        return self._total_frames
    
    def get_saved_files(self):
        """获取本次录制保存的文件列表"""
        return self._saved_files.copy()
    
    def has_unsaved_data(self):
        """检查是否有未保存的数据"""
        return len(self.rgb_buffer) > 0
    
    def __len__(self):
        """获取当前缓冲区大小"""
        return len(self.rgb_buffer)
