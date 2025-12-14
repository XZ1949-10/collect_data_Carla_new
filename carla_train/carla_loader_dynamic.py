#!/usr/bin/env python
# coding=utf-8
'''
动态帧数版本的数据加载器
支持不同帧数的h5文件混合训练
针对P100多卡训练优化
'''

import glob
import os

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from imgaug import augmenters as iaa
from helper import RandomTransWrapper


class CarlaH5DataDDP():
    """支持DDP的数据加载器，动态帧数版本"""
    def __init__(self,
                 train_folder,
                 eval_folder,
                 batch_size=4,
                 num_workers=4,
                 distributed=False,
                 world_size=1,
                 rank=0,
                 prefetch_factor=2,
                 use_cache=False,
                 min_frames=10):
        
        train_dataset = CarlaH5DatasetDynamic(
            data_dir=train_folder,
            train_eval_flag="train",
            use_cache=use_cache,
            min_frames=min_frames)
        
        eval_dataset = CarlaH5DatasetDynamic(
            data_dir=eval_folder,
            train_eval_flag="eval",
            use_cache=use_cache,
            min_frames=min_frames)
        
        # 打印数据集信息
        if rank == 0:
            print(f"📊 训练集: {len(train_dataset.data_list)} 个文件, {len(train_dataset)} 帧")
            print(f"📊 验证集: {len(eval_dataset.data_list)} 个文件, {len(eval_dataset)} 帧")
        
        # 分布式采样器
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True)
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False)
        else:
            train_sampler = None
            eval_sampler = None
        
        self.samplers = {
            "train": train_sampler,
            "eval": eval_sampler
        }
        
        # 优化的DataLoader配置
        loader_kwargs = {
            'pin_memory': True,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'persistent_workers': num_workers > 0,
            'multiprocessing_context': 'spawn' if num_workers > 0 else None,
        }
        
        self.loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=num_workers,
                sampler=train_sampler,
                drop_last=True,
                **loader_kwargs
            ),
            "eval": DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=eval_sampler,
                **loader_kwargs
            )
        }


class CarlaH5DatasetDynamic(Dataset):
    """
    动态帧数H5数据集
    
    特点:
    - 自动检测每个h5文件的实际帧数
    - 支持不同帧数的文件混合
    - 构建全局索引映射表
    """
    def __init__(self, data_dir, train_eval_flag="train", use_cache=False, min_frames=10):
        self.data_dir = data_dir
        self.train_eval_flag = train_eval_flag
        self.use_cache = use_cache
        self.min_frames = min_frames
        
        # 查找所有h5文件
        if not data_dir.endswith(('/', '\\')):
            data_dir = data_dir + '/'
        all_files = glob.glob(data_dir + '*.h5')
        all_files.sort()
        
        # 扫描每个文件的帧数，构建索引映射
        self.data_list = []           # 有效的文件列表
        self.file_frames = []         # 每个文件的帧数
        self.cumulative_frames = [0]  # 累积帧数，用于快速定位
        
        print(f"🔍 扫描 {len(all_files)} 个h5文件...")
        skipped = 0
        for file_path in all_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    num_frames = f['rgb'].shape[0]
                    
                    # 跳过帧数太少的文件
                    if num_frames < min_frames:
                        skipped += 1
                        continue
                    
                    self.data_list.append(file_path)
                    self.file_frames.append(num_frames)
                    self.cumulative_frames.append(
                        self.cumulative_frames[-1] + num_frames
                    )
            except Exception as e:
                print(f"⚠️ 跳过损坏文件: {os.path.basename(file_path)} - {e}")
                skipped += 1
        
        self.total_frames = self.cumulative_frames[-1]
        
        if skipped > 0:
            print(f"⚠️ 跳过 {skipped} 个文件 (损坏或帧数<{min_frames})")
        
        if len(self.data_list) == 0:
            raise ValueError(f"❌ 目录 {data_dir} 中没有有效的h5文件!")
        
        # 打印帧数统计
        if self.file_frames:
            print(f"✅ 加载 {len(self.data_list)} 个文件, 共 {self.total_frames} 帧")
            print(f"   帧数范围: {min(self.file_frames)} ~ {max(self.file_frames)}, "
                  f"平均: {np.mean(self.file_frames):.1f}")
        
        # 内存缓存 (可选)
        self._cache = {} if use_cache else None
        
        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur((0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05), per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout((0.0, 0.10), per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add((-20, 20), per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply((0.9, 1.1), per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),
                        p=0.09),
                ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.total_frames

    def _global_to_local(self, global_idx):
        """
        将全局索引转换为 (文件索引, 文件内帧索引)
        使用二分查找，O(log n) 复杂度
        """
        # 二分查找找到对应的文件
        left, right = 0, len(self.data_list) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self.cumulative_frames[mid] <= global_idx:
                left = mid
            else:
                right = mid - 1
        
        file_idx = left
        frame_idx = global_idx - self.cumulative_frames[file_idx]
        return file_idx, frame_idx

    def __getitem__(self, idx):
        # 检查缓存
        if self._cache is not None and idx in self._cache:
            cached = self._cache[idx]
            img = self.transform(cached['img'].copy())
            return img, cached['speed'], cached['target'], cached['mask']
        
        # 转换全局索引到局部索引
        file_idx, frame_idx = self._global_to_local(idx)
        file_name = self.data_list[file_idx]

        # 读取数据
        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'][frame_idx])
            target = np.array(h5_file['targets'][frame_idx]).astype(np.float32)
        
        # 处理命令和目标
        # 2 Follow lane, 3 Left, 4 Right, 5 Straight
        # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
        command = int(target[24]) - 2
        
        # 确保command在有效范围内
        command = max(0, min(3, command))
        
        target_vec = np.zeros((4, 3), dtype=np.float32)
        target_vec[command, :] = target[:3]  # Steer, Gas, Brake
        
        speed = np.array([target[10] / 25, ]).astype(np.float32)  # 归一化速度
        
        mask_vec = np.zeros((4, 3), dtype=np.float32)
        mask_vec[command, :] = 1
        
        # 缓存原始数据
        if self._cache is not None:
            self._cache[idx] = {
                'img': img,
                'speed': speed,
                'target': target_vec.reshape(-1),
                'mask': mask_vec.reshape(-1)
            }
        
        img = self.transform(img)
        return img, speed, target_vec.reshape(-1), mask_vec.reshape(-1)

    def get_file_info(self):
        """返回文件信息，用于调试"""
        return {
            'num_files': len(self.data_list),
            'total_frames': self.total_frames,
            'frames_per_file': self.file_frames,
            'min_frames': min(self.file_frames) if self.file_frames else 0,
            'max_frames': max(self.file_frames) if self.file_frames else 0,
            'avg_frames': np.mean(self.file_frames) if self.file_frames else 0,
        }
