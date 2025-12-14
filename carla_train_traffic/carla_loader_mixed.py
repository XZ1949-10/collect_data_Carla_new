#!/usr/bin/env python
# coding=utf-8
'''
混合数据加载器
支持同时加载旧数据和新数据，按比例混合训练
用于防止灾难性遗忘
'''

import glob
import os
import random

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from imgaug import augmenters as iaa
from helper import RandomTransWrapper


class MixedDataLoader():
    """
    混合数据加载器
    
    支持两种混合模式:
    1. concat: 简单拼接，按数据量自然比例
    2. balanced: 平衡采样，可指定新旧数据比例
    """
    def __init__(self,
                 old_train_folder,      # 旧数据训练集
                 old_eval_folder,       # 旧数据验证集
                 new_train_folder,      # 新数据训练集
                 new_eval_folder,       # 新数据验证集
                 batch_size=4,
                 num_workers=4,
                 distributed=False,
                 world_size=1,
                 rank=0,
                 mix_ratio=0.5,         # 新数据占比 (0.5 = 各50%)
                 mix_mode='balanced',   # 'concat' 或 'balanced'
                 min_frames=10):
        
        self.mix_ratio = mix_ratio
        self.mix_mode = mix_mode
        
        # 加载旧数据
        if rank == 0:
            print("\n📦 加载旧数据集...")
        old_train = CarlaH5DatasetDynamic(
            data_dir=old_train_folder, train_eval_flag="train", min_frames=min_frames)
        old_eval = CarlaH5DatasetDynamic(
            data_dir=old_eval_folder, train_eval_flag="eval", min_frames=min_frames)
        
        # 加载新数据
        if rank == 0:
            print("\n🆕 加载新数据集...")
        new_train = CarlaH5DatasetDynamic(
            data_dir=new_train_folder, train_eval_flag="train", min_frames=min_frames)
        new_eval = CarlaH5DatasetDynamic(
            data_dir=new_eval_folder, train_eval_flag="eval", min_frames=min_frames)
        
        # 打印统计
        if rank == 0:
            print(f"\n📊 数据集统计:")
            print(f"  旧训练集: {len(old_train)} 帧")
            print(f"  新训练集: {len(new_train)} 帧")
            print(f"  混合模式: {mix_mode}, 新数据占比: {mix_ratio*100:.0f}%")
        
        # 创建混合数据集
        if mix_mode == 'concat':
            # 简单拼接
            train_dataset = ConcatDataset([old_train, new_train])
            eval_dataset = ConcatDataset([old_eval, new_eval])
            train_sampler = self._create_sampler(train_dataset, distributed, world_size, rank, shuffle=True)
            eval_sampler = self._create_sampler(eval_dataset, distributed, world_size, rank, shuffle=False)
        else:
            # 平衡采样
            train_dataset = ConcatDataset([old_train, new_train])
            eval_dataset = ConcatDataset([old_eval, new_eval])
            
            if distributed:
                # DDP模式下使用自定义采样器
                train_sampler = BalancedDistributedSampler(
                    train_dataset,
                    old_size=len(old_train),
                    new_size=len(new_train),
                    new_ratio=mix_ratio,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True)
                eval_sampler = DistributedSampler(
                    eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            else:
                # 单卡模式使用WeightedRandomSampler
                weights = self._compute_weights(len(old_train), len(new_train), mix_ratio)
                train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                eval_sampler = None
        
        self.samplers = {"train": train_sampler, "eval": eval_sampler}
        
        loader_kwargs = {
            'pin_memory': True,
            'prefetch_factor': 2 if num_workers > 0 else None,
            'persistent_workers': num_workers > 0,
        }
        
        self.loaders = {
            "train": DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=(train_sampler is None), num_workers=num_workers,
                sampler=train_sampler, drop_last=True, **loader_kwargs),
            "eval": DataLoader(
                eval_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, sampler=eval_sampler, **loader_kwargs)
        }
        
        # 保存数据集引用
        self.old_train = old_train
        self.new_train = new_train
    
    def _create_sampler(self, dataset, distributed, world_size, rank, shuffle):
        if distributed:
            return DistributedSampler(
                dataset, num_replicas=world_size, rank=rank,
                shuffle=shuffle, drop_last=True)
        return None
    
    def _compute_weights(self, old_size, new_size, new_ratio):
        """计算采样权重"""
        total = old_size + new_size
        # 新数据权重
        new_weight = new_ratio / new_size if new_size > 0 else 0
        # 旧数据权重
        old_weight = (1 - new_ratio) / old_size if old_size > 0 else 0
        
        weights = [old_weight] * old_size + [new_weight] * new_size
        return weights


class BalancedDistributedSampler(DistributedSampler):
    """
    平衡分布式采样器
    确保每个epoch中新旧数据按指定比例采样
    """
    def __init__(self, dataset, old_size, new_size, new_ratio=0.5,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=True)
        self.old_size = old_size
        self.new_size = new_size
        self.new_ratio = new_ratio
        
    def __iter__(self):
        # 计算每个epoch采样数量
        total_samples = len(self.dataset)
        new_samples = int(total_samples * self.new_ratio)
        old_samples = total_samples - new_samples
        
        # 生成索引
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 旧数据索引 [0, old_size)
        old_indices = list(range(self.old_size))
        # 新数据索引 [old_size, old_size + new_size)
        new_indices = list(range(self.old_size, self.old_size + self.new_size))
        
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(old_indices)
            random.shuffle(new_indices)
        
        # 按比例采样
        sampled_old = old_indices * (old_samples // self.old_size + 1)
        sampled_new = new_indices * (new_samples // self.new_size + 1)
        sampled_old = sampled_old[:old_samples]
        sampled_new = sampled_new[:new_samples]
        
        # 混合并打乱
        indices = sampled_old + sampled_new
        if self.shuffle:
            random.shuffle(indices)
        
        # 分配给当前rank
        indices = indices[self.rank:len(indices):self.num_replicas]
        
        return iter(indices)


class CarlaH5DatasetDynamic(Dataset):
    """动态帧数H5数据集"""
    def __init__(self, data_dir, train_eval_flag="train", use_cache=False, min_frames=10):
        self.data_dir = data_dir
        self.train_eval_flag = train_eval_flag
        self.use_cache = use_cache
        
        if not data_dir.endswith(('/', '\\')):
            data_dir = data_dir + '/'
        all_files = glob.glob(data_dir + '*.h5')
        all_files.sort()
        
        self.data_list = []
        self.file_frames = []
        self.cumulative_frames = [0]
        
        for file_path in all_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    num_frames = f['rgb'].shape[0]
                    if num_frames < min_frames:
                        continue
                    self.data_list.append(file_path)
                    self.file_frames.append(num_frames)
                    self.cumulative_frames.append(self.cumulative_frames[-1] + num_frames)
            except:
                continue
        
        self.total_frames = self.cumulative_frames[-1]
        
        if len(self.data_list) == 0:
            raise ValueError(f"❌ 目录 {data_dir} 中没有有效的h5文件!")
        
        print(f"  ✅ {len(self.data_list)} 个文件, {self.total_frames} 帧")
        
        self._cache = {} if use_cache else None
        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(seq=iaa.GaussianBlur((0, 1.5)), p=0.09),
                    RandomTransWrapper(seq=iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5), p=0.09),
                    RandomTransWrapper(seq=iaa.Dropout((0.0, 0.10), per_channel=0.5), p=0.3),
                    RandomTransWrapper(seq=iaa.CoarseDropout(
                        (0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5), p=0.3),
                    RandomTransWrapper(seq=iaa.Add((-20, 20), per_channel=0.5), p=0.3),
                    RandomTransWrapper(seq=iaa.Multiply((0.9, 1.1), per_channel=0.2), p=0.4),
                    RandomTransWrapper(seq=iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5), p=0.09),
                ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.total_frames

    def _global_to_local(self, global_idx):
        left, right = 0, len(self.data_list) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self.cumulative_frames[mid] <= global_idx:
                left = mid
            else:
                right = mid - 1
        return left, global_idx - self.cumulative_frames[left]

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            cached = self._cache[idx]
            img = self.transform(cached['img'].copy())
            return img, cached['speed'], cached['target'], cached['mask']
        
        file_idx, frame_idx = self._global_to_local(idx)
        file_name = self.data_list[file_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'][frame_idx])
            target = np.array(h5_file['targets'][frame_idx]).astype(np.float32)
        
        command = max(0, min(3, int(target[24]) - 2))
        target_vec = np.zeros((4, 3), dtype=np.float32)
        target_vec[command, :] = target[:3]
        speed = np.array([target[10] / 25, ]).astype(np.float32)
        mask_vec = np.zeros((4, 3), dtype=np.float32)
        mask_vec[command, :] = 1
        
        if self._cache is not None:
            self._cache[idx] = {
                'img': img, 'speed': speed,
                'target': target_vec.reshape(-1), 'mask': mask_vec.reshape(-1)
            }
        
        img = self.transform(img)
        return img, speed, target_vec.reshape(-1), mask_vec.reshape(-1)
