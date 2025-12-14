#!/usr/bin/env python
# coding=utf-8
'''
动态帧数版本的数据加载器
支持不同帧数的h5文件混合训练
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
        
        if rank == 0:
            print(f"📊 训练集: {len(train_dataset.data_list)} 个文件, {len(train_dataset)} 帧")
            print(f"📊 验证集: {len(eval_dataset.data_list)} 个文件, {len(eval_dataset)} 帧")
        
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True)
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=False)
        else:
            train_sampler = None
            eval_sampler = None
        
        self.samplers = {"train": train_sampler, "eval": eval_sampler}
        
        loader_kwargs = {
            'pin_memory': True,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'persistent_workers': num_workers > 0,
            'multiprocessing_context': 'spawn' if num_workers > 0 else None,
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


class CarlaH5DatasetDynamic(Dataset):
    """动态帧数H5数据集"""
    def __init__(self, data_dir, train_eval_flag="train", use_cache=False, min_frames=10):
        self.data_dir = data_dir
        self.train_eval_flag = train_eval_flag
        self.use_cache = use_cache
        self.min_frames = min_frames
        
        if not data_dir.endswith(('/', '\\')):
            data_dir = data_dir + '/'
        all_files = glob.glob(data_dir + '*.h5')
        all_files.sort()
        
        self.data_list = []
        self.file_frames = []
        self.cumulative_frames = [0]
        
        print(f"🔍 扫描 {len(all_files)} 个h5文件...")
        skipped = 0
        for file_path in all_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    num_frames = f['rgb'].shape[0]
                    if num_frames < min_frames:
                        skipped += 1
                        continue
                    self.data_list.append(file_path)
                    self.file_frames.append(num_frames)
                    self.cumulative_frames.append(self.cumulative_frames[-1] + num_frames)
            except Exception as e:
                print(f"⚠️ 跳过损坏文件: {os.path.basename(file_path)} - {e}")
                skipped += 1
        
        self.total_frames = self.cumulative_frames[-1]
        
        if skipped > 0:
            print(f"⚠️ 跳过 {skipped} 个文件")
        
        if len(self.data_list) == 0:
            raise ValueError(f"❌ 目录 {data_dir} 中没有有效的h5文件!")
        
        if self.file_frames:
            print(f"✅ 加载 {len(self.data_list)} 个文件, 共 {self.total_frames} 帧")
            print(f"   帧数范围: {min(self.file_frames)} ~ {max(self.file_frames)}")
        
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
