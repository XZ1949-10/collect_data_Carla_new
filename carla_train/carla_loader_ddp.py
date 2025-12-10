#!/usr/bin/env python
# coding=utf-8
'''
DDP版本的数据加载器
支持分布式采样和预取优化
针对P100多卡训练优化
'''

import glob
import os

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from imgaug import augmenters as iaa
from helper import RandomTransWrapper


class CarlaH5DataDDP():
    """支持DDP的数据加载器，带进阶优化"""
    def __init__(self,
                 train_folder,
                 eval_folder,
                 batch_size=4,
                 num_workers=4,
                 distributed=False,
                 world_size=1,
                 rank=0,
                 prefetch_factor=2,
                 use_cache=False):
        
        train_dataset = CarlaH5Dataset(
            data_dir=train_folder,
            train_eval_flag="train",
            use_cache=use_cache)
        
        eval_dataset = CarlaH5Dataset(
            data_dir=eval_folder,
            train_eval_flag="eval",
            use_cache=use_cache)
        
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


class CarlaH5Dataset(Dataset):
    """
    H5数据集，带优化:
    - 线程本地H5文件句柄 (避免多进程问题)
    - 可选内存缓存
    - 优化的数据读取
    """
    def __init__(self, data_dir, train_eval_flag="train", sequence_len=200, use_cache=False):
        self.data_dir = data_dir
        if not data_dir.endswith(('/', '\\')):
            data_dir = data_dir + '/'
        self.data_list = glob.glob(data_dir + '*.h5')
        self.data_list.sort()
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag
        self.use_cache = use_cache
        
        # 内存缓存 (可选，适合小数据集)
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
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        # 检查缓存
        if self._cache is not None and idx in self._cache:
            cached = self._cache[idx]
            img = self.transform(cached['img'].copy())
            return img, cached['speed'], cached['target'], cached['mask']
        
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]

        # 每次读取后关闭文件，避免文件描述符耗尽
        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'][file_idx])
            target = np.array(h5_file['targets'][file_idx]).astype(np.float32)
        
        command = int(target[24]) - 2
        target_vec = np.zeros((4, 3), dtype=np.float32)
        target_vec[command, :] = target[:3]
        speed = np.array([target[10] / 25, ]).astype(np.float32)
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
