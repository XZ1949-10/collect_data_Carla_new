#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot DDP 数据加载器 (固定帧数版本)

特点:
- 支持分布式采样和预取优化
- 加载 linear_vel 和 angular_vel (而不是 steer, throttle, brake)
- 与 turtlebot_collect 数据格式兼容

数据格式:
    H5 文件结构:
    {
        'rgb': (N, 88, 200, 3),      # 图像数据
        'targets': (N, 25),          # 控制信号
    }
    
    targets 向量:
        targets[10] = speed (km/h)
        targets[20] = linear_vel (m/s)
        targets[21] = angular_vel (rad/s)
        targets[24] = command (2/3/4/5)
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


# ============ 配置常量 ============
# targets 向量索引 (与 turtlebot_collect/config/command_config.py 一致)
TARGETS_SPEED_IDX = 10
TARGETS_LINEAR_VEL_IDX = 20
TARGETS_ANGULAR_VEL_IDX = 21
TARGETS_COMMAND_IDX = 24

# 每个分支的输出维度
BRANCH_OUTPUT_DIM = 2  # [linear_vel, angular_vel]

# 分支数量
NUM_BRANCHES = 4

# 归一化参数
SPEED_NORMALIZATION = 25.0  # 速度归一化因子 (km/h)
MAX_LINEAR_VEL = 0.7        # 最大线速度 (m/s)，用于归一化
MAX_ANGULAR_VEL = 1.0       # 最大角速度 (rad/s)，用于归一化


class CarlaH5DataDDP():
    """
    支持 DDP 的 TurtleBot 数据加载器 (固定帧数版本)
    
    注意: 类名保持 CarlaH5DataDDP 以兼容 main_ddp.py
    """
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
        
        train_dataset = TurtleBotH5Dataset(
            data_dir=train_folder,
            train_eval_flag="train",
            use_cache=use_cache)
        
        eval_dataset = TurtleBotH5Dataset(
            data_dir=eval_folder,
            train_eval_flag="eval",
            use_cache=use_cache)
        
        # 打印数据集信息
        if rank == 0:
            print(f"📊 训练集: {len(train_dataset.data_list)} 个文件, {len(train_dataset)} 帧")
            print(f"📊 验证集: {len(eval_dataset.data_list)} 个文件, {len(eval_dataset)} 帧")
            print(f"📊 输出格式: [linear_vel, angular_vel] × 4 分支 = 8 维")
        
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
        
        # 优化的 DataLoader 配置
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


class TurtleBotH5Dataset(Dataset):
    """
    TurtleBot H5 数据集 (固定帧数版本)
    
    特点:
    - 假设每个 h5 文件有固定帧数 (sequence_len)
    - 加载 linear_vel 和 angular_vel
    """
    def __init__(self, data_dir, train_eval_flag="train", sequence_len=200, use_cache=False):
        self.data_dir = data_dir
        if not data_dir.endswith(('/', '\\')):
            data_dir = data_dir + '/'
        self.data_list = glob.glob(data_dir + '*.h5')
        self.data_list.sort()
        self.sequence_len = sequence_len
        self.train_eval_flag = train_eval_flag
        self.use_cache = use_cache
        
        # 内存缓存 (可选)
        self._cache = {} if use_cache else None
        
        self.build_transform()

    def build_transform(self):
        """构建数据增强变换"""
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
        return self.sequence_len * len(self.data_list)

    def __getitem__(self, idx):
        # 检查缓存
        if self._cache is not None and idx in self._cache:
            cached = self._cache[idx]
            img = self.transform(cached['img'].copy())
            return img, cached['speed'], cached['target'], cached['mask']
        
        data_idx = idx // self.sequence_len
        file_idx = idx % self.sequence_len
        file_name = self.data_list[data_idx]

        # 读取数据
        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'][file_idx])
            target = np.array(h5_file['targets'][file_idx]).astype(np.float32)
        
        # 处理命令
        # 2 Follow lane, 3 Left, 4 Right, 5 Straight
        # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
        command = int(target[TARGETS_COMMAND_IDX]) - 2
        command = max(0, min(3, command))  # 确保在有效范围内
        
        # 提取 linear_vel 和 angular_vel
        linear_vel = target[TARGETS_LINEAR_VEL_IDX]
        angular_vel = target[TARGETS_ANGULAR_VEL_IDX]
        
        # 归一化速度控制信号
        linear_vel_norm = np.clip(linear_vel / MAX_LINEAR_VEL, -1.0, 1.0)
        angular_vel_norm = np.clip(angular_vel / MAX_ANGULAR_VEL, -1.0, 1.0)
        
        # 构建目标向量 (4个分支 × 2维)
        target_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
        target_vec[command, 0] = linear_vel_norm   # 线速度
        target_vec[command, 1] = angular_vel_norm  # 角速度
        
        # 速度输入 (归一化)
        speed = np.array([target[TARGETS_SPEED_IDX] / SPEED_NORMALIZATION], dtype=np.float32)
        
        # 掩码向量
        mask_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
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


# ============ 兼容性别名 ============
CarlaH5Dataset = TurtleBotH5Dataset
