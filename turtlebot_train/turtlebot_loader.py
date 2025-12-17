#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 数据加载器 (固定帧数版本，单 GPU)

特点:
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

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from helper import RandomTransWrapper


# ============ 配置常量 ============
TARGETS_SPEED_IDX = 10
TARGETS_LINEAR_VEL_IDX = 20
TARGETS_ANGULAR_VEL_IDX = 21
TARGETS_COMMAND_IDX = 24

BRANCH_OUTPUT_DIM = 2  # [linear_vel, angular_vel]
NUM_BRANCHES = 4

SPEED_NORMALIZATION = 25.0
MAX_LINEAR_VEL = 0.7
MAX_ANGULAR_VEL = 1.0


class CarlaH5Data():
    """TurtleBot 数据加载器 (单 GPU 版本)"""
    def __init__(self,
                 train_folder,
                 eval_folder,
                 batch_size=4, num_workers=4, distributed=False):

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                TurtleBotH5Dataset(
                    data_dir=train_folder,
                    train_eval_flag="train"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            ),
            "eval": torch.utils.data.DataLoader(
                TurtleBotH5Dataset(
                    data_dir=eval_folder,
                    train_eval_flag="eval"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}


class TurtleBotH5Dataset(Dataset):
    """TurtleBot H5 数据集"""
    def __init__(self, data_dir, train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        if not data_dir.endswith(('/', '\\')):
            data_dir = data_dir + '/'
        self.data_list = glob.glob(data_dir + '*.h5')
        self.data_list.sort()
        self.sequence_len = sequence_len
        self.train_eval_flag = train_eval_flag

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
        return self.sequence_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequence_len
        file_idx = idx % self.sequence_len
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'][file_idx])
            target = np.array(h5_file['targets'][file_idx]).astype(np.float32)
        
        # 处理命令
        command = int(target[TARGETS_COMMAND_IDX]) - 2
        command = max(0, min(3, command))
        
        # 提取 linear_vel 和 angular_vel
        linear_vel = target[TARGETS_LINEAR_VEL_IDX]
        angular_vel = target[TARGETS_ANGULAR_VEL_IDX]
        
        # 归一化
        linear_vel_norm = np.clip(linear_vel / MAX_LINEAR_VEL, -1.0, 1.0)
        angular_vel_norm = np.clip(angular_vel / MAX_ANGULAR_VEL, -1.0, 1.0)
        
        # 构建目标向量 (4个分支 × 2维)
        target_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
        target_vec[command, 0] = linear_vel_norm
        target_vec[command, 1] = angular_vel_norm
        
        # 速度输入
        speed = np.array([target[TARGETS_SPEED_IDX] / SPEED_NORMALIZATION], dtype=np.float32)
        
        # 掩码向量
        mask_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
        mask_vec[command, :] = 1
        
        img = self.transform(img)
        return img, speed, target_vec.reshape(-1), mask_vec.reshape(-1)


# 兼容性别名
CarlaH5Dataset = TurtleBotH5Dataset
