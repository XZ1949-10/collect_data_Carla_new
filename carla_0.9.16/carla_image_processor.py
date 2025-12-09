#!/usr/bin/env python
# coding=utf-8
'''
图像预处理模块
负责将原始图像转换为模型输入格式

图像处理参数（与数据收集保持一致）：
- 原始分辨率: 800x600
- 裁剪区域: [90:485, :] （去除天空和车头）
- 目标分辨率: 88x200
- 插值方法: INTER_CUBIC（双三次插值）
'''

import numpy as np
import cv2
import torch
from carla_config import IMAGE_HEIGHT, IMAGE_WIDTH


class ImageProcessor:
    """图像预处理器"""
    
    def __init__(self, device, enable_crop=True, crop_top=90, crop_bottom=485):
        """
        初始化图像预处理器
        
        参数:
            device: torch.device 对象
            enable_crop (bool): 是否启用图像裁剪（默认True）
            crop_top (int): 裁剪上边界像素（默认90）
            crop_bottom (int): 裁剪下边界像素（默认485）
        
        图像处理流程：
            1. 裁剪 [90:485, :] 去除天空和车头，得到 395x800
            2. 双三次插值缩放到 88x200
        """
        self.device = device
        self.enable_crop = enable_crop
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
    
    def preprocess(self, image):
        """
        预处理图像（与数据收集时保持一致）
        
        处理流程：
        1. 裁剪 [90:485, :] 去除天空和车头
        2. 双三次插值缩放到 88x200
        3. 归一化到 [0, 1]
        4. 转换为 PyTorch 张量
        
        参数:
            image: numpy数组 (H, W, 3)，RGB格式，值范围 [0, 255]
            
        返回:
            torch.Tensor: (1, 3, 88, 200)，值范围 [0, 1]
        """
        # 步骤1: 图像裁剪 [90:485, :] 去除天空和车头
        if self.enable_crop:
            image = image[self.crop_top:self.crop_bottom, :, :]
        
        # 步骤2: 双三次插值缩放到 88x200
        if image.shape[0] != IMAGE_HEIGHT or image.shape[1] != IMAGE_WIDTH:
            image_input = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), 
                                     interpolation=cv2.INTER_CUBIC)
        else:
            image_input = image
        
        # 步骤3: 转换数据类型
        image_input = image_input.astype(np.float32)
        
        # 步骤4: 调整维度顺序 (H, W, C) -> (C, H, W)
        image_input = np.transpose(image_input, (2, 0, 1))
        
        # 步骤5: 增加batch维度 -> (1, C, H, W)
        image_input = np.expand_dims(image_input, axis=0)
        
        # 步骤6: 归一化到 [0, 1] (与训练时保持一致)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        
        # 步骤7: 转换为PyTorch张量并移到设备
        img_tensor = torch.from_numpy(image_input).to(self.device)
        
        return img_tensor
    
    def get_processed_image(self, image):
        """
        获取处理后的图像（用于可视化，不转换为tensor）
        
        参数:
            image: numpy数组 (H, W, 3)，RGB格式，值范围 [0, 255]
            
        返回:
            numpy数组: (88, 200, 3)，RGB格式，值范围 [0, 255]
        """
        # 步骤1: 图像裁剪 [90:485, :]
        if self.enable_crop:
            image = image[self.crop_top:self.crop_bottom, :, :]
        
        # 步骤2: 双三次插值缩放到 88x200
        if image.shape[0] != IMAGE_HEIGHT or image.shape[1] != IMAGE_WIDTH:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), 
                               interpolation=cv2.INTER_CUBIC)
        
        return image
