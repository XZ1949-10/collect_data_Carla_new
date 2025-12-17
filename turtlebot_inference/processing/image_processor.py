#!/usr/bin/env python
# coding=utf-8
'''
图像预处理模块
将 ROS 图像转换为模型输入格式
'''

import numpy as np
import cv2
import torch

from ..config.inference_config import IMAGE_WIDTH, IMAGE_HEIGHT, CROP_TOP_RATIO, CROP_BOTTOM_RATIO


class ImageProcessor:
    """图像预处理器"""
    
    def __init__(self, device, 
                 output_width=IMAGE_WIDTH, 
                 output_height=IMAGE_HEIGHT,
                 crop_top_ratio=CROP_TOP_RATIO,
                 crop_bottom_ratio=CROP_BOTTOM_RATIO):
        """
        初始化图像预处理器
        
        参数:
            device: torch.device 对象
            output_width (int): 输出图像宽度
            output_height (int): 输出图像高度
            crop_top_ratio (float): 裁剪顶部比例 (0.0-1.0)
            crop_bottom_ratio (float): 裁剪底部比例 (0.0-1.0)
        """
        self.device = device
        self.output_width = output_width
        self.output_height = output_height
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
    
    def preprocess(self, image):
        """
        预处理图像为模型输入
        
        参数:
            image: numpy 数组 (H, W, 3)，RGB 格式，值范围 [0, 255]
            
        返回:
            torch.Tensor: (1, 3, 88, 200)，值范围 [0, 1]
        """
        if image is None:
            return None
        
        # 1. 裁剪
        image = self._crop(image)
        
        # 2. 缩放
        image = self._resize(image)
        
        # 3. 转换为 float32
        image = image.astype(np.float32)
        
        # 4. 调整维度 (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # 5. 增加 batch 维度 -> (1, C, H, W)
        image = np.expand_dims(image, axis=0)
        
        # 6. 归一化到 [0, 1]
        image = image / 255.0
        
        # 7. 转换为 PyTorch 张量
        img_tensor = torch.from_numpy(image).to(self.device)
        
        return img_tensor
    
    def get_processed_image(self, image):
        """
        获取处理后的图像（用于可视化，不转换为 tensor）
        
        参数:
            image: numpy 数组 (H, W, 3)，RGB 格式
            
        返回:
            numpy 数组: (88, 200, 3)，RGB 格式
        """
        if image is None:
            return None
        
        image = self._crop(image)
        image = self._resize(image)
        return image
    
    def _crop(self, image):
        """裁剪图像"""
        h, w = image.shape[:2]
        top = int(h * self.crop_top_ratio)
        bottom = h - int(h * self.crop_bottom_ratio)
        return image[top:bottom, :]
    
    def _resize(self, image):
        """缩放图像"""
        return cv2.resize(image, (self.output_width, self.output_height), 
                          interpolation=cv2.INTER_CUBIC)
