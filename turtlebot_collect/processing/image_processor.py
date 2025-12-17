#!/usr/bin/env python
# coding=utf-8
'''
图像处理模块
负责图像的预处理，与 CARLA 训练数据保持一致
'''

import cv2

from ..config import ImageConfig


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, 
                 output_width=None, 
                 output_height=None,
                 crop_top_ratio=None,
                 crop_bottom_ratio=None):
        """
        初始化图像处理器
        
        参数:
            output_width (int): 输出图像宽度，None 使用配置默认值
            output_height (int): 输出图像高度，None 使用配置默认值
            crop_top_ratio (float): 裁剪顶部比例 (0.0-1.0)，None 使用配置默认值
            crop_bottom_ratio (float): 裁剪底部比例 (0.0-1.0)，None 使用配置默认值
        """
        self.output_width = output_width or ImageConfig.OUTPUT_WIDTH
        self.output_height = output_height or ImageConfig.OUTPUT_HEIGHT
        self.crop_top_ratio = crop_top_ratio if crop_top_ratio is not None else ImageConfig.CROP_TOP_RATIO
        self.crop_bottom_ratio = crop_bottom_ratio if crop_bottom_ratio is not None else ImageConfig.CROP_BOTTOM_RATIO
        
    def process(self, image):
        """
        处理图像
        
        参数:
            image (np.ndarray): 输入图像 (H, W, 3) RGB格式
            
        返回:
            np.ndarray: 处理后的图像 (output_height, output_width, 3)
        """
        if image is None:
            return None
        
        # 裁剪
        cropped = self._crop(image)
        
        # 缩放
        resized = self._resize(cropped)
        
        return resized
    
    def _crop(self, image):
        """裁剪图像"""
        h, w = image.shape[:2]
        
        top = int(h * self.crop_top_ratio)
        bottom = h - int(h * self.crop_bottom_ratio)
        
        return image[top:bottom, :]
    
    def _resize(self, image):
        """缩放图像"""
        return cv2.resize(image, (self.output_width, self.output_height))
    
    def bgr_to_rgb(self, image):
        """BGR 转 RGB"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def rgb_to_bgr(self, image):
        """RGB 转 BGR"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
