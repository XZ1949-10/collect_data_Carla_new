#!/usr/bin/env python
# coding=utf-8
'''
原生 ROS 图像数据处理模块
负责 ROS Image 消息的解析和基础处理
'''

import cv2
import rospy

from cv_bridge import CvBridge


class ROSImageHandler:
    """
    ROS 图像处理器
    
    处理原生 ROS Image 消息，提供格式转换功能
    """
    
    def __init__(self):
        """初始化图像处理器"""
        self.bridge = CvBridge()
    
    def msg_to_numpy(self, msg, encoding="rgb8"):
        """
        将 ROS Image 消息转换为 NumPy 数组
        
        参数:
            msg (sensor_msgs/Image): ROS 图像消息
            encoding (str): 目标编码格式
                - "rgb8": RGB 格式
                - "bgr8": BGR 格式 (OpenCV 默认)
                - "mono8": 灰度图
                
        返回:
            np.ndarray: 图像数组，如果失败返回 None
        """
        if msg is None:
            return None
        
        try:
            return self.bridge.imgmsg_to_cv2(msg, encoding)
        except Exception as e:
            rospy.logwarn(f"图像转换错误: {e}")
            return None
    
    def numpy_to_msg(self, image, encoding="rgb8"):
        """
        将 NumPy 数组转换为 ROS Image 消息
        
        参数:
            image (np.ndarray): 图像数组
            encoding (str): 编码格式
            
        返回:
            sensor_msgs/Image: ROS 图像消息
        """
        try:
            return self.bridge.cv2_to_imgmsg(image, encoding)
        except Exception as e:
            rospy.logwarn(f"图像转换错误: {e}")
            return None
    
    def get_image_info(self, msg):
        """
        获取图像消息的基本信息
        
        参数:
            msg (sensor_msgs/Image): ROS 图像消息
            
        返回:
            dict: {
                'width': int,
                'height': int,
                'encoding': str,
                'step': int,
                'is_bigendian': bool,
                'timestamp': rospy.Time,
            }
        """
        if msg is None:
            return None
        
        return {
            'width': msg.width,
            'height': msg.height,
            'encoding': msg.encoding,
            'step': msg.step,
            'is_bigendian': msg.is_bigendian,
            'timestamp': msg.header.stamp,
        }
    
    def convert_encoding(self, image, from_encoding, to_encoding):
        """
        转换图像编码格式
        
        参数:
            image (np.ndarray): 输入图像
            from_encoding (str): 源编码 ("rgb8", "bgr8")
            to_encoding (str): 目标编码 ("rgb8", "bgr8")
            
        返回:
            np.ndarray: 转换后的图像
        """
        if from_encoding == to_encoding:
            return image
        
        if from_encoding == "rgb8" and to_encoding == "bgr8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif from_encoding == "bgr8" and to_encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif to_encoding == "mono8":
            if from_encoding == "rgb8":
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif from_encoding == "bgr8":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def resize_image(self, image, width, height):
        """
        缩放图像
        
        参数:
            image (np.ndarray): 输入图像
            width (int): 目标宽度
            height (int): 目标高度
            
        返回:
            np.ndarray: 缩放后的图像
        """
        return cv2.resize(image, (width, height))
    
    def crop_image(self, image, top=0, bottom=0, left=0, right=0):
        """
        裁剪图像
        
        参数:
            image (np.ndarray): 输入图像
            top (int): 顶部裁剪像素数
            bottom (int): 底部裁剪像素数
            left (int): 左侧裁剪像素数
            right (int): 右侧裁剪像素数
            
        返回:
            np.ndarray: 裁剪后的图像
        """
        h, w = image.shape[:2]
        bottom_idx = h - bottom if bottom > 0 else h
        right_idx = w - right if right > 0 else w
        return image[top:bottom_idx, left:right_idx]
    
    def crop_image_ratio(self, image, top_ratio=0.0, bottom_ratio=0.0):
        """
        按比例裁剪图像
        
        参数:
            image (np.ndarray): 输入图像
            top_ratio (float): 顶部裁剪比例 (0.0-1.0)
            bottom_ratio (float): 底部裁剪比例 (0.0-1.0)
            
        返回:
            np.ndarray: 裁剪后的图像
        """
        h, w = image.shape[:2]
        top = int(h * top_ratio)
        bottom = int(h * bottom_ratio)
        return self.crop_image(image, top=top, bottom=bottom)
