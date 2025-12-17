#!/usr/bin/env python
# coding=utf-8
'''
ROS 传感器数据接口
订阅图像和里程计话题
'''

from threading import Lock
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from ..config.topics import TopicConfig


class ROSSensor:
    """ROS 传感器数据接口"""
    
    def __init__(self, image_topic=None, odom_topic=None):
        """
        初始化传感器接口
        
        参数:
            image_topic (str): 图像话题，None 使用默认
            odom_topic (str): 里程计话题，None 使用默认
        """
        self.image_topic = image_topic or TopicConfig.IMAGE_RAW
        self.odom_topic = odom_topic or TopicConfig.ODOM
        
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # 数据存储
        self._image = None
        self._odom = None
        self._linear_vel = 0.0
        self._angular_vel = 0.0
        
        # 订阅者
        self._subscribers = []
        
    def setup(self):
        """设置订阅者"""
        print(f"订阅图像话题: {self.image_topic}")
        self._subscribers.append(
            rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)
        )
        
        print(f"订阅里程计话题: {self.odom_topic}")
        self._subscribers.append(
            rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=1)
        )
        
    def _image_callback(self, msg):
        """图像回调"""
        with self.lock:
            try:
                self._image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            except Exception as e:
                rospy.logwarn(f"图像转换错误: {e}")
                
    def _odom_callback(self, msg):
        """里程计回调"""
        with self.lock:
            self._odom = msg
            self._linear_vel = msg.twist.twist.linear.x
            self._angular_vel = msg.twist.twist.angular.z
            
    def get_image(self):
        """获取当前图像 (RGB 格式)"""
        with self.lock:
            return self._image.copy() if self._image is not None else None
    
    def get_speed(self):
        """
        获取当前速度
        
        返回:
            dict: {
                'linear_vel': float,   # m/s
                'angular_vel': float,  # rad/s
            }
        """
        with self.lock:
            return {
                'linear_vel': self._linear_vel,
                'angular_vel': self._angular_vel,
            }
    
    def has_data(self):
        """检查是否有数据"""
        with self.lock:
            return self._image is not None
    
    def cleanup(self):
        """清理订阅者"""
        for sub in self._subscribers:
            sub.unregister()
        self._subscribers.clear()
