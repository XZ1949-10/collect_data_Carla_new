#!/usr/bin/env python
# coding=utf-8
'''
原生 ROS 数据结构收集模块
负责订阅和收集原始 ROS 话题数据
'''

import rospy
import numpy as np
from threading import Lock

from sensor_msgs.msg import Image, Joy, LaserScan, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from config import TopicConfig


class ROSDataCollector:
    """
    原生 ROS 数据收集器
    
    收集原始 ROS 话题数据，不做任何转换处理
    """
    
    def __init__(self):
        """初始化 ROS 数据收集器"""
        self.lock = Lock()
        
        # 原始数据存储
        self._raw_image = None           # sensor_msgs/Image
        self._raw_odom = None            # nav_msgs/Odometry
        self._raw_cmd_vel = None         # geometry_msgs/Twist
        self._raw_joy = None             # sensor_msgs/Joy
        self._raw_imu = None             # sensor_msgs/Imu (可选)
        self._raw_scan = None            # sensor_msgs/LaserScan (可选)
        
        # 时间戳
        self._image_stamp = None
        self._odom_stamp = None
        self._cmd_vel_stamp = None
        self._joy_stamp = None
        
        # 订阅者列表
        self._subscribers = []
        
    def setup_subscribers(self, 
                          subscribe_image=True,
                          subscribe_odom=True,
                          subscribe_cmd_vel=True,
                          subscribe_joy=True,
                          subscribe_imu=False,
                          subscribe_scan=False):
        """
        设置 ROS 订阅者
        
        参数:
            subscribe_image (bool): 是否订阅图像话题
            subscribe_odom (bool): 是否订阅里程计话题
            subscribe_cmd_vel (bool): 是否订阅速度命令话题
            subscribe_joy (bool): 是否订阅手柄话题
            subscribe_imu (bool): 是否订阅 IMU 话题
            subscribe_scan (bool): 是否订阅激光雷达话题
        """
        if subscribe_image:
            sub = rospy.Subscriber(
                TopicConfig.IMAGE_RAW, Image, 
                self._image_callback, queue_size=1
            )
            self._subscribers.append(sub)
            rospy.loginfo(f"订阅图像话题: {TopicConfig.IMAGE_RAW}")
        
        if subscribe_odom:
            sub = rospy.Subscriber(
                TopicConfig.ODOM, Odometry,
                self._odom_callback, queue_size=1
            )
            self._subscribers.append(sub)
            rospy.loginfo(f"订阅里程计话题: {TopicConfig.ODOM}")
        
        if subscribe_cmd_vel:
            sub = rospy.Subscriber(
                TopicConfig.CMD_VEL, Twist,
                self._cmd_vel_callback, queue_size=1
            )
            self._subscribers.append(sub)
            rospy.loginfo(f"订阅速度命令话题: {TopicConfig.CMD_VEL}")
        
        if subscribe_joy:
            sub = rospy.Subscriber(
                TopicConfig.JOY, Joy,
                self._joy_callback, queue_size=1
            )
            self._subscribers.append(sub)
            rospy.loginfo(f"订阅手柄话题: {TopicConfig.JOY}")
        
        if subscribe_imu:
            sub = rospy.Subscriber(
                TopicConfig.IMU, Imu,
                self._imu_callback, queue_size=1
            )
            self._subscribers.append(sub)
            rospy.loginfo(f"订阅 IMU 话题: {TopicConfig.IMU}")
        
        if subscribe_scan:
            sub = rospy.Subscriber(
                TopicConfig.SCAN, LaserScan,
                self._scan_callback, queue_size=1
            )
            self._subscribers.append(sub)
            rospy.loginfo(f"订阅激光雷达话题: {TopicConfig.SCAN}")
    
    # ============ 回调函数 ============
    
    def _image_callback(self, msg):
        """图像话题回调"""
        with self.lock:
            self._raw_image = msg
            self._image_stamp = msg.header.stamp
    
    def _odom_callback(self, msg):
        """里程计话题回调"""
        with self.lock:
            self._raw_odom = msg
            self._odom_stamp = msg.header.stamp
    
    def _cmd_vel_callback(self, msg):
        """速度命令话题回调"""
        with self.lock:
            self._raw_cmd_vel = msg
            self._cmd_vel_stamp = rospy.Time.now()
    
    def _joy_callback(self, msg):
        """手柄话题回调"""
        with self.lock:
            self._raw_joy = msg
            self._joy_stamp = msg.header.stamp
    
    def _imu_callback(self, msg):
        """IMU 话题回调"""
        with self.lock:
            self._raw_imu = msg
    
    def _scan_callback(self, msg):
        """激光雷达话题回调"""
        with self.lock:
            self._raw_scan = msg
    
    # ============ 数据获取接口 ============
    
    def get_raw_image(self):
        """获取原始图像消息"""
        with self.lock:
            return self._raw_image
    
    def get_raw_odom(self):
        """获取原始里程计消息"""
        with self.lock:
            return self._raw_odom
    
    def get_raw_cmd_vel(self):
        """获取原始速度命令消息"""
        with self.lock:
            return self._raw_cmd_vel
    
    def get_raw_joy(self):
        """获取原始手柄消息"""
        with self.lock:
            return self._raw_joy
    
    def get_raw_imu(self):
        """获取原始 IMU 消息"""
        with self.lock:
            return self._raw_imu
    
    def get_raw_scan(self):
        """获取原始激光雷达消息"""
        with self.lock:
            return self._raw_scan
    
    def get_odom_data(self):
        """
        获取里程计数据（解析后的数值）
        
        返回:
            dict: {
                'position': (x, y, z),
                'orientation': (x, y, z, w),
                'linear_vel': (x, y, z),
                'angular_vel': (x, y, z),
            } 或 None
        """
        with self.lock:
            if self._raw_odom is None:
                return None
            
            odom = self._raw_odom
            return {
                'position': (
                    odom.pose.pose.position.x,
                    odom.pose.pose.position.y,
                    odom.pose.pose.position.z
                ),
                'orientation': (
                    odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w
                ),
                'linear_vel': (
                    odom.twist.twist.linear.x,
                    odom.twist.twist.linear.y,
                    odom.twist.twist.linear.z
                ),
                'angular_vel': (
                    odom.twist.twist.angular.x,
                    odom.twist.twist.angular.y,
                    odom.twist.twist.angular.z
                ),
            }
    
    def get_cmd_vel_data(self):
        """
        获取速度命令数据（解析后的数值）
        
        返回:
            dict: {
                'linear_vel': float,   # m/s
                'angular_vel': float,  # rad/s
            } 或 None
        """
        with self.lock:
            if self._raw_cmd_vel is None:
                return None
            
            return {
                'linear_vel': self._raw_cmd_vel.linear.x,
                'angular_vel': self._raw_cmd_vel.angular.z,
            }
    
    def get_joy_data(self):
        """
        获取手柄数据（解析后的数值）
        
        返回:
            dict: {
                'axes': list,      # 摇杆轴值列表
                'buttons': list,   # 按钮状态列表
            } 或 None
        """
        with self.lock:
            if self._raw_joy is None:
                return None
            
            return {
                'axes': list(self._raw_joy.axes),
                'buttons': list(self._raw_joy.buttons),
            }
    
    def get_timestamps(self):
        """
        获取传感器时间戳
        
        返回:
            dict: {
                'image': rospy.Time 或 None,
                'odom': rospy.Time 或 None,
            }
        """
        with self.lock:
            return {
                'image': self._image_stamp,
                'odom': self._odom_stamp,
            }
    
    def get_speed(self):
        """
        获取当前速度 (m/s)
        
        返回:
            float: 速度值，如果无数据返回 0.0
        """
        with self.lock:
            if self._raw_odom is None:
                return 0.0
            vx = self._raw_odom.twist.twist.linear.x
            vy = self._raw_odom.twist.twist.linear.y
            return np.sqrt(vx**2 + vy**2)
    
    def has_data(self):
        """检查是否有数据"""
        with self.lock:
            return self._raw_image is not None
    
    def cleanup(self):
        """清理订阅者"""
        for sub in self._subscribers:
            sub.unregister()
        self._subscribers.clear()
