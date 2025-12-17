#!/usr/bin/env python
# coding=utf-8
'''
ROS 控制器接口
发布速度命令到 TurtleBot
'''

import rospy
from geometry_msgs.msg import Twist

from ..config.topics import TopicConfig


class ROSController:
    """ROS 控制器接口"""
    
    def __init__(self, cmd_vel_topic=None):
        """
        初始化控制器
        
        参数:
            cmd_vel_topic (str): 速度命令话题，None 使用默认
        """
        self.cmd_vel_topic = cmd_vel_topic or TopicConfig.CMD_VEL
        self._publisher = None
        
    def setup(self):
        """设置发布者"""
        print(f"发布速度命令到: {self.cmd_vel_topic}")
        self._publisher = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1
        )
        
    def send_velocity(self, linear_vel, angular_vel):
        """
        发送速度命令
        
        参数:
            linear_vel (float): 线速度 (m/s)
            angular_vel (float): 角速度 (rad/s)
        """
        if self._publisher is None:
            rospy.logwarn("控制器未初始化")
            return
            
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self._publisher.publish(twist)
        
    def stop(self):
        """停止机器人"""
        self.send_velocity(0.0, 0.0)
        
    def cleanup(self):
        """清理"""
        self.stop()
        if self._publisher is not None:
            self._publisher.unregister()
