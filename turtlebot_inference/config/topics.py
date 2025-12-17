#!/usr/bin/env python
# coding=utf-8
'''
ROS 话题配置
'''


class TopicConfig:
    """ROS 话题配置"""
    
    # ============ 图像话题 ============
    IMAGE_RAW = '/camera/rgb/image_raw'
    
    # 备选话题
    # IMAGE_RAW = '/raspicam_node/image'        # 树莓派摄像头
    # IMAGE_RAW = '/usb_cam/image_raw'          # USB摄像头
    
    # ============ 控制话题 ============
    CMD_VEL = '/cmd_vel'
    # CMD_VEL = '/mobile_base/commands/velocity'  # Kobuki
    
    # ============ 传感器话题 ============
    ODOM = '/odom'
    
    # ============ 手柄话题 ============
    JOY = '/joy'


class JoystickConfig:
    """手柄按键配置（用于切换导航命令）"""
    
    class Xbox:
        BTN_Y = 3        # Follow
        BTN_X = 2        # Left
        BTN_B = 1        # Right
        BTN_A = 0        # Straight
        BTN_START = 7    # 开始/停止推理
        BTN_BACK = 6     # 退出
        
    class PS4:
        BTN_TRIANGLE = 3  # Follow
        BTN_SQUARE = 2    # Left
        BTN_O = 1         # Right
        BTN_X = 0         # Straight
        BTN_OPTIONS = 7   # 开始/停止推理
        BTN_SHARE = 6     # 退出
