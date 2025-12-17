#!/usr/bin/env python
# coding=utf-8
'''
键盘控制器模块
处理键盘输入并发送速度命令

注意: 此模块仅支持 Linux/Unix 系统，Windows 用户请使用手柄控制器
'''

import sys
import platform
import rospy
from geometry_msgs.msg import Twist

from ..config import TopicConfig, KeyboardConfig, CommandConfig

# 检查平台兼容性
_IS_WINDOWS = platform.system() == 'Windows'

if not _IS_WINDOWS:
    import tty
    import termios
else:
    rospy.logwarn("键盘控制器不支持 Windows 系统，请使用手柄控制器 (--control joystick)")


class KeyboardController:
    """键盘控制器 (仅支持 Linux/Unix)"""
    
    def __init__(self, max_linear=0.22, max_angular=2.84):
        """
        初始化键盘控制器
        
        参数:
            max_linear (float): 最大线速度 (m/s)
            max_angular (float): 最大角速度 (rad/s)
            
        异常:
            RuntimeError: 在 Windows 系统上初始化时抛出
        """
        if _IS_WINDOWS:
            raise RuntimeError(
                "键盘控制器不支持 Windows 系统。\n"
                "请使用手柄控制器: python collector.py --control joystick"
            )
        
        self.max_linear = max_linear
        self.max_angular = max_angular
        
        # 当前状态
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # 回调函数
        self.on_record_start = None
        self.on_record_stop = None
        self.on_record_toggle = None
        self.on_command_change = None
        self.on_emergency_stop = None
        self.on_quit = None
        
        # 终端设置 (仅 Unix)
        self.settings = termios.tcgetattr(sys.stdin)
        
        # ROS 发布者
        self.cmd_pub = rospy.Publisher(TopicConfig.CMD_VEL, Twist, queue_size=1)
        
    def get_key(self):
        """获取键盘输入（非阻塞）"""
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def process_key(self, key):
        """
        处理按键输入
        
        参数:
            key (str): 按键字符
            
        返回:
            bool: 是否继续运行
        """
        # 退出
        if key == KeyboardConfig.KEY_QUIT:
            if self.on_quit:
                self.on_quit()
            return False
        
        # 移动控制
        if key.lower() in KeyboardConfig.MOVE_BINDINGS:
            linear, angular = KeyboardConfig.MOVE_BINDINGS[key.lower()]
            self._send_velocity(linear, angular)
        
        # 导航命令 (使用 CommandConfig 中的常量)
        if key in KeyboardConfig.COMMAND_BINDINGS:
            command = KeyboardConfig.COMMAND_BINDINGS[key]
            if self.on_command_change:
                cmd_name = CommandConfig.COMMAND_NAMES.get(command, 'Unknown')
                self.on_command_change(command, cmd_name)
        
        # 录制控制
        if key.lower() == KeyboardConfig.KEY_RECORD:
            if self.on_record_toggle:
                self.on_record_toggle()
        
        return True
    
    def _send_velocity(self, linear_ratio, angular_ratio):
        """发送速度命令"""
        twist = Twist()
        twist.linear.x = linear_ratio * self.max_linear
        twist.angular.z = angular_ratio * self.max_angular
        self.cmd_pub.publish(twist)
        
        self.linear_vel = twist.linear.x
        self.angular_vel = twist.angular.z
    
    def stop(self):
        """停止机器人"""
        self._send_velocity(0, 0)
    
    def get_velocity(self):
        """获取当前速度命令"""
        return self.linear_vel, self.angular_vel
    
    def cleanup(self):
        """清理终端设置"""
        if not _IS_WINDOWS:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
    
    def set_callbacks(self, on_record_start=None, on_record_stop=None, 
                     on_record_toggle=None, on_command_change=None, 
                     on_emergency_stop=None, on_quit=None):
        """
        设置回调函数
        
        参数:
            on_record_start: 开始录制回调 (键盘模式下不使用，使用 toggle)
            on_record_stop: 停止录制回调 (键盘模式下不使用，使用 toggle)
            on_record_toggle: 录制切换回调 ()
            on_command_change: 命令变化回调 (command, name)
            on_emergency_stop: 紧急停止回调 ()
            on_quit: 退出回调 ()
        """
        # 键盘模式使用 toggle 方式，但也保存 start/stop 以备用
        self.on_record_start = on_record_start
        self.on_record_stop = on_record_stop
        self.on_record_toggle = on_record_toggle
        self.on_command_change = on_command_change
        self.on_emergency_stop = on_emergency_stop
        self.on_quit = on_quit
    
    def print_controls(self):
        """打印控制说明"""
        print("\n" + "="*50)
        print("键盘控制说明")
        print("="*50)
        print("\n移动控制:")
        print("  W/S - 前进/后退")
        print("  A/D - 左转/右转")
        print("  Q/E - 左前/右前")
        print("  空格 - 停止")
        print("\n导航命令:")
        print("  1 - Follow | 2 - Left | 3 - Right | 4 - Straight")
        print("\n录制控制:")
        print("  R - 开始/停止录制")
        print("  ESC - 退出")
        print("="*50 + "\n")
