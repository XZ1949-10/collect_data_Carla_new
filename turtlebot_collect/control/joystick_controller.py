#!/usr/bin/env python
# coding=utf-8
'''
手柄控制器模块
处理手柄输入并发送速度命令
'''

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from config import TopicConfig, JoystickConfig, CommandConfig


class JoystickController:
    """手柄控制器"""
    
    def __init__(self, 
                 max_linear=0.22, 
                 max_angular=2.84,
                 joystick_type='xbox'):
        """
        初始化手柄控制器
        
        参数:
            max_linear (float): 最大线速度 (m/s)
            max_angular (float): 最大角速度 (rad/s)
            joystick_type (str): 手柄类型 ('xbox' 或 'ps4')
        """
        self.max_linear = max_linear
        self.max_angular = max_angular
        
        # 加载手柄配置
        if joystick_type.lower() == 'ps4':
            self.config = JoystickConfig.PS4
        else:
            self.config = JoystickConfig.Xbox
        
        # 当前状态
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.last_btn_states = {}
        
        # 控制启用状态 (防误触功能)
        self.control_enabled = False
        
        # 回调函数
        self.on_record_start = None
        self.on_record_stop = None
        self.on_record_toggle = None
        self.on_command_change = None
        self.on_emergency_stop = None
        self.on_quit = None
        self.on_control_toggle = None  # 控制启用/禁用回调
        
        # ROS 发布者
        self.cmd_pub = rospy.Publisher(TopicConfig.CMD_VEL, Twist, queue_size=1)
        
        # ROS 订阅者
        rospy.Subscriber(TopicConfig.JOY, Joy, self._joy_callback, queue_size=1)
        
    def _joy_callback(self, msg):
        """手柄消息回调"""
        # 先处理按钮 (包括控制启用/禁用)
        self._handle_buttons(msg.buttons)
        
        # 处理摇杆 (只有启用控制时才发送速度命令)
        if len(msg.axes) > max(self.config.AXIS_LINEAR, self.config.AXIS_ANGULAR):
            linear = msg.axes[self.config.AXIS_LINEAR]
            angular = msg.axes[self.config.AXIS_ANGULAR]
            
            if self.control_enabled:
                self._send_velocity(linear, angular)
            else:
                # 控制禁用时，只更新内部状态但不发送命令
                self.linear_vel = linear * self.max_linear
                self.angular_vel = angular * self.max_angular
                # 发送零速度确保机器人停止
                self._send_zero_velocity()
    
    def _handle_buttons(self, buttons):
        """处理按钮输入"""
        def btn_pressed(btn_id):
            """检测按钮按下（上升沿）"""
            if btn_id >= len(buttons):
                return False
            current = buttons[btn_id]
            last = self.last_btn_states.get(btn_id, 0)
            self.last_btn_states[btn_id] = current
            return current == 1 and last == 0
        
        # 控制启用/禁用切换 (优先处理，防误触)
        if hasattr(self.config, 'BTN_ENABLE_CONTROL') and btn_pressed(self.config.BTN_ENABLE_CONTROL):
            self.control_enabled = not self.control_enabled
            if not self.control_enabled:
                # 禁用时立即停止机器人
                self._send_zero_velocity()
            if self.on_control_toggle:
                self.on_control_toggle(self.control_enabled)
            rospy.loginfo(f"控制已{'启用' if self.control_enabled else '禁用'}")
        
        # 录制控制
        if btn_pressed(self.config.BTN_RECORD):
            if self.on_record_start:
                self.on_record_start()
        
        # 停止录制
        if btn_pressed(self.config.BTN_STOP):
            if self.on_record_stop:
                self.on_record_stop()
        
        # 紧急停止 (无论控制是否启用都生效)
        if hasattr(self.config, 'BTN_EMERGENCY') and btn_pressed(self.config.BTN_EMERGENCY):
            self.control_enabled = False  # 紧急停止时也禁用控制
            self._send_zero_velocity()
            if self.on_control_toggle:
                self.on_control_toggle(False)
            if self.on_emergency_stop:
                self.on_emergency_stop()
        
        # 导航命令 (使用 CommandConfig 中的常量)
        if btn_pressed(self.config.BTN_FOLLOW):
            if self.on_command_change:
                self.on_command_change(CommandConfig.CMD_FOLLOW, 
                                       CommandConfig.COMMAND_NAMES[CommandConfig.CMD_FOLLOW])
        elif btn_pressed(self.config.BTN_LEFT):
            if self.on_command_change:
                self.on_command_change(CommandConfig.CMD_LEFT, 
                                       CommandConfig.COMMAND_NAMES[CommandConfig.CMD_LEFT])
        elif btn_pressed(self.config.BTN_RIGHT):
            if self.on_command_change:
                self.on_command_change(CommandConfig.CMD_RIGHT, 
                                       CommandConfig.COMMAND_NAMES[CommandConfig.CMD_RIGHT])
        elif btn_pressed(self.config.BTN_STRAIGHT):
            if self.on_command_change:
                self.on_command_change(CommandConfig.CMD_STRAIGHT, 
                                       CommandConfig.COMMAND_NAMES[CommandConfig.CMD_STRAIGHT])
    
    def _send_velocity(self, linear_ratio, angular_ratio):
        """
        发送速度命令
        
        参数:
            linear_ratio (float): 线速度比例 (-1.0 ~ 1.0)
            angular_ratio (float): 角速度比例 (-1.0 ~ 1.0)
        """
        twist = Twist()
        twist.linear.x = linear_ratio * self.max_linear
        twist.angular.z = angular_ratio * self.max_angular
        self.cmd_pub.publish(twist)
        
        self.linear_vel = twist.linear.x
        self.angular_vel = twist.angular.z
    
    def _send_zero_velocity(self):
        """发送零速度命令 (停止机器人)"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
    
    def stop(self):
        """停止机器人"""
        self._send_velocity(0, 0)
    
    def is_control_enabled(self):
        """获取控制启用状态"""
        return self.control_enabled
    
    def set_control_enabled(self, enabled):
        """设置控制启用状态"""
        self.control_enabled = enabled
        if not enabled:
            self._send_zero_velocity()
    
    def get_velocity(self):
        """获取当前速度命令"""
        return self.linear_vel, self.angular_vel
    
    def set_callbacks(self, on_record_start=None, on_record_stop=None, 
                     on_record_toggle=None, on_command_change=None, 
                     on_emergency_stop=None, on_quit=None, on_control_toggle=None):
        """
        设置回调函数
        
        参数:
            on_record_start: 开始录制回调 ()
            on_record_stop: 停止录制回调 ()
            on_record_toggle: 录制切换回调 (手柄模式下不使用，使用 start/stop)
            on_command_change: 命令变化回调 (command, name)
            on_emergency_stop: 紧急停止回调 ()
            on_quit: 退出回调 ()
            on_control_toggle: 控制启用/禁用回调 (enabled: bool)
        """
        self.on_record_start = on_record_start
        self.on_record_stop = on_record_stop
        self.on_record_toggle = on_record_toggle
        self.on_command_change = on_command_change
        self.on_emergency_stop = on_emergency_stop
        self.on_quit = on_quit
        self.on_control_toggle = on_control_toggle
    
    def cleanup(self):
        """清理资源（与 KeyboardController 接口一致）"""
        # 手柄控制器不需要特殊清理，但保持接口一致
        self.stop()
