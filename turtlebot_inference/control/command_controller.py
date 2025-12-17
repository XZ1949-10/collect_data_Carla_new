#!/usr/bin/env python
# coding=utf-8
'''
导航命令控制器
通过手柄或键盘切换导航命令
'''

from threading import Lock
import rospy
from sensor_msgs.msg import Joy

from ..config.topics import TopicConfig, JoystickConfig
from ..config.inference_config import (
    COMMAND_FOLLOW, COMMAND_LEFT, COMMAND_RIGHT, COMMAND_STRAIGHT,
    DEFAULT_COMMAND, COMMAND_NAMES
)


class CommandController:
    """导航命令控制器"""
    
    def __init__(self, joystick_type='xbox', joy_topic=None):
        """
        初始化命令控制器
        
        参数:
            joystick_type (str): 手柄类型 ('xbox' 或 'ps4')
            joy_topic (str): 手柄话题，None 使用默认
        """
        self.joystick_type = joystick_type.lower()
        self.joy_topic = joy_topic or TopicConfig.JOY
        
        self.lock = Lock()
        self._current_command = DEFAULT_COMMAND
        self._running = True
        self._subscriber = None
        
        # 按钮映射
        if self.joystick_type == 'ps4':
            self.btn_config = JoystickConfig.PS4
        else:
            self.btn_config = JoystickConfig.Xbox
            
        # 按钮状态（防止重复触发）
        self._prev_buttons = []
        
    def setup(self):
        """设置订阅者"""
        print(f"订阅手柄话题: {self.joy_topic}")
        self._subscriber = rospy.Subscriber(
            self.joy_topic, Joy, self._joy_callback, queue_size=1
        )
        
    def _joy_callback(self, msg):
        """手柄回调"""
        buttons = list(msg.buttons)
        
        with self.lock:
            # 检测按钮按下（上升沿）
            if len(self._prev_buttons) == len(buttons):
                # Follow
                if self._button_pressed(buttons, self.btn_config.BTN_Y 
                                        if hasattr(self.btn_config, 'BTN_Y') 
                                        else self.btn_config.BTN_TRIANGLE):
                    self._current_command = COMMAND_FOLLOW
                    print(f"命令切换: {COMMAND_NAMES[self._current_command]}")
                    
                # Left
                elif self._button_pressed(buttons, self.btn_config.BTN_X 
                                          if hasattr(self.btn_config, 'BTN_X') 
                                          else self.btn_config.BTN_SQUARE):
                    self._current_command = COMMAND_LEFT
                    print(f"命令切换: {COMMAND_NAMES[self._current_command]}")
                    
                # Right
                elif self._button_pressed(buttons, self.btn_config.BTN_B 
                                          if hasattr(self.btn_config, 'BTN_B') 
                                          else self.btn_config.BTN_O):
                    self._current_command = COMMAND_RIGHT
                    print(f"命令切换: {COMMAND_NAMES[self._current_command]}")
                    
                # Straight
                elif self._button_pressed(buttons, self.btn_config.BTN_A 
                                          if hasattr(self.btn_config, 'BTN_A') 
                                          else self.btn_config.BTN_X):
                    self._current_command = COMMAND_STRAIGHT
                    print(f"命令切换: {COMMAND_NAMES[self._current_command]}")
                    
                # 退出
                elif self._button_pressed(buttons, self.btn_config.BTN_BACK 
                                          if hasattr(self.btn_config, 'BTN_BACK') 
                                          else self.btn_config.BTN_SHARE):
                    self._running = False
                    print("收到退出信号")
                    
            self._prev_buttons = buttons
            
    def _button_pressed(self, buttons, btn_idx):
        """检测按钮是否刚被按下"""
        if btn_idx >= len(buttons) or btn_idx >= len(self._prev_buttons):
            return False
        return buttons[btn_idx] == 1 and self._prev_buttons[btn_idx] == 0
    
    def get_command(self):
        """获取当前导航命令"""
        with self.lock:
            return self._current_command
    
    def set_command(self, command):
        """设置导航命令"""
        if command in COMMAND_NAMES:
            with self.lock:
                self._current_command = command
                print(f"命令设置: {COMMAND_NAMES[command]}")
                
    def is_running(self):
        """检查是否继续运行"""
        with self.lock:
            return self._running
            
    def stop(self):
        """停止"""
        with self.lock:
            self._running = False
            
    def cleanup(self):
        """清理"""
        if self._subscriber is not None:
            self._subscriber.unregister()
