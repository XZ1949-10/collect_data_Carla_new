#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 数据收集主模块
整合所有子模块，提供统一的数据收集接口
'''

import sys
import os
# 确保可以找到同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rospy

from config import (
    TopicConfig, 
    CommandConfig, 
    ImageConfig,
    RobotConfig,
    StorageConfig,
    CollectorConfig
)
from ros_data import ROSDataCollector, ROSImageHandler
from processing import ImageProcessor, ControlConverter
from control import JoystickController, KeyboardController
from storage import DataSaver, DataBuffer
from visualization import CollectorVisualizer


class TurtleBotCollector:
    """TurtleBot 数据收集器"""
    
    def __init__(self,
                 output_dir=None,
                 control_type=None,
                 joystick_type=None,
                 turtlebot_model=None,
                 image_width=None,
                 image_height=None,
                 rate_hz=None,
                 frames_per_file=None,
                 enable_sensor_sync=None):
        """
        初始化收集器
        
        参数:
            output_dir (str): 数据输出目录，None 使用配置默认值
            control_type (str): 控制类型 ('joystick' 或 'keyboard')
            joystick_type (str): 手柄类型 ('xbox' 或 'ps4')
            turtlebot_model (str): TurtleBot 型号，None 使用默认
            image_width (int): 输出图像宽度，None 使用配置默认值
            image_height (int): 输出图像高度，None 使用配置默认值
            rate_hz (int): 收集频率 (Hz)，None 使用配置默认值
            frames_per_file (int): 每个文件的帧数，None 使用配置默认值
            enable_sensor_sync (bool): 是否启用传感器同步，None 使用配置默认值
        """
        rospy.init_node('turtlebot_collector', anonymous=True)
        
        # 使用配置默认值
        self.rate_hz = rate_hz or CollectorConfig.DEFAULT_RATE_HZ
        self.control_type = control_type or CollectorConfig.DEFAULT_CONTROL_TYPE
        joystick_type = joystick_type or CollectorConfig.DEFAULT_JOYSTICK_TYPE
        turtlebot_model = turtlebot_model or RobotConfig.DEFAULT_MODEL
        image_width = image_width or ImageConfig.OUTPUT_WIDTH
        image_height = image_height or ImageConfig.OUTPUT_HEIGHT
        output_dir = output_dir or StorageConfig.DEFAULT_OUTPUT_DIR
        frames_per_file = frames_per_file or StorageConfig.FRAMES_PER_FILE
        
        # 传感器同步配置
        self.enable_sensor_sync = enable_sensor_sync if enable_sensor_sync is not None else CollectorConfig.ENABLE_SENSOR_SYNC
        self.max_sensor_time_diff = CollectorConfig.MAX_SENSOR_TIME_DIFF
        
        # 初始化 ROS 数据收集器
        self.ros_collector = ROSDataCollector()
        self.image_handler = ROSImageHandler()
        
        # 初始化图像处理器
        self.image_processor = ImageProcessor(
            output_width=image_width,
            output_height=image_height
        )
        
        # 初始化其他子模块
        self.control_converter = ControlConverter(model=turtlebot_model)
        self.data_saver = DataSaver(output_dir=output_dir)
        self.visualizer = CollectorVisualizer()
        
        # 初始化数据缓冲区 (带自动保存回调)
        self.data_buffer = DataBuffer(
            frames_per_file=frames_per_file,
            auto_save_callback=self._auto_save_callback
        )
        
        # 初始化控制器
        max_linear = self.control_converter.max_linear
        max_angular = self.control_converter.max_angular
        
        if self.control_type == 'keyboard':
            self.controller = KeyboardController(max_linear, max_angular)
        else:
            self.controller = JoystickController(max_linear, max_angular, joystick_type)
        
        # 设置控制器回调
        self.controller.set_callbacks(
            on_record_start=self._start_recording,
            on_record_stop=self._stop_recording,
            on_command_change=self._on_command_change,
            on_record_toggle=self._on_record_toggle,
            on_emergency_stop=self._on_emergency_stop,
            on_quit=self._on_quit,
            on_control_toggle=self._on_control_toggle
        )
        
        # 状态
        self.current_image = None
        self.current_command = CommandConfig.DEFAULT_COMMAND
        self.is_collecting = False
        self.running = True
        self.sync_status = True  # 传感器同步状态
        self.control_enabled = False  # 控制启用状态 (默认禁用，防误触)
        
        # 丢帧统计
        self._sync_drop_count = 0      # 因同步问题丢弃的帧数
        self._total_attempt_count = 0  # 总尝试收集的帧数
        
        # 设置 ROS 订阅 (只订阅需要的话题)
        self.ros_collector.setup_subscribers(
            subscribe_image=True,
            subscribe_odom=True,
            subscribe_cmd_vel=False,  # 控制器自己处理
            subscribe_joy=False,      # 控制器自己处理
            subscribe_imu=False,
            subscribe_scan=False
        )
        
        rospy.loginfo("TurtleBot 数据收集器初始化完成")
        rospy.loginfo(f"控制类型: {self.control_type}")
        rospy.loginfo(f"输出目录: {output_dir}")
        rospy.loginfo(f"收集帧率: {self.rate_hz} Hz")
        rospy.loginfo(f"每文件帧数: {frames_per_file}")
        rospy.loginfo(f"传感器同步: {'启用' if self.enable_sensor_sync else '禁用'}")

    def _get_processed_image(self):
        """获取处理后的图像"""
        raw_image = self.ros_collector.get_raw_image()
        if raw_image is None:
            return None
        
        # 转换为 numpy 数组
        cv_image = self.image_handler.msg_to_numpy(raw_image, "rgb8")
        if cv_image is None:
            return None
        
        # 图像预处理
        return self.image_processor.process(cv_image)
    
    def _check_sensor_sync(self):
        """
        检查传感器数据是否同步
        
        返回:
            bool: True 表示同步，False 表示不同步
        """
        if not self.enable_sensor_sync:
            return True
        
        timestamps = self.ros_collector.get_timestamps()
        image_stamp = timestamps['image']
        odom_stamp = timestamps['odom']
        
        if image_stamp is None or odom_stamp is None:
            return False
        
        # 计算时间差
        time_diff = abs((image_stamp - odom_stamp).to_sec())
        return time_diff <= self.max_sensor_time_diff
    
    def _on_record_toggle(self):
        """录制切换回调"""
        if self.is_collecting:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _on_command_change(self, command, name):
        """命令变化回调"""
        self.current_command = command
        rospy.loginfo(f"导航命令: {name}")
    
    def _on_emergency_stop(self):
        """紧急停止回调"""
        self.controller.stop()
        self.control_enabled = False
        if self.is_collecting:
            self._stop_recording()
        rospy.logwarn("紧急停止!")
    
    def _on_control_toggle(self, enabled):
        """控制启用/禁用回调"""
        self.control_enabled = enabled
        status = "启用" if enabled else "禁用"
        rospy.loginfo(f"手柄控制已{status}")
    
    def _on_quit(self):
        """退出回调"""
        if self.is_collecting:
            self._stop_recording()
        self.running = False
    
    def _start_recording(self):
        """开始录制"""
        self.data_buffer.clear()
        self._sync_drop_count = 0
        self._total_attempt_count = 0
        self.is_collecting = True
        rospy.loginfo(">>> 开始录制 <<<")
    
    def _auto_save_callback(self, rgb_data, targets_data):
        """自动保存回调 (达到帧数上限时调用)"""
        filepath = self.data_saver.save(rgb_data, targets_data)
        if filepath:
            rospy.loginfo(f"自动保存: {filepath} ({len(rgb_data)} 帧)")
        return filepath
    
    def _stop_recording(self):
        """停止录制并保存剩余数据"""
        self.is_collecting = False
        
        # 统计已自动保存的文件数
        saved_count = len(self.data_buffer.get_saved_files())
        
        # 保存剩余数据
        if self.data_buffer.has_unsaved_data():
            rgb_data, targets_data = self.data_buffer.get_data()
            filepath = self.data_saver.save(rgb_data, targets_data)
            if filepath:
                saved_count += 1
                rospy.loginfo(f"保存: {filepath} ({len(rgb_data)} 帧)")
        
        # 打印统计
        total_frames = self.data_buffer.get_total_frames()
        rospy.loginfo(f"录制完成: 共 {total_frames} 帧, {saved_count} 个文件")
        
        # 打印丢帧统计
        if self._total_attempt_count > 0 and self._sync_drop_count > 0:
            drop_rate = self._sync_drop_count / self._total_attempt_count * 100
            rospy.loginfo(f"同步丢帧: {self._sync_drop_count}/{self._total_attempt_count} ({drop_rate:.1f}%)")
            if drop_rate > 20:
                rospy.logwarn("丢帧率较高，建议增大 MAX_SENSOR_TIME_DIFF 或检查传感器")
        
        self.data_buffer.clear()
    
    def _collect_frame(self):
        """收集一帧数据"""
        self._total_attempt_count += 1
        
        # 检查传感器同步
        self.sync_status = self._check_sensor_sync()
        if not self.sync_status:
            self._sync_drop_count += 1
            return  # 跳过不同步的帧
        
        # 获取处理后的图像
        image = self._get_processed_image()
        if image is None:
            return
        
        # 获取速度和命令
        speed = self.ros_collector.get_speed()
        command = self.current_command
        
        linear_vel, angular_vel = self.controller.get_velocity()
        
        # 构建 targets 向量
        targets = self.control_converter.build_targets_vector(
            linear_vel, angular_vel, speed, command
        )
        
        # 添加到缓冲区 (可能触发自动保存)
        self.data_buffer.add(image, targets)
    
    def run(self):
        """运行收集器"""
        self._print_controls()
        
        rate = rospy.Rate(self.rate_hz)
        
        try:
            while not rospy.is_shutdown() and self.running:
                # 键盘控制需要手动处理按键
                if self.control_type == 'keyboard':
                    key = self.controller.get_key()
                    if not self.controller.process_key(key):
                        break
                
                # 收集数据
                if self.is_collecting:
                    self._collect_frame()
                
                # 获取显示所需数据
                image = self._get_processed_image()
                speed = self.ros_collector.get_speed()
                command = self.current_command
                
                linear_vel, angular_vel = self.controller.get_velocity()
                
                # 获取转换后的 CARLA 格式控制信号 (用于显示)
                carla_ctrl = self.control_converter.to_carla_format(
                    linear_vel, angular_vel, speed
                )
                
                # 获取控制启用状态
                if hasattr(self.controller, 'is_control_enabled'):
                    self.control_enabled = self.controller.is_control_enabled()
                
                # 创建并显示图像
                display = self.visualizer.create_display(
                    image=image,
                    is_collecting=self.is_collecting,
                    current_command=command,
                    frame_count=len(self.data_buffer),
                    speed=speed,
                    linear_vel=linear_vel,
                    angular_vel=angular_vel,
                    sync_status=self.sync_status,
                    episode_count=self.data_saver.get_episode_count(),
                    steer=carla_ctrl['steer'],
                    throttle=carla_ctrl['throttle'],
                    brake=carla_ctrl['brake'],
                    control_enabled=self.control_enabled
                )
                
                key = self.visualizer.show(display)
                
                # ESC 退出 (手柄模式)
                if key == 27:
                    if self.is_collecting:
                        self._stop_recording()
                    break
                
                rate.sleep()
                
        finally:
            self.controller.stop()
            self.controller.cleanup()
            self.ros_collector.cleanup()
            self.visualizer.cleanup()
            rospy.loginfo("收集器已退出")
    
    def _print_controls(self):
        """打印控制说明"""
        print("\n" + "="*50)
        print("TurtleBot 数据收集器")
        print("="*50)
        
        if self.control_type == 'keyboard':
            self.controller.print_controls()
        else:
            print("\n手柄控制:")
            print("  LB/Share - 启用/禁用控制 (防误触)")
            print("  左摇杆 - 移动控制 (需先启用)")
            print("  Start/Back - 开始/停止录制")
            print("  Y/△ - Follow | X/□ - Left")
            print("  B/O - Right | A/X - Straight")
            print("  RB/R1 - 紧急停止")
            print("  ESC - 退出")
            print("\n注意: 启动后需按 LB/Share 启用控制才能移动机器人")
        
        print("="*50 + "\n")


def main():
    """主函数"""
    import argparse
    
    # 获取支持的型号列表
    supported_models = list(RobotConfig.TURTLEBOT_PARAMS.keys())
    
    parser = argparse.ArgumentParser(description='TurtleBot 数据收集器')
    parser.add_argument('--output', '-o', type=str, default=StorageConfig.DEFAULT_OUTPUT_DIR,
                       help=f'输出目录 (默认: {StorageConfig.DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--control', '-c', type=str, default=CollectorConfig.DEFAULT_CONTROL_TYPE,
                       choices=['joystick', 'keyboard'], help='控制类型')
    parser.add_argument('--joystick', '-j', type=str, default=CollectorConfig.DEFAULT_JOYSTICK_TYPE,
                       choices=['xbox', 'ps4'], help='手柄类型')
    parser.add_argument('--model', '-m', type=str, default=RobotConfig.DEFAULT_MODEL,
                       choices=supported_models, 
                       help='TurtleBot 型号')
    parser.add_argument('--rate', '-r', type=int, default=CollectorConfig.DEFAULT_RATE_HZ, 
                       help=f'收集频率 Hz (默认: {CollectorConfig.DEFAULT_RATE_HZ})')
    parser.add_argument('--frames', '-f', type=int, default=StorageConfig.FRAMES_PER_FILE,
                       help=f'每个文件的帧数 (默认: {StorageConfig.FRAMES_PER_FILE})')
    parser.add_argument('--no-sync', action='store_true',
                       help='禁用传感器时间戳同步')
    
    args = parser.parse_args()
    
    try:
        collector = TurtleBotCollector(
            output_dir=args.output,
            control_type=args.control,
            joystick_type=args.joystick,
            turtlebot_model=args.model,
            rate_hz=args.rate,
            frames_per_file=args.frames,
            enable_sensor_sync=not args.no_sync
        )
        collector.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
