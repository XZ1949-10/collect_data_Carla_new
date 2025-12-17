#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 推理主模块
整合所有子模块，提供统一的推理接口
'''

import time
import torch
import rospy

from .config.inference_config import (
    CONTROL_RATE_HZ, DEFAULT_COMMAND, COMMAND_NAMES, SPEED_NORMALIZATION_KMH
)
from .model.model_loader import ModelLoader
from .model.model_predictor import ModelPredictor
from .processing.image_processor import ImageProcessor
from .processing.control_converter import ControlConverter
from .ros_interface.ros_sensor import ROSSensor
from .ros_interface.ros_controller import ROSController
from .control.command_controller import CommandController


class TurtleBotInference:
    """TurtleBot 推理主类"""
    
    def __init__(self,
                 model_path,
                 turtlebot_model='burger',
                 gpu_id=0,
                 joystick_type='xbox',
                 control_rate=CONTROL_RATE_HZ,
                 image_topic=None,
                 odom_topic=None,
                 cmd_vel_topic=None,
                 joy_topic=None,
                 net_structure=2):
        """
        初始化推理器
        
        参数:
            model_path (str): 模型权重路径
            turtlebot_model (str): TurtleBot 型号
            gpu_id (int): GPU ID，-1 表示使用 CPU
            joystick_type (str): 手柄类型 ('xbox' 或 'ps4')
            control_rate (int): 控制频率 (Hz)
            image_topic (str): 图像话题
            odom_topic (str): 里程计话题
            cmd_vel_topic (str): 速度命令话题
            joy_topic (str): 手柄话题
            net_structure (int): 网络结构类型
        """
        self.model_path = model_path
        self.turtlebot_model = turtlebot_model
        self.control_rate = control_rate
        self.net_structure = net_structure
        
        # 设置设备
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化各模块（延迟加载）
        self.model_loader = ModelLoader(model_path, self.device, net_structure)
        self.model = None
        self.predictor = None
        
        self.image_processor = ImageProcessor(self.device)
        self.control_converter = ControlConverter(model=turtlebot_model)
        
        self.sensor = ROSSensor(image_topic, odom_topic)
        self.controller = ROSController(cmd_vel_topic)
        self.command_controller = CommandController(joystick_type, joy_topic)
        
        # 统计信息
        self.frame_count = 0
        self.total_inference_time = 0.0
        
    def load_model(self, network_module=None):
        """加载模型"""
        self.model = self.model_loader.load(network_module)
        self.predictor = ModelPredictor(self.model, self.device)
        print("✅ 模型加载完成")
        
    def setup_ros(self):
        """设置 ROS 接口"""
        self.sensor.setup()
        self.controller.setup()
        self.command_controller.setup()
        print("✅ ROS 接口设置完成")
        
        # 等待数据
        print("等待传感器数据...")
        timeout = 10.0
        start = time.time()
        while not self.sensor.has_data() and time.time() - start < timeout:
            rospy.sleep(0.1)
            
        if not self.sensor.has_data():
            raise RuntimeError("超时：未收到传感器数据")
        print("✅ 传感器数据就绪")

    def run(self, duration=None, print_interval=10):
        """
        运行推理循环
        
        参数:
            duration (float): 运行时长（秒），None 表示无限运行
            print_interval (int): 打印间隔（帧数）
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
            
        print(f"\n{'='*60}")
        print("开始推理")
        print(f"控制频率: {self.control_rate} Hz")
        print(f"TurtleBot 型号: {self.turtlebot_model}")
        print(f"{'='*60}\n")
        
        rate = rospy.Rate(self.control_rate)
        start_time = time.time()
        
        try:
            while not rospy.is_shutdown() and self.command_controller.is_running():
                # 检查时长
                if duration is not None and time.time() - start_time > duration:
                    print("达到运行时长，停止推理")
                    break
                    
                # 获取传感器数据
                image = self.sensor.get_image()
                if image is None:
                    rate.sleep()
                    continue
                    
                speed_data = self.sensor.get_speed()
                current_command = self.command_controller.get_command()
                
                # 预处理图像
                img_tensor = self.image_processor.preprocess(image)
                
                # 计算归一化速度
                speed_kmh = abs(speed_data['linear_vel']) * 3.6
                speed_normalized = speed_kmh / SPEED_NORMALIZATION_KMH
                
                # 模型预测 (直接输出 linear_vel, angular_vel)
                result = self.predictor.predict(img_tensor, speed_normalized, current_command)
                
                # 直接发送控制命令 (无需格式转换)
                self.controller.send_velocity(
                    result['linear_vel'], result['angular_vel']
                )
                
                # 统计
                self.frame_count += 1
                self.total_inference_time += result['inference_time']
                
                # 打印状态
                if self.frame_count % print_interval == 0:
                    self._print_status(start_time, speed_data, result, current_command)
                    
                rate.sleep()
                
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            self.controller.stop()
            
    def _print_status(self, start_time, speed_data, result, current_command):
        """打印状态信息"""
        elapsed = time.time() - start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        speed_kmh = abs(speed_data['linear_vel']) * 3.6
        cmd_name = COMMAND_NAMES.get(current_command, 'Unknown')
        
        print(f"[{elapsed:.1f}s] "
              f"Cmd: {cmd_name:8s} | "
              f"Spd: {speed_kmh:4.1f} km/h | "
              f"Str: {result['steer']:+.3f} | "
              f"Thr: {result['throttle']:.3f} | "
              f"Brk: {result['brake']:.3f} | "
              f"FPS: {fps:.1f}")
              
    def print_statistics(self):
        """打印统计信息"""
        if self.frame_count == 0:
            return
            
        print(f"\n{'='*60}")
        print("推理统计")
        print(f"{'='*60}")
        print(f"总帧数: {self.frame_count}")
        print(f"平均推理时间: {self.total_inference_time/self.frame_count*1000:.2f} ms")
        print(f"{'='*60}\n")
        
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        self.controller.cleanup()
        self.sensor.cleanup()
        self.command_controller.cleanup()
        print("清理完成")
