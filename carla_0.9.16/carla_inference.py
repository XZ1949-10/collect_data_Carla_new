#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant
日期: 2025-11-25
说明: Carla自动驾驶模型实时推理脚本（模块化版本）
      从Carla实时获取图像和速度，使用训练好的模型预测控制信号，并控制车辆
'''

import os
import sys
import time
import argparse

# 设置标准输出编码为UTF-8，避免Windows下的编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
import cv2
import carla

# 导入项目模块
from carla_config import *
from carla_sensors import SensorManager
from carla_visualizer import CarlaVisualizer

# 可视化模式常量
VIS_MODE_SPECTATOR = 'spectator'  # Spectator跟随模式（在CARLA窗口中显示）
VIS_MODE_OPENCV = 'opencv'        # OpenCV独立窗口模式（旧模式）
from navigation_planner_adapter import NavigationPlannerAdapter
from carla_model_loader import ModelLoader
from carla_image_processor import ImageProcessor
from carla_vehicle_controller import VehicleController
from carla_model_predictor import ModelPredictor
from carla_vehicle_spawner import VehicleSpawner
from carla_npc_manager import NPCManager, NPCConfig

# 可解释性模块（可选）
try:
    from carla_interpretability import (
        InterpretabilityVisualizer, 
        GradCAM, 
        BrakeAnalyzer,
        create_interpretability_visualizer
    )
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    print("⚠️ 可解释性模块未找到，--interpret 功能不可用")


class CarlaInference:
    """
    Carla自动驾驶推理类（模块化版本）
    
    核心功能：
    1. 连接到Carla服务器
    2. 加载训练好的模型
    3. 实时获取传感器数据
    4. 使用模型预测控制信号
    5. 控制车辆行驶
    """
    
    def __init__(self, 
                 model_path,
                 host='localhost',
                 port=2000,
                 town='Town01',
                 gpu_id=0,
                 enable_post_processing=False,
                 post_processor_config=None,
                 enable_image_crop=True,
                 visualization_mode='spectator',
                 npc_config=None,
                 weather='ClearNoon',
                 enable_interpretability=False,
                 interpret_save_dir=None,
                 interpret_save_interval=10,
                 interpret_device='gpu',
                 interpret_full_analysis=True,
                 interpret_row1_layer=-3,
                 interpret_row2_layers=None,
                 interpret_ig_steps=30):
        """
        初始化推理器
        
        参数:
            model_path (str): 训练好的模型权重路径
            host (str): Carla服务器地址
            port (int): Carla服务器端口
            town (str): 地图名称
            gpu_id (int): GPU ID，-1表示使用CPU
            enable_post_processing (bool): 是否启用后处理
            post_processor_config (dict): 后处理器配置
            enable_image_crop (bool): 是否启用图像裁剪（去除天空和引擎盖）
            visualization_mode (str): 可视化模式
                - 'spectator': Spectator跟随模式（在CARLA UE4窗口中第三人称跟随）
                - 'opencv': OpenCV独立窗口模式（旧模式，小弹窗）
            npc_config (NPCConfig): NPC配置，None表示不生成NPC
            weather (str): 天气预设名称，支持:
                - ClearNoon, ClearSunset
                - CloudyNoon, CloudySunset
                - WetNoon, WetSunset
                - WetCloudyNoon, WetCloudySunset
                - HardRainNoon, HardRainSunset
                - SoftRainNoon, SoftRainSunset
            interpret_row1_layer (int): 第一行热力图使用的卷积层索引
            interpret_row2_layers (list): 第二行多层级热力图使用的卷积层索引列表
            interpret_ig_steps (int): 积分梯度的积分步数
        """
        # Carla连接参数
        self.host = host
        self.port = port
        self.town = town
        self.weather = weather
        
        # 设备配置
        self.gpu_id = gpu_id
        self.device = torch.device(
            f'cuda:{gpu_id}' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu'
        )
        
        # Carla对象
        self.client = None
        self.world = None
        self.vehicle = None
        
        # 功能模块
        self.model_loader = ModelLoader(model_path, self.device)
        # 图像处理器（与数据收集保持一致的裁剪参数）
        # 裁剪区域：[90:485, :] 去除天空和车头
        self.image_processor = ImageProcessor(
            self.device,
            enable_crop=enable_image_crop,
            crop_top=90,
            crop_bottom=485
        )
        self.vehicle_controller = VehicleController()
        self.model_predictor = None  # 在加载模型后初始化
        self.vehicle_spawner = None  # 在连接Carla后初始化
        
        # 后处理器配置
        self.enable_post_processing = enable_post_processing
        self.post_processor_config = post_processor_config
        
        # 可视化模式
        self.visualization_mode = visualization_mode
        
        # NPC配置
        self.npc_config = npc_config
        self.npc_manager = None
        
        # 组件模块
        self.sensor_manager = None
        self.navigation_planner = None
        self.visualizer = CarlaVisualizer(mode=visualization_mode)
        
        # 状态
        self.current_command = 2  # 默认命令：2=跟车
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        # 可解释性模块
        self.enable_interpretability = enable_interpretability and INTERPRETABILITY_AVAILABLE
        self.interpret_save_dir = interpret_save_dir
        self.interpret_save_interval = interpret_save_interval  # 仪表板保存频率（每N帧保存一次，0表示不自动保存）
        self.interpret_device = interpret_device  # 可解释性分析设备: 'gpu' 或 'cpu'
        self.interpret_full_analysis = interpret_full_analysis  # 是否启用完整分析
        self.interpret_row1_layer = interpret_row1_layer  # 第一行热力图卷积层索引
        self.interpret_row2_layers = interpret_row2_layers if interpret_row2_layers else [-1, -3, -5]  # 第二行多层级热力图卷积层索引
        self.interpret_ig_steps = interpret_ig_steps  # 积分梯度步数
        self.interp_visualizer = None
        self.grad_cam = None
        self.brake_analyzer = None
        
        # 设置可解释性分析的设备
        if interpret_device == 'cpu':
            self.interp_compute_device = torch.device('cpu')
        else:
            self.interp_compute_device = self.device  # 与模型推理使用同一设备
        
        print(f"初始化推理器 - 设备: {self.device}")
        if self.enable_interpretability:
            print(f"✅ 可解释性可视化已启用")
            print(f"   - 分析设备: {self.interp_compute_device}")
            print(f"   - 完整分析: {'是' if interpret_full_analysis else '否 (仅Grad-CAM)'}")
            print(f"   - 第一行热力图层索引: {self.interpret_row1_layer}")
            print(f"   - 第二行多层索引: {self.interpret_row2_layers}")
            print(f"   - 积分梯度步数: {self.interpret_ig_steps}")
        
    def load_model(self, net_structure=2):
        """加载训练好的模型"""
        self.model_loader.net_structure = net_structure
        model = self.model_loader.load()
        self.model_predictor = ModelPredictor(
            model, 
            self.device,
            enable_post_processing=self.enable_post_processing,
            post_processor_config=self.post_processor_config
        )
        
        # 初始化可解释性工具（学术严谨版）
        if self.enable_interpretability:
            # 使用新的综合分析器，支持选择计算设备和热力图层配置
            self.interp_visualizer = create_interpretability_visualizer(
                model, self.interp_compute_device, self.interpret_save_dir,
                full_analysis=self.interpret_full_analysis,  # 根据参数决定是否启用完整分析
                grad_cam_layer_index=self.interpret_row1_layer,  # 第一行热力图层索引
                multi_layer_indices=self.interpret_row2_layers,  # 第二行多层级热力图层索引
                ig_steps=self.interpret_ig_steps  # 积分梯度步数
            )
            # 保留旧接口兼容性
            self.grad_cam = self.interp_visualizer.grad_cam
            self.brake_analyzer = self.interp_visualizer.brake_analyzer
            
            if self.interpret_full_analysis:
                print("✅ 学术严谨版可解释性分析器已初始化")
                print("   包含: Grad-CAM, 遮挡敏感性, 积分梯度, 删除/插入曲线")
            else:
                print("✅ 轻量级可解释性分析器已初始化")
                print("   包含: Grad-CAM (高计算量方法已禁用)")
            print(f"   计算设备: {self.interp_compute_device}")
        
    def connect_carla(self):
        """连接到Carla服务器"""
        print(f"正在连接到Carla服务器 {self.host}:{self.port}...")
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        
        print(f"正在加载地图 {self.town}...")
        self.world = self.client.load_world(self.town)
        
        # 设置天气
        self._set_weather()
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SYNC_MODE_DELTA_SECONDS
        self.world.apply_settings(settings)
        
        # 初始化车辆生成器
        self.vehicle_spawner = VehicleSpawner(self.world)
        
        # 创建导航规划器
        print("正在初始化导航规划器...")
        self.navigation_planner = NavigationPlannerAdapter(
            self.world, 
            sampling_resolution=ROUTE_SAMPLING_RESOLUTION
        )
        
        # 生成NPC（如果配置了）
        if self.npc_config is not None:
            self._spawn_npcs()
        
        print("成功连接到Carla服务器！")
    
    def _set_weather(self):
        """设置天气"""
        if self.weather is None:
            return
            
        if hasattr(carla.WeatherParameters, self.weather):
            weather_params = getattr(carla.WeatherParameters, self.weather)
            self.world.set_weather(weather_params)
            print(f"天气设置为: {self.weather}")
        else:
            print(f"⚠️ 未知的天气预设: {self.weather}，使用默认天气")
            print(f"   支持的预设: ClearNoon, ClearSunset, CloudyNoon, CloudySunset, "
                  f"WetNoon, WetSunset, WetCloudyNoon, WetCloudySunset, "
                  f"HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset")
    
    def _spawn_npcs(self):
        """生成NPC车辆和行人"""
        if self.npc_config is None:
            return
        
        print("\n正在生成NPC...")
        self.npc_manager = NPCManager(self.client, self.world)
        stats = self.npc_manager.spawn_all(self.npc_config)
        print(f"NPC生成完成: {stats['vehicles_spawned']} 辆车, {stats['walkers_spawned']} 个行人\n")
        
    def spawn_vehicle(self, vehicle_filter='vehicle.tesla.model3', 
                      spawn_index=None, destination_index=None, max_retries=5):
        """
        生成车辆并设置路线
        
        参数:
            vehicle_filter (str): 车辆类型
            spawn_index (int): 起点索引，None表示随机
            destination_index (int): 终点索引，None表示随机
            max_retries (int): 最大重试次数
        """
        # 生成车辆
        self.vehicle = self.vehicle_spawner.spawn(vehicle_filter, spawn_index)
        
        # 创建传感器管理器
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        
        # 等待传感器初始化
        for _ in range(3):
            self.world.tick()
        
        # 设置目的地
        self._setup_destination(destination_index)
        
        # 如果是spectator模式，设置跟随
        if self.visualization_mode == VIS_MODE_SPECTATOR:
            self.visualizer.setup_spectator_mode(self.world, self.vehicle)
            # 初始化路网数据（用于导航地图显示周围道路）
            self.visualizer.init_road_network(self.world)
        
        return True
    
    def _setup_destination(self, destination_index):
        """设置目的地"""
        print("\n正在规划路线...")
        spawn_points = self.world.get_map().get_spawn_points()
        
        if destination_index is not None and 0 <= destination_index < len(spawn_points):
            destination = spawn_points[destination_index].location
            print(f"使用指定终点索引: {destination_index}")
            if not self.navigation_planner.set_destination(self.vehicle, destination):
                print("⚠️ 警告：无法规划到指定终点，将使用随机目的地")
                self.navigation_planner.set_random_destination(self.vehicle)
        else:
            print("使用随机终点")
            if not self.navigation_planner.set_random_destination(self.vehicle):
                print("⚠️ 警告：无法规划路线，将使用默认命令（跟车）")
        
        # 将路线数据传递给可视化器（用于路线图显示）
        self._update_visualizer_route()
        print()
    
    def _update_visualizer_route(self):
        """更新可视化器的路线数据"""
        if self.navigation_planner is not None and hasattr(self.navigation_planner, '_route'):
            route = self.navigation_planner._route
            if route:
                self.visualizer.set_route(route)
                print(f"✅ 路线图已更新（{len(route)} 个路点）")
        
    def setup_sensors(self):
        """设置所有传感器"""
        self.sensor_manager.setup_camera()
        
    def run_inference(self, duration=60, visualize=True, auto_replan=True):
        """
        运行实时推理
        
        参数:
            duration (int): 运行时长（秒），-1表示无限运行
            visualize (bool): 是否显示可视化窗口
            auto_replan (bool): 到达目的地后是否自动重新规划路线
        """
        print(f"\n{'='*60}")
        print("开始实时推理控制")
        print(f"{'='*60}")
        print(f"运行时长: {'无限' if duration < 0 else f'{duration}秒'}")
        print(f"可视化: {'开启' if visualize else '关闭'}")
        if visualize:
            mode_desc = "Spectator跟随模式（CARLA窗口第三人称视角）" if self.visualization_mode == VIS_MODE_SPECTATOR else "OpenCV独立窗口模式"
            print(f"可视化模式: {mode_desc}")
        print(f"自动重新规划: {'开启' if auto_replan else '关闭'}")
        print(f"目标帧率: {1.0/SYNC_MODE_DELTA_SECONDS:.0f} FPS (与模拟时间同步)")
        print("模型输出: 直接控制（无后处理）")
        if self.enable_interpretability:
            print("🔍 可解释性可视化: 已启用 (按 'i' 切换显示)")
        print(f"{'='*60}\n")
        
        # 可解释性窗口（学术严谨版）
        show_interpretability = self.enable_interpretability
        if self.enable_interpretability:
            cv2.namedWindow('Model Interpretability', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Model Interpretability', 2560, 1440)  # 2K分辨率
        
        # 等待摄像头数据
        print("等待摄像头数据...")
        while not self.sensor_manager.has_image():
            self.world.tick()
            time.sleep(0.01)
        print("摄像头数据就绪！\n")
        
        start_time = time.time()
        self.visualizer.set_start_time(start_time)
        self.frame_count = 0
        
        # 帧率控制：确保模拟时间与现实时间同步
        target_frame_time = SYNC_MODE_DELTA_SECONDS  # 每帧目标耗时（秒）
        
        try:
            while True:
                frame_start_time = time.time()  # 记录帧开始时间
                
                # 检查超时
                if duration > 0 and time.time() - start_time > duration:
                    print(f"\n已运行 {duration} 秒，停止推理")
                    break
                
                # 推进模拟
                self.world.tick()
                
                if not self.sensor_manager.has_image():
                    continue
                
                # 【重要】调用 run_step 更新 LocalPlanner 状态
                # 这样 target_road_option 才会正确更新
                self.navigation_planner.run_step()
                
                # 获取导航命令（现在使用与数据收集一致的方式）
                self.current_command = self.navigation_planner.get_navigation_command(self.vehicle)
                
                # 调试：打印命令信息
                if self.frame_count % PRINT_INTERVAL_FRAMES == 0:
                    route_info = self.navigation_planner.get_route_info(self.vehicle)
                    print(f"[DEBUG] Cmd: {self.current_command} "
                          f"({COMMAND_NAMES_EN.get(self.current_command, 'Unknown')}), "
                          f"Branch: {self.current_command - 2}")
                
                # 检查是否到达
                if self.navigation_planner.is_route_completed(self.vehicle):
                    print("\n🎯 已到达目的地！")
                    if auto_replan:
                        print("正在重新规划路线...")
                        if self.navigation_planner.set_random_destination(self.vehicle):
                            # 更新可视化器的路线数据
                            self._update_visualizer_route()
                            print("新路线规划成功，继续行驶\n")
                        else:
                            print("⚠️ 无法规划新路线，停止推理\n")
                            break
                    else:
                        print("停止推理\n")
                        break
                
                # 获取数据
                current_image = self.sensor_manager.get_latest_image()
                # 注意：get_speed_normalized 默认已使用25 KM/H，与训练配置一致
                current_speed = self.vehicle_controller.get_speed_normalized(
                    self.vehicle, SPEED_NORMALIZATION_MPS
                )
                
                # 预处理图像
                img_tensor = self.image_processor.preprocess(current_image)
                
                # 预测控制
                control_result = self.model_predictor.predict(
                    img_tensor, current_speed, self.current_command
                )
                
                # 累计推理时间
                self.total_inference_time += control_result['inference_time']
                
                # 调试：打印所有分支的预测值
                if self.frame_count % PRINT_INTERVAL_FRAMES == 0:
                    self._debug_print_all_branches(control_result)
                
                # 应用控制
                self.vehicle_controller.apply_control(
                    self.vehicle,
                    control_result['steer'],
                    control_result['throttle'],
                    control_result['brake']
                )
                
                # 更新计数
                self.frame_count += 1
                
                # 打印信息
                if self.frame_count % PRINT_INTERVAL_FRAMES == 0:
                    self._print_status(start_time, current_speed, control_result)
                
                # 获取模型实际看到的图像（裁剪+缩放后的 200x88）
                model_input_image = self.image_processor.get_processed_image(current_image)
                
                # 可视化
                if visualize:
                    route_info = self.navigation_planner.get_route_info(self.vehicle)
                    
                    # 获取车辆位置和朝向（用于路线图）
                    vehicle_transform = self.vehicle.get_transform()
                    vehicle_location = (vehicle_transform.location.x, vehicle_transform.location.y)
                    vehicle_yaw = vehicle_transform.rotation.yaw
                    current_waypoint_index = self.navigation_planner._current_waypoint_index
                    
                    self.visualizer.visualize(
                        model_input_image, 
                        control_result, 
                        current_speed, 
                        route_info,
                        self.frame_count,
                        vehicle_location=vehicle_location,
                        vehicle_yaw=vehicle_yaw,
                        current_waypoint_index=current_waypoint_index
                    )
                
                # 可解释性可视化
                if self.enable_interpretability and show_interpretability:
                    # 将张量移到可解释性分析设备上
                    img_tensor_interp = img_tensor.to(self.interp_compute_device)
                    speed_tensor_interp = torch.FloatTensor([[current_speed]]).to(self.interp_compute_device)
                    interp_dashboard = self._create_interpretability_dashboard(
                        img_tensor_interp, speed_tensor_interp, model_input_image, 
                        control_result, current_speed
                    )
                    cv2.imshow('Model Interpretability', interp_dashboard)
                    
                    # 自动保存仪表板（按设定频率保存，0表示不自动保存）
                    if (self.interpret_save_dir is not None and 
                        self.interpret_save_interval > 0 and 
                        self.frame_count % self.interpret_save_interval == 0):
                        save_path = os.path.join(self.interpret_save_dir, f"dashboard_{self.frame_count:06d}.png")
                        cv2.imwrite(save_path, interp_dashboard)
                
                # 键盘处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('i') and self.enable_interpretability:
                    show_interpretability = not show_interpretability
                    if show_interpretability:
                        print("🔍 可解释性窗口: 显示")
                    else:
                        print("🔍 可解释性窗口: 隐藏")
                        cv2.destroyWindow('Model Interpretability')
                        cv2.namedWindow('Model Interpretability', cv2.WINDOW_NORMAL)
                elif key == ord('s') and self.enable_interpretability:
                    # 手动保存当前帧
                    self._save_interpretability_frame(model_input_image, control_result)
                elif key == ord('p') and self.enable_interpretability:
                    # 打印刹车统计
                    self._print_brake_statistics()
                
                # 帧率控制：等待到目标帧时间，确保模拟时间与现实时间1:1同步
                frame_elapsed = time.time() - frame_start_time
                sleep_time = target_frame_time - frame_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n用户中断推理")
            
        finally:
            if visualize:
                self.visualizer.close()
            if self.enable_interpretability:
                cv2.destroyAllWindows()
                self._print_brake_statistics()
                
    def _create_interpretability_dashboard(self, img_tensor, speed_tensor, 
                                            original_image, control_result, current_speed):
        """
        创建可解释性仪表板（学术严谨版）
        
        使用新的综合分析器，包含：
        - Grad-CAM 热力图（定性）
        - 遮挡敏感性分析（定量）
        - 积分梯度（定量）
        - 删除/插入曲线（定量）
        
        只显示当前选中分支的可视化结果。
        """
        # 添加速度信息到control_result
        control_result_with_speed = control_result.copy()
        control_result_with_speed['speed_normalized'] = current_speed
        
        # 使用综合分析器分析帧
        if self.interp_visualizer is not None:
            analysis_results = self.interp_visualizer.analyze_frame(
                img_tensor, speed_tensor, original_image,
                control_result_with_speed, self.current_command
            )
            
            # 获取红绿灯信息
            traffic_light_info = self._get_traffic_light_info()
            
            # 获取所有分支预测
            all_branch_predictions = self.model_predictor.get_all_branch_predictions()
            
            # 渲染仪表板
            dashboard = self.interp_visualizer.render_dashboard(
                original_image, analysis_results, control_result,
                self.current_command, traffic_light_info, all_branch_predictions
            )
            
            # 嵌入历史曲线图到 Control History 面板
            # 布局计算 (2560x1440):
            # row6_y = 60 + 150 + 8 + 130 + 8 + 130 + 8 + 130 + 8 + 190 + 8 = 830
            # row6_h = (1440 - 45 - 5) - 830 = 560 (动态计算，最小200)
            # 面板标题高度约25px，所以内容从 row6_y + 25 开始
            # History面板宽度 = (2560 - 24 - 20) * 0.40 ≈ 1006
            if self.brake_analyzer is not None:
                # 计算实际可用的高度
                row6_y = 830
                footer_y = 1440 - 45
                row6_h = max(footer_y - 5 - row6_y, 200)
                
                history_w = 985  # 面板宽度减去边距
                history_h = row6_h - 30  # 减去标题和边距
                history_x = 18  # MARGIN + 6
                history_y = row6_y + 25  # 标题高度
                
                history_plot = self.brake_analyzer.plot_history(width=history_w, height=history_h)
                # 确保不越界
                y_end = min(history_y + history_h, footer_y - 5)
                x_end = min(history_x + history_w, dashboard.shape[1])
                h_actual = y_end - history_y
                w_actual = x_end - history_x
                if h_actual > 0 and w_actual > 0:
                    dashboard[history_y:y_end, history_x:x_end] = history_plot[:h_actual, :w_actual]
            
            return dashboard
        else:
            # 回退到简单仪表板
            return self._create_simple_dashboard(original_image, control_result)
    
    def _get_traffic_light_info(self):
        """获取最近红绿灯的信息"""
        if self.vehicle is None or self.world is None:
            return None
        
        try:
            vehicle_location = self.vehicle.get_location()
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
            
            nearest_tl = None
            min_distance = float('inf')
            
            for tl in traffic_lights:
                tl_location = tl.get_location()
                distance = vehicle_location.distance(tl_location)
                if distance < min_distance and distance < 50:  # 50米范围内
                    min_distance = distance
                    nearest_tl = tl
            
            if nearest_tl is not None:
                state_map = {
                    carla.TrafficLightState.Red: 'Red',
                    carla.TrafficLightState.Yellow: 'Yellow',
                    carla.TrafficLightState.Green: 'Green',
                }
                state = state_map.get(nearest_tl.get_state(), 'Unknown')
                return {
                    'state': state,
                    'distance': min_distance
                }
        except:
            pass
        
        return None
    
    def _create_simple_dashboard(self, original_image, control_result):
        """创建简单仪表板（回退方案）"""
        dash_width, dash_height = 800, 400
        dashboard = np.zeros((dash_height, dash_width, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 32)
        
        cv2.putText(dashboard, "Simple Dashboard (Interpretability module not fully loaded)", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # 显示原图
        orig_resized = cv2.resize(original_image, (300, 132))
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
        dashboard[50:182, 20:320] = orig_bgr
        
        # 显示控制值
        cv2.putText(dashboard, f"Steer: {control_result['steer']:+.3f}", (350, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(dashboard, f"Throttle: {control_result['throttle']:.3f}", (350, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 230, 100), 1, cv2.LINE_AA)
        cv2.putText(dashboard, f"Brake: {control_result['brake']:.3f}", (350, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1, cv2.LINE_AA)
        
        return dashboard
    
    def _draw_panel(self, img, x, y, w, h, title, title_color=(220, 220, 220)):
        """绘制面板（更清晰的边框和标题）"""
        # 面板背景
        cv2.rectangle(img, (x, y), (x+w, y+h), (42, 42, 48), -1)
        # 边框（更粗）
        cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 80), 2)
        # 标题（更大字体）
        cv2.putText(img, title, (x+8, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_color, 1, cv2.LINE_AA)
    
    def _draw_control_bar(self, img, x, y, w, h, value, label, color, warning_threshold=None):
        """绘制控制条（更大更清晰）"""
        # 背景
        cv2.rectangle(img, (x, y), (x+w, y+h), (55, 55, 60), -1)
        # 值条
        bar_w = int(w * min(1.0, max(0.0, value)))
        if warning_threshold and value > warning_threshold:
            bar_color = (60, 60, 255)  # 警告色（红）
        else:
            bar_color = color
        if bar_w > 0:
            cv2.rectangle(img, (x, y), (x+bar_w, y+h), bar_color, -1)
        # 边框
        cv2.rectangle(img, (x, y), (x+w, y+h), (90, 90, 100), 2)
        # 标签和值（放在条的右侧，留足够空间）
        text = f"{label}: {value:.3f}"
        cv2.putText(img, text, (x+w+12, y+h-6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    
    def _draw_steer_bar(self, img, x, y, w, h, value, label):
        """绘制转向条（中心对称，更清晰）"""
        cv2.rectangle(img, (x, y), (x+w, y+h), (55, 55, 60), -1)
        center = x + w // 2
        steer_x = center + int((w//2) * value)
        cv2.rectangle(img, (min(center, steer_x), y), (max(center, steer_x), y+h), (0, 230, 230), -1)
        cv2.line(img, (center, y), (center, y+h), (120, 120, 130), 3)
        cv2.rectangle(img, (x, y), (x+w, y+h), (90, 90, 100), 2)
        # 标签和值（放在条的右侧，留足够空间）
        text = f"{label}: {value:+.3f}"
        cv2.putText(img, text, (x+w+12, y+h-6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 230), 2, cv2.LINE_AA)
    
    def _draw_stat_item(self, img, x, y, label, value, warn=False):
        """绘制统计项（更大字体）"""
        color = (120, 120, 255) if warn else (220, 220, 220)
        cv2.putText(img, f"{label}:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(img, str(value), (x + 180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    
    def _draw_traffic_light_indicator(self, img, x, y):
        """
        绘制红绿灯指示器
        
        说明：显示 CARLA 仿真环境中车辆附近最近的红绿灯状态
        用途：帮助判断模型在红灯时是否正确输出刹车信号
        """
        if self.vehicle is None or self.world is None:
            cv2.putText(img, "N/A", (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
            return
        
        vehicle_loc = self.vehicle.get_location()
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        nearest_tl, nearest_dist = None, float('inf')
        for tl in traffic_lights:
            dist = vehicle_loc.distance(tl.get_location())
            if dist < nearest_dist:
                nearest_dist, nearest_tl = dist, tl
        
        if nearest_tl and nearest_dist < 50:
            state = str(nearest_tl.get_state()).split('.')[-1]
            
            # 绘制红绿灯图标（更大）
            light_x, light_y = x + 45, y + 50
            cv2.rectangle(img, (light_x-20, light_y-45), (light_x+20, light_y+45), (25, 25, 25), -1)
            cv2.rectangle(img, (light_x-20, light_y-45), (light_x+20, light_y+45), (90, 90, 90), 2)
            
            # 三个灯（更大）
            colors = [(60, 60, 60), (60, 60, 60), (60, 60, 60)]
            if 'Red' in state:
                colors[0] = (0, 0, 255)
            elif 'Yellow' in state:
                colors[1] = (0, 220, 255)
            else:
                colors[2] = (0, 255, 0)
            
            cv2.circle(img, (light_x, light_y-28), 14, colors[0], -1)
            cv2.circle(img, (light_x, light_y), 14, colors[1], -1)
            cv2.circle(img, (light_x, light_y+28), 14, colors[2], -1)
            
            # 状态文字（更大）
            cv2.putText(img, state, (light_x + 35, light_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(img, f"Distance: {nearest_dist:.0f}m", (light_x + 35, light_y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
            
            # 红灯警告（更醒目）
            if 'Red' in state and nearest_dist < 30:
                all_preds = self.model_predictor.get_all_branch_predictions()
                if all_preds is not None:
                    max_brake = max(all_preds[2], all_preds[5], all_preds[8], all_preds[11])
                    if max_brake < 0.3:
                        cv2.rectangle(img, (x, y + 95), (x + 200, y + 120), (0, 0, 180), -1)
                        cv2.putText(img, "WARNING: LOW BRAKE!", (x + 10, y + 113), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "No traffic light", (x, y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(img, "within 50m", (x, y + 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    
    def _save_interpretability_frame(self, original_image, control_result):
        """保存可解释性帧"""
        if self.interpret_save_dir is None:
            self.interpret_save_dir = './interpret_output'
        
        import os
        os.makedirs(self.interpret_save_dir, exist_ok=True)
        
        filename = f"interp_{self.frame_count:06d}.png"
        filepath = os.path.join(self.interpret_save_dir, filename)
        
        # 保存原图
        orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, orig_bgr)
        
        print(f"✅ 已保存: {filepath}")
    
    def _print_brake_statistics(self):
        """打印刹车统计和可解释性指标"""
        if self.brake_analyzer is None:
            print("刹车分析器未初始化")
            return
        
        stats = self.brake_analyzer.get_statistics()
        print(f"\n{'='*60}")
        print("刹车行为统计")
        print(f"{'='*60}")
        print(f"总帧数: {stats.get('total_frames', 0)}")
        print(f"刹车帧占比 (>0.1): {stats.get('brake_ratio', 0)*100:.1f}%")
        print(f"急刹车帧占比 (>0.5): {stats.get('hard_brake_ratio', 0)*100:.1f}%")
        print(f"平均刹车值: {stats.get('avg_brake', 0):.3f}")
        print(f"最大刹车值: {stats.get('max_brake', 0):.3f}")
        
        # 打印可解释性定量指标
        if self.interp_visualizer is not None:
            print(f"\n{'='*60}")
            print("可解释性定量指标汇总 (Academic Metrics)")
            print(f"{'='*60}")
            summary = self.interp_visualizer.get_metrics_summary()
            
            if summary:
                occ = summary.get('occlusion_sensitivity', {})
                ig = summary.get('integrated_gradients', {})
                di = summary.get('deletion_insertion', {})
                
                print(f"分析帧数: {summary.get('total_frames_analyzed', 0)}")
                print(f"\n遮挡敏感性 (Occlusion Sensitivity):")
                print(f"  平均值: {occ.get('mean', 0):.4f}")
                print(f"  标准差: {occ.get('std', 0):.4f}")
                
                print(f"\n积分梯度 (Integrated Gradients):")
                print(f"  完整性误差: {ig.get('mean_completeness_error', 0):.4f}")
                
                print(f"\n删除/插入曲线 (Deletion/Insertion):")
                print(f"  删除AUC (越低越好): {di.get('mean_deletion_auc', 0):.4f}")
                print(f"  插入AUC (越高越好): {di.get('mean_insertion_auc', 0):.4f}")
                print(f"  综合得分: {di.get('mean_combined_score', 0):+.4f}")
            
            # 导出指标到文件
            if self.interpret_save_dir:
                metrics_path = os.path.join(self.interpret_save_dir, 'metrics.json')
                self.interp_visualizer.save_metrics(metrics_path)
                print(f"\n📊 指标已导出到: {metrics_path}")
        
        print(f"{'='*60}\n")

    def _debug_print_all_branches(self, control_result):
        """调试：打印所有分支的预测值"""
        all_predictions = self.model_predictor.get_all_branch_predictions()
        if all_predictions is None:
            return
            
        print(f"\n{'='*70}")
        print(f"[调试] 所有分支预测值 (帧 {self.frame_count})")
        print(f"{'='*70}")
        print(f"当前命令: {self.current_command} ({COMMAND_NAMES_EN.get(self.current_command, 'Unknown')})")
        print(f"当前分支索引: {self.current_command - 2}")
        print(f"\n{'分支':<12} {'命令':<10} {'Steer':<10} {'Throttle':<10} {'Brake':<10} {'使用?'}")
        print(f"{'-'*70}")
        
        branch_names = ['Follow', 'Left', 'Right', 'Straight']
        for i, name in enumerate(branch_names):
            start_idx = i * 3
            steer = all_predictions[start_idx]
            throttle = all_predictions[start_idx + 1]
            brake = all_predictions[start_idx + 2]
            
            is_current = '>>> YES' if (i == self.current_command - 2) else ''
            
            print(f"Branch {i:<4} {name:<10} {steer:+.3f}     {throttle:.3f}      {brake:.3f}      {is_current}")
        
        print(f"{'='*70}")
        print(f"{'='*70}\n")
    
    def _print_status(self, start_time, current_speed, control_result):
        """打印状态信息"""
        elapsed = time.time() - start_time
        fps = self.frame_count / elapsed
        
        actual_speed = current_speed * SPEED_NORMALIZATION_MPS * 3.6
        route_info = self.navigation_planner.get_route_info(self.vehicle)
        command_en = COMMAND_NAMES_EN.get(route_info['current_command'], 'Unknown')
        
        print(f"[{elapsed:.1f}s] "
              f"Cmd: {command_en:8s} | "
              f"Prog: {route_info['progress']:5.1f}% | "
              f"Dist: {route_info['remaining_distance']:4.0f}m | "
              f"Spd: {actual_speed:4.1f} | "
              f"Str: {control_result['steer']:+.3f} | "
              f"Thr: {control_result['throttle']:.3f} | "
              f"Brk: {control_result['brake']:.3f} | "
              f"FPS: {fps:.1f}")
              
    def print_statistics(self):
        """打印统计信息"""
        if self.frame_count == 0:
            return
            
        print(f"\n{'='*60}")
        print("推理统计信息")
        print(f"{'='*60}")
        print(f"总帧数: {self.frame_count}")
        print(f"平均推理时间: {self.total_inference_time/self.frame_count*1000:.2f} ms")
        print(f"{'='*60}\n")
        
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        # 清理可解释性模块（释放钩子和内存）
        if self.interp_visualizer is not None:
            self.interp_visualizer.cleanup()
            print("  - 可解释性模块已清理")
        
        if self.sensor_manager is not None:
            self.sensor_manager.cleanup()
            
        if self.vehicle is not None:
            self.vehicle.destroy()
        
        # 清理NPC
        if self.npc_manager is not None:
            self.npc_manager.cleanup_all()
            
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
        print("清理完成！")


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Carla自动驾驶模型实时推理（模块化版本）')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, default='./model/ddp_dynamic_5_best.pth',
                        help='训练好的模型权重路径')
    parser.add_argument('--net-structure', type=int, default=2,
                        help='网络结构类型 (1|2|3)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID，-1表示使用CPU')
    
    # Carla参数
    parser.add_argument('--host', type=str, default='localhost',
                        help='Carla服务器地址')
    parser.add_argument('--port', type=int, default=2000,
                        help='Carla服务器端口')
    parser.add_argument('--town', type=str, default='Town01',
                        help='地图名称')
    parser.add_argument('--vehicle', type=str, default='vehicle.tesla.model3',
                        help='车辆类型')
    
    # 路线规划参数
    parser.add_argument('--spawn-index', type=int, default=1,
                        help='起点索引')
    parser.add_argument('--dest-index', type=int, default=41,
                        help='终点索引')
    parser.add_argument('--list-spawns', action='store_true',        
                        help='列出所有生成点位置后退出')
    
    # 运行参数
    parser.add_argument('--duration', type=int, default=-1,
                        help='运行时长（秒），-1表示无限运行')
    
    # 功能开关
    parser.add_argument('--auto-replan', type=str2bool, default=True,
                        help='到达目的地后自动重新规划路线')
    parser.add_argument('--visualize', type=str2bool, default=True,
                        help='显示可视化窗口')
    parser.add_argument('--post-processing', type=str2bool, default=True,
                        help='启用模型输出后处理（启发式规则优化）')
    parser.add_argument('--image-crop', type=str2bool, default=True,
                        help='启用图像裁剪（去除天空和引擎盖，与训练一致）')
    parser.add_argument('--vis-mode', type=str, default='spectator',
                        choices=['spectator', 'opencv'],
                        help='可视化模式: spectator=CARLA窗口第三人称跟随(推荐), opencv=独立小窗口(旧模式)')
    
    # 天气参数
    parser.add_argument('--weather', type=str, default='ClearSunset',
                        help='天气预设: ClearNoon, ClearSunset, CloudyNoon, CloudySunset, '
                             'WetNoon, WetSunset, WetCloudyNoon, WetCloudySunset, '
                             'HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset')
    
    # NPC参数
    parser.add_argument('--npc-vehicles', type=int, default=0,
                        help='NPC车辆数量，0表示不生成')
    parser.add_argument('--npc-walkers', type=int, default=0,
                        help='NPC行人数量，0表示不生成')
    parser.add_argument('--npc-ignore-lights', type=str2bool, default=False,
                        help='NPC车辆是否忽略红绿灯（默认遵守）')
    parser.add_argument('--npc-ignore-signs', type=str2bool, default=True,
                        help='NPC车辆是否忽略交通标志（默认遵守）')
    parser.add_argument('--npc-vehicle-distance', type=float, default=5.0,
                        help='NPC车辆跟车距离（米）')
    parser.add_argument('--npc-speed-diff', type=float, default=30.0,
                        help='NPC车辆速度差异百分比')
    
    # 可解释性参数
    parser.add_argument('--interpret', type=str2bool, default=False,
                        help='启用可解释性可视化（Grad-CAM热力图、刹车分析等）')
    parser.add_argument('--interpret-save-dir', type=str, default='./interpret_output_1_best',
                        help='可解释性分析结果保存目录')
    parser.add_argument('--interpret-save-interval', type=int, default=1,
                        help='可解释性仪表板保存频率（每N帧保存一次，0表示不自动保存）')
    parser.add_argument('--interpret-device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='可解释性分析计算设备: gpu=使用GPU(快但与CARLA竞争资源), cpu=使用CPU(慢但不影响CARLA渲染)')
    parser.add_argument('--interpret-full', type=str2bool, default=False,
                        help='启用完整可解释性分析(Occlusion/IG/Deletion-Insertion)，False则只用Grad-CAM')
    parser.add_argument('--interpret-row1-layer', type=int, default=-3,
                        help='第一行热力图使用的卷积层索引 (-1=最后层, -3=推荐, -5=高分辨率)')
    parser.add_argument('--interpret-row2-layers', type=str, default='-8,-7,-6,-5,-4,-3,-2,-1',
                        help='第二行多层级热力图使用的卷积层索引列表，逗号分隔 (如: -8,-7,-6,-5,-4,-3,-2,-1 表示所有8层)')
    parser.add_argument('--interpret-ig-steps', type=int, default=30,
                        help='积分梯度(Integrated Gradients)的积分步数，越大精度越高但越慢 (推荐: 30-50)')
    
    args = parser.parse_args()
    
    # 将相对路径转换为基于脚本目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(script_dir, args.model_path)
    
    # 创建NPC配置（如果需要）
    npc_config = None
    if args.npc_vehicles > 0 or args.npc_walkers > 0:
        npc_config = NPCConfig(
            num_vehicles=args.npc_vehicles,
            num_walkers=args.npc_walkers,
            vehicles_ignore_lights=args.npc_ignore_lights,
            vehicles_ignore_signs=args.npc_ignore_signs,
            vehicle_distance=args.npc_vehicle_distance,
            vehicle_speed_difference=args.npc_speed_diff
        )
    
    # 解析第二行多层级热力图的层索引
    interpret_row2_layers = [int(x.strip()) for x in args.interpret_row2_layers.split(',')]
    
    # 创建推理器
    inferencer = CarlaInference(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        town=args.town,
        gpu_id=args.gpu,
        enable_post_processing=args.post_processing,
        enable_image_crop=args.image_crop,
        visualization_mode=args.vis_mode,
        npc_config=npc_config,
        weather=args.weather,
        enable_interpretability=args.interpret,
        interpret_save_dir=args.interpret_save_dir,
        interpret_save_interval=args.interpret_save_interval,
        interpret_device=args.interpret_device,
        interpret_full_analysis=args.interpret_full,
        interpret_row1_layer=args.interpret_row1_layer,
        interpret_row2_layers=interpret_row2_layers,
        interpret_ig_steps=args.interpret_ig_steps
    )
    
    try:
        # 初始化
        inferencer.load_model(net_structure=args.net_structure)
        inferencer.connect_carla()
        
        # 如果是列出生成点模式
        if args.list_spawns:
            spawn_points = inferencer.world.get_map().get_spawn_points()
            print(f"\n{'='*80}")
            print(f"{args.town} 地图的所有生成点（共 {len(spawn_points)} 个）")
            print(f"{'='*80}")
            print(f"{'索引':<6} {'X坐标':<12} {'Y坐标':<12} {'Z坐标':<12} {'朝向(Yaw)':<12}")
            print(f"{'-'*80}")
            
            for i, spawn in enumerate(spawn_points):
                loc = spawn.location
                rot = spawn.rotation
                print(f"{i:<6} {loc.x:<12.2f} {loc.y:<12.2f} {loc.z:<12.2f} {rot.yaw:<12.2f}")
            
            print(f"{'='*80}")
            return
        
        inferencer.spawn_vehicle(
            vehicle_filter=args.vehicle,
            spawn_index=args.spawn_index,
            destination_index=args.dest_index
        )
        inferencer.setup_sensors()
        
        # 等待传感器初始化
        time.sleep(1.0)
        
        # 运行推理
        inferencer.run_inference(
            duration=args.duration,
            visualize=args.visualize,
            auto_replan=args.auto_replan
        )
        
        # 打印统计
        inferencer.print_statistics()
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        inferencer.cleanup()
        print("程序结束")


if __name__ == '__main__':
    main()
