#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant
日期: 2025-12-06
说明: 数据收集器基类
      包含CARLA连接、车辆生成、摄像头设置、导航命令获取等共享功能
'''

import glob
import os
import sys
import time
import numpy as np
import cv2
import h5py
from collections import deque

# 设置Windows编码
if sys.platform == 'win32':
    try:
        import io
        if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# 添加CARLA Python API路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入agents模块
try:
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.local_planner import RoadOption
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    print(f"⚠️  警告: 无法导入agents模块: {e}")

# 导入噪声模块
try:
    from noiser import Noiser
    NOISER_AVAILABLE = True
except ImportError:
    NOISER_AVAILABLE = False
    print(f"⚠️  警告: 无法导入noiser模块，噪声功能不可用")

# 导入资源管理器 V2
try:
    from carla_resource_manager_v2 import CarlaResourceManagerV2, ResourceState
    RESOURCE_MANAGER_V2_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_V2_AVAILABLE = False
    print(f"⚠️  警告: 无法导入资源管理器V2，使用传统方式管理资源")


class BaseDataCollector:
    """数据收集器基类，包含共享功能"""
    
    # 命令常量
    COMMAND_NAMES = {2: 'Follow', 3: 'Left', 4: 'Right', 5: 'Straight'}
    COMMAND_COLORS = {2: (100, 255, 100), 3: (100, 100, 255), 
                      4: (255, 100, 100), 5: (255, 255, 100)}
    
    def __init__(self, host='localhost', port=2000, town='Town01',
                 ignore_traffic_lights=True, ignore_signs=True,
                 ignore_vehicles_percentage=80, target_speed=10.0, simulation_fps=20):
        """
        初始化基类
        
        参数:
            host: CARLA服务器地址
            port: CARLA服务器端口
            town: 地图名称
            ignore_traffic_lights: 是否忽略红绿灯
            ignore_signs: 是否忽略停车标志
            ignore_vehicles_percentage: 忽略其他车辆的百分比
            target_speed: 目标速度（km/h）
            simulation_fps: 模拟帧率
        """
        self.host = host
        self.port = port
        self.town = town
        
        # 交通规则配置
        self.ignore_traffic_lights = ignore_traffic_lights
        self.ignore_signs = ignore_signs
        self.ignore_vehicles_percentage = ignore_vehicles_percentage
        self.target_speed = target_speed
        self.simulation_fps = simulation_fps
        
        # CARLA对象
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.camera = None
        self.traffic_manager = None
        self.agent = None
        
        # 数据缓冲
        self.image_buffer = deque(maxlen=1)
        self.current_segment_data = {'rgb': [], 'targets': []}
        
        # 摄像头配置
        self.camera_raw_width = 800
        self.camera_raw_height = 600
        self.image_width = 200
        self.image_height = 88
        
        # 命令追踪
        self.current_command = None
        self.previous_command = None
        self.segment_count = 0
        
        # 转弯命令持久化
        self._last_turn_command = None
        self._turn_command_frames = 0
        self._max_turn_frames = 100
        
        # 统计
        self.total_saved_segments = 0
        self.total_saved_frames = 0
        
        # RoadOption到命令的映射
        if AGENTS_AVAILABLE:
            self.road_option_to_command = {
                RoadOption.LANEFOLLOW: 2.0,
                RoadOption.LEFT: 3.0,
                RoadOption.RIGHT: 4.0,
                RoadOption.STRAIGHT: 5.0,
                RoadOption.CHANGELANELEFT: 2.0,
                RoadOption.CHANGELANERIGHT: 2.0,
                RoadOption.VOID: 2.0
            }
        else:
            self.road_option_to_command = {}
        
        # 可视化
        self.enable_visualization = False
        
        # ========== 噪声注入配置 ==========
        self.noise_enabled = False  # 总开关
        self.lateral_noise_enabled = True   # 横向噪声（转向）
        self.longitudinal_noise_enabled = False  # 纵向噪声（油门/刹车）
        
        # 噪声参数（直观参数）
        self.noise_ratio = 0.4           # 噪声时间占比
        self.max_steer_offset = 0.35     # 最大转向偏移
        self.max_throttle_offset = 0.2   # 最大油门偏移
        
        # 噪声模式配置（可通过配置文件覆盖）
        self.noise_mode_config = None
        
        # 噪声器
        self.lateral_noiser = None
        self.longitudinal_noiser = None
        
        # 初始化默认噪声器
        self._init_noisers()
        
        # 保存专家动作（用于标签，噪声模式下使用）
        self._expert_control = None
        
        # ========== 碰撞检测配置 ==========
        self.collision_sensor = None
        self.collision_detected = False
        self.collision_history = []  # 记录碰撞历史
        
        # ========== 异常行为检测配置 ==========
        self.anomaly_detected = False           # 是否检测到异常行为
        self.anomaly_type = None                # 异常类型: 'spin', 'rollover', 'stuck'
        self.anomaly_detection_enabled = True   # 是否启用异常检测
        
        # 打转检测参数
        self.spin_detection_enabled = True      # 是否检测打转
        self.spin_threshold_degrees = 270.0     # 累计旋转角度阈值（度）
        self.spin_time_window = 3.0             # 检测时间窗口（秒）
        self._yaw_history = []                  # 航向角历史 [(timestamp, yaw), ...]
        
        # 翻车检测参数
        self.rollover_detection_enabled = True  # 是否检测翻车
        self.rollover_pitch_threshold = 45.0    # 俯仰角阈值（度）
        self.rollover_roll_threshold = 45.0     # 横滚角阈值（度）
        
        # 卡住检测参数
        self.stuck_detection_enabled = True     # 是否检测卡住
        self.stuck_speed_threshold = 0.5        # 速度阈值（m/s）
        self.stuck_time_threshold = 5.0         # 卡住时间阈值（秒）
        self._stuck_start_time = None           # 开始卡住的时间
        
        # ========== 资源管理器 V2 ==========
        self._resource_manager = None           # V2 资源管理器实例
    
    def _init_noisers(self, segment_frames=200):
        """初始化噪声器（使用当前参数和帧率）"""
        if NOISER_AVAILABLE:
            # 横向噪声器：影响转向
            self.lateral_noiser = Noiser(
                'Spike', 
                max_offset=self.max_steer_offset, 
                fps=self.simulation_fps,
                mode_config=self.noise_mode_config,
                noise_ratio=self.noise_ratio,
                segment_frames=segment_frames
            )
            
            # 纵向噪声器：影响油门/刹车
            self.longitudinal_noiser = Noiser(
                'Throttle', 
                max_offset=self.max_throttle_offset, 
                fps=self.simulation_fps,
                mode_config=self.noise_mode_config,
                noise_ratio=self.noise_ratio,
                segment_frames=segment_frames
            )
    
    def configure_noise(self, enabled=None, lateral_enabled=None, longitudinal_enabled=None,
                        noise_ratio=None, max_steer_offset=None, max_throttle_offset=None,
                        noise_modes=None):
        """
        配置噪声参数并重新初始化噪声器
        
        参数:
            enabled: 噪声总开关
            lateral_enabled: 横向噪声开关
            longitudinal_enabled: 纵向噪声开关
            noise_ratio: 噪声时间占比 (0-1)
            max_steer_offset: 最大转向偏移 (0-1)
            max_throttle_offset: 最大油门偏移 (0-1)
            noise_modes: 噪声模式配置字典
        """
        # 更新开关
        if enabled is not None:
            self.noise_enabled = enabled
        if lateral_enabled is not None:
            self.lateral_noise_enabled = lateral_enabled
        if longitudinal_enabled is not None:
            self.longitudinal_noise_enabled = longitudinal_enabled
        
        # 更新参数
        if noise_ratio is not None:
            self.noise_ratio = noise_ratio
        if max_steer_offset is not None:
            self.max_steer_offset = max_steer_offset
        if max_throttle_offset is not None:
            self.max_throttle_offset = max_throttle_offset
        
        # 更新噪声模式配置
        if noise_modes is not None:
            self.noise_mode_config = noise_modes
        
        # 重新初始化噪声器
        self._init_noisers()
        
        if self.noise_enabled:
            print(f"🎲 噪声配置已更新:")
            print(f"  • 噪声占比: {self.noise_ratio*100:.0f}%")
            print(f"  • 横向噪声: {'✅' if self.lateral_noise_enabled else '❌'} (max_offset={self.max_steer_offset})")
            print(f"  • 纵向噪声: {'✅' if self.longitudinal_noise_enabled else '❌'} (max_offset={self.max_throttle_offset})")
    
    def reset_noisers(self):
        """重置噪声器状态（在新路线开始时调用）"""
        if self.lateral_noiser is not None:
            self.lateral_noiser.reset()
        if self.longitudinal_noiser is not None:
            self.longitudinal_noiser.reset()
    
    def connect(self):
        """连接到CARLA服务器"""
        print(f"正在连接到CARLA服务器 {self.host}:{self.port}...")
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)  # 增加超时时间到30秒，避免路线切换时超时
        
        print(f"正在加载地图 {self.town}...")
        self.world = self.client.load_world(self.town)
        self.blueprint_library = self.world.get_blueprint_library()
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.simulation_fps
        self.world.apply_settings(settings)
        print(f"✅ 已设置同步模式: {self.simulation_fps} FPS")
        
        print("成功连接到CARLA服务器！")
    
    def spawn_vehicle(self, spawn_index, destination_index):
        """生成车辆并规划路线"""
        print(f"正在生成车辆...")
        
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        if spawn_index >= len(spawn_points) or destination_index >= len(spawn_points):
            print(f"❌ 索引超出范围！最大索引: {len(spawn_points)-1}")
            return False
        
        spawn_point = spawn_points[spawn_index]
        destination = spawn_points[destination_index].location
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        
        if self.vehicle is None:
            print("生成车辆失败！")
            return False
        
        print(f"车辆生成成功！")
        
        # 等待车辆稳定
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)
        
        # 配置车辆控制
        if AGENTS_AVAILABLE:
            self._setup_basic_agent(spawn_point, destination)
        else:
            self._setup_traffic_manager()
        
        # 重置噪声器状态（新路线开始）
        self.reset_noisers()
        
        return True
    
    def _setup_basic_agent(self, spawn_point, destination):
        """配置BasicAgent"""
        print(f"正在配置 BasicAgent...")
        
        # ignore_vehicles_percentage: 
        #   0 = 不忽略任何车辆（完全避让）
        #   1-99 = 部分忽略（BasicAgent只支持布尔值，这里>50视为忽略）
        #   100 = 完全忽略所有车辆
        # 注意：BasicAgent的ignore_vehicles是布尔值，无法精确控制百分比
        # 如需精确百分比控制，请使用Traffic Manager模式
        ignore_vehicles = self.ignore_vehicles_percentage > 50
        
        opt_dict = {
            'target_speed': self.target_speed,
            'ignore_traffic_lights': self.ignore_traffic_lights,
            'ignore_stop_signs': self.ignore_signs,
            'ignore_vehicles': ignore_vehicles,
            'sampling_resolution': 1.0,
            'base_tlight_threshold': 5.0,
            'lateral_control_dict': {
                'K_P': 1.5, 'K_I': 0.0, 'K_D': 0.05,
                'dt': 1.0 / self.simulation_fps
            },
            'longitudinal_control_dict': {
                'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0,
                'dt': 1.0 / self.simulation_fps
            },
            'max_steering': 0.8,
            'max_throttle': 0.75,
            'max_brake': 0.5,
            'base_min_distance': 2.0,
            'distance_ratio': 0.3
        }
        
        self.agent = BasicAgent(
            self.vehicle,
            target_speed=self.target_speed,
            opt_dict=opt_dict,
            map_inst=self.world.get_map()
        )
        
        self.agent.set_destination(destination, start_location=spawn_point.location)
        print(f"  ✅ BasicAgent 已配置 (忽略车辆: {'是' if ignore_vehicles else '否'})")
    
    def _setup_traffic_manager(self):
        """配置Traffic Manager（降级方案）"""
        print(f"正在配置 Traffic Manager...")
        
        self.traffic_manager = self.client.get_trafficmanager()
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        
        if self.ignore_traffic_lights:
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 100)
        if self.ignore_signs:
            self.traffic_manager.ignore_signs_percentage(self.vehicle, 100)
        # 使用配置的百分比值（0-100）
        self.traffic_manager.ignore_vehicles_percentage(self.vehicle, self.ignore_vehicles_percentage)
        
        self.traffic_manager.auto_lane_change(self.vehicle, False)
        print(f"  ✅ Traffic Manager 已配置 (忽略车辆: {self.ignore_vehicles_percentage}%)")
    
    def setup_camera(self):
        """设置摄像头"""
        print("正在设置摄像头...")
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_raw_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_raw_height))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        self.camera.listen(lambda image: self._on_camera_update(image))
        print(f"摄像头设置完成！{self.camera_raw_width}x{self.camera_raw_height} → {self.image_width}x{self.image_height}")
    
    def setup_collision_sensor(self):
        """设置碰撞传感器"""
        if self.vehicle is None:
            print("⚠️  无法设置碰撞传感器：车辆未生成")
            return False
        
        print("正在设置碰撞传感器...")
        
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.collision_detected = False
        self.collision_history = []
        print("✅ 碰撞传感器设置完成！")
        return True
    
    def create_resources_v2(self, spawn_transform, destination=None):
        """使用资源管理器 V2 创建所有资源
        
        参数:
            spawn_transform: 车辆生成位置 (carla.Transform)
            destination: 目的地位置 (carla.Location)，用于配置 BasicAgent
            
        返回:
            bool: 是否成功
        """
        if not RESOURCE_MANAGER_V2_AVAILABLE:
            print("⚠️ 资源管理器 V2 不可用，请使用传统方式")
            return False
        
        # 创建资源管理器
        self._resource_manager = CarlaResourceManagerV2(
            self.world, 
            self.blueprint_library, 
            self.simulation_fps
        )
        
        # 使用 create_all 一次性创建所有资源
        if not self._resource_manager.create_all(
            spawn_transform,
            lambda img: self._on_camera_update(img),
            lambda evt: self._on_collision(evt),
            camera_width=self.camera_raw_width,
            camera_height=self.camera_raw_height
        ):
            self._resource_manager = None
            return False
        
        # 同步引用到 BaseDataCollector
        self.vehicle = self._resource_manager.vehicle
        self.camera = self._resource_manager.camera
        self.collision_sensor = self._resource_manager.collision_sensor
        self.collision_detected = False
        self.collision_history = []
        
        # 配置导航代理
        if destination is not None and AGENTS_AVAILABLE:
            self._setup_basic_agent(spawn_transform, destination)
        
        # 重置噪声器
        self.reset_noisers()
        
        return True
    
    def _on_collision(self, event):
        """碰撞事件回调"""
        self.collision_detected = True
        
        # 获取碰撞对象信息
        other_actor = event.other_actor
        actor_type = other_actor.type_id if other_actor else "unknown"
        
        # 记录碰撞信息
        collision_info = {
            'frame': self.world.get_snapshot().frame if self.world else 0,
            'other_actor': actor_type,
            'impulse': (event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z)
        }
        self.collision_history.append(collision_info)
        
        print(f"💥 检测到碰撞！碰撞对象: {actor_type}")
    
    def reset_collision_state(self):
        """重置碰撞状态（在新segment开始时调用）"""
        self.collision_detected = False
    
    def reset_anomaly_state(self):
        """重置异常状态（在新segment开始时调用）"""
        self.anomaly_detected = False
        self.anomaly_type = None
        self._yaw_history = []
        self._stuck_start_time = None
    
    def check_vehicle_anomaly(self):
        """检测车辆异常行为
        
        检测以下异常：
        1. 打转 - 短时间内累计旋转角度过大
        2. 翻车 - 车辆倾斜角度过大
        3. 卡住 - 长时间速度接近0
        
        返回:
            bool: 是否检测到异常
        """
        if not self.anomaly_detection_enabled or self.vehicle is None:
            return False
        
        if self.anomaly_detected:
            return True
        
        current_time = time.time()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        # 1. 翻车检测
        if self.rollover_detection_enabled:
            pitch = abs(transform.rotation.pitch)
            roll = abs(transform.rotation.roll)
            if pitch > self.rollover_pitch_threshold or roll > self.rollover_roll_threshold:
                self.anomaly_detected = True
                self.anomaly_type = 'rollover'
                print(f"🔄 检测到翻车！俯仰角: {pitch:.1f}°, 横滚角: {roll:.1f}°")
                return True
        
        # 2. 打转检测
        if self.spin_detection_enabled:
            yaw = transform.rotation.yaw
            self._yaw_history.append((current_time, yaw))
            
            # 清理过期数据
            cutoff_time = current_time - self.spin_time_window
            self._yaw_history = [(t, y) for t, y in self._yaw_history if t >= cutoff_time]
            
            # 计算累计旋转角度
            if len(self._yaw_history) >= 2:
                total_rotation = 0.0
                for i in range(1, len(self._yaw_history)):
                    prev_yaw = self._yaw_history[i-1][1]
                    curr_yaw = self._yaw_history[i][1]
                    # 处理角度跨越 -180/180 的情况
                    delta = curr_yaw - prev_yaw
                    if delta > 180:
                        delta -= 360
                    elif delta < -180:
                        delta += 360
                    total_rotation += abs(delta)
                
                if total_rotation > self.spin_threshold_degrees:
                    self.anomaly_detected = True
                    self.anomaly_type = 'spin'
                    print(f"🌀 检测到打转！{self.spin_time_window:.1f}秒内旋转 {total_rotation:.1f}°")
                    return True
        
        # 3. 卡住检测
        if self.stuck_detection_enabled:
            if speed < self.stuck_speed_threshold:
                if self._stuck_start_time is None:
                    self._stuck_start_time = current_time
                elif current_time - self._stuck_start_time > self.stuck_time_threshold:
                    self.anomaly_detected = True
                    self.anomaly_type = 'stuck'
                    print(f"⏸️ 检测到卡住！速度 {speed:.2f} m/s 持续 {self.stuck_time_threshold:.1f}秒")
                    return True
            else:
                self._stuck_start_time = None
        
        return False
    
    def configure_anomaly_detection(self, enabled=None, spin_enabled=None, rollover_enabled=None, 
                                     stuck_enabled=None, spin_threshold=None, spin_time_window=None,
                                     rollover_pitch=None, rollover_roll=None, stuck_speed=None, stuck_time=None):
        """配置异常检测参数"""
        if enabled is not None:
            self.anomaly_detection_enabled = enabled
        if spin_enabled is not None:
            self.spin_detection_enabled = spin_enabled
        if rollover_enabled is not None:
            self.rollover_detection_enabled = rollover_enabled
        if stuck_enabled is not None:
            self.stuck_detection_enabled = stuck_enabled
        if spin_threshold is not None:
            self.spin_threshold_degrees = spin_threshold
        if spin_time_window is not None:
            self.spin_time_window = spin_time_window
        if rollover_pitch is not None:
            self.rollover_pitch_threshold = rollover_pitch
        if rollover_roll is not None:
            self.rollover_roll_threshold = rollover_roll
        if stuck_speed is not None:
            self.stuck_speed_threshold = stuck_speed
        if stuck_time is not None:
            self.stuck_time_threshold = stuck_time
    
    def _on_camera_update(self, image):
        """摄像头回调"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        
        bgr = array[:, :, :3]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        
        # 裁剪区域: [90:485, :] 去除天空和车头
        crop_top = 90
        crop_bottom = 485
        cropped = rgb[crop_top:crop_bottom, :, :]
        
        # 使用双三次插值缩放到目标分辨率 88x200
        processed = cv2.resize(cropped, (self.image_width, self.image_height),
                               interpolation=cv2.INTER_CUBIC)
        self.image_buffer.append(processed)
    
    def _get_navigation_command(self):
        """获取当前导航命令"""
        if not AGENTS_AVAILABLE or self.agent is None:
            return 2.0
        
        try:
            local_planner = self.agent.get_local_planner()
            if local_planner is None:
                return 2.0
            
            waypoints_queue = local_planner.get_plan()
            if waypoints_queue is None or len(waypoints_queue) == 0:
                return 2.0
            
            search_range = min(5, len(waypoints_queue))
            found_turn_command = None
            turn_waypoint_index = -1
            
            for i in range(search_range):
                _, direction = waypoints_queue[i]
                if direction in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
                    found_turn_command = direction
                    turn_waypoint_index = i
                    break
                if direction in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT, RoadOption.LANEFOLLOW]:
                    continue
            
            if found_turn_command is not None and turn_waypoint_index >= 0:
                turn_waypoint = waypoints_queue[turn_waypoint_index][0]
                vehicle_location = self.vehicle.get_location()
                distance_to_turn = vehicle_location.distance(turn_waypoint.transform.location)
                
                if distance_to_turn < 15.0:
                    self._last_turn_command = self.road_option_to_command.get(found_turn_command, 2.0)
                    self._turn_command_frames = 0
                    return self._last_turn_command
                else:
                    return 2.0
            
            if self._last_turn_command is not None and self._last_turn_command != 2.0:
                check_range = min(5, len(waypoints_queue))
                all_lane_follow = all(
                    waypoints_queue[i][1] == RoadOption.LANEFOLLOW
                    for i in range(check_range)
                )
                
                current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
                is_in_junction = current_waypoint.is_junction if current_waypoint else False
                steering = abs(self.vehicle.get_control().steer) if self.vehicle else 0
                
                self._turn_command_frames += 1
                
                should_reset = False
                if self._turn_command_frames >= self._max_turn_frames:
                    should_reset = True
                elif all_lane_follow and not is_in_junction and steering < 0.15:
                    should_reset = True
                elif all_lane_follow and not is_in_junction and self._turn_command_frames > 30:
                    should_reset = True
                elif all_lane_follow and self._turn_command_frames > 50:
                    should_reset = True
                
                if should_reset:
                    self._last_turn_command = None
                    self._turn_command_frames = 0
                    return 2.0
                else:
                    return self._last_turn_command
            
            incoming_wp, incoming_direction = local_planner.get_incoming_waypoint_and_direction(steps=3)
            if incoming_direction is not None and incoming_direction != RoadOption.VOID:
                road_option = incoming_direction
            else:
                road_option = local_planner.target_road_option
                if road_option is None:
                    road_option = RoadOption.LANEFOLLOW
            
            return self.road_option_to_command.get(road_option, 2.0)
            
        except Exception as e:
            print(f"⚠️  获取导航命令失败: {e}")
            return 2.0
    
    def _is_route_completed(self):
        """检查是否到达目的地"""
        if not AGENTS_AVAILABLE or self.agent is None:
            return False
        try:
            return self.agent.done()
        except Exception:
            return False
    
    def _get_vehicle_speed(self):
        """获取车辆速度（km/h）"""
        if self.vehicle is None:
            return 0.0
        velocity = self.vehicle.get_velocity()
        return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def _build_targets(self, speed_kmh, command):
        """构建targets数组
        
        关键：当启用噪声时，使用专家动作作为标签，而非实际执行的带噪声动作。
        这样模型学习的是"从偏离状态如何纠正回来"。
        """
        # 噪声模式下使用专家动作，否则使用实际控制
        if self.noise_enabled and self._expert_control is not None:
            control = self._expert_control
        else:
            control = self.vehicle.get_control()
        
        targets = np.zeros(25, dtype=np.float32)
        targets[0] = control.steer
        targets[1] = control.throttle
        targets[2] = control.brake
        targets[10] = speed_kmh
        targets[24] = command
        return targets
    
    def _save_data_to_h5(self, rgb_list, targets_list, save_path, command, suffix=''):
        """保存数据到H5文件"""
        if len(rgb_list) == 0:
            return
        
        rgb_array = np.array(rgb_list, dtype=np.uint8)
        targets_array = np.array(targets_list, dtype=np.float32)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        command_name = self.COMMAND_NAMES.get(int(command), 'Unknown')
        filename = os.path.join(save_path, f"carla_cmd{int(command)}_{command_name}_{timestamp}{suffix}.h5")
        
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('rgb', data=rgb_array, compression='gzip', compression_opts=4)
            hf.create_dataset('targets', data=targets_array, compression='gzip', compression_opts=4)
        
        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"  ✓ {os.path.basename(filename)} ({len(rgb_array)} 样本, {file_size_mb:.2f} MB)")
        
        self.total_saved_segments += 1
        self.total_saved_frames += len(rgb_array)

    def _visualize_frame(self, image, speed, command, current_frame, total_frames,
                         paused=False, is_collecting=True):
        """可视化当前帧"""
        command = int(command)
        
        # 放大图像
        display_image = cv2.resize(image, (800, 600))
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        
        if paused:
            overlay = display_image.copy()
            cv2.rectangle(overlay, (0, 0), (800, 600), (0, 0, 0), -1)
            display_image = cv2.addWeighted(display_image, 0.6, overlay, 0.4, 0)
        
        # 创建信息面板（加宽以容纳更多信息）
        panel_width = 320
        panel_height = 600
        info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 25
        
        cv2.putText(info_panel, "Data Collection", (10, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 25
        
        if paused:
            cv2.putText(info_panel, "*** PAUSED ***", (10, y_pos), font, 0.7, (0, 165, 255), 2)
            y_pos += 25
        
        if not paused:
            status_text = "SAVING" if is_collecting else "SKIPPING"
            status_color = (100, 255, 100) if is_collecting else (100, 100, 255)
            cv2.putText(info_panel, f"*** {status_text} ***", (10, y_pos), font, 0.6, status_color, 2)
            y_pos += 25
        
        cv2.putText(info_panel, f"Progress: {current_frame}/{total_frames}", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 20
        
        cv2.putText(info_panel, f"Segment: {self.segment_count} frames", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 28
        
        cmd_name = self.COMMAND_NAMES.get(command, 'Unknown')
        cmd_color = self.COMMAND_COLORS.get(command, (255, 255, 255))
        cv2.putText(info_panel, f"Command: {cmd_name}", (10, y_pos), font, 0.6, cmd_color, 2)
        y_pos += 28
        
        speed_color = (100, 255, 100) if speed < 60 else (255, 200, 100)
        cv2.putText(info_panel, f"Speed: {speed:.1f} km/h", (10, y_pos), font, 0.5, speed_color, 1)
        y_pos += 20
        
        cv2.putText(info_panel, f"Target: {self.target_speed:.1f} km/h", (10, y_pos), font, 0.4, (150, 150, 150), 1)
        y_pos += 25
        
        # === 获取控制值 ===
        # 实际执行的控制（可能带噪声）
        if self.vehicle is not None:
            actual_control = self.vehicle.get_control()
            actual_steer = actual_control.steer
            actual_throttle = actual_control.throttle
            actual_brake = actual_control.brake
        else:
            actual_steer = actual_throttle = actual_brake = 0.0
        
        # 专家控制值（标签值，保存到数据集的值）
        if self._expert_control is not None:
            expert_steer = self._expert_control.steer
            expert_throttle = self._expert_control.throttle
            expert_brake = self._expert_control.brake
        else:
            expert_steer = actual_steer
            expert_throttle = actual_throttle
            expert_brake = actual_brake
        
        # === Control 区域：显示数据集中保存的值（专家值/标签值）===
        cv2.putText(info_panel, "=== Label (Dataset) ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        
        # 显示标签值（这是真正保存到数据集的值）
        cv2.putText(info_panel, f"Steer: {expert_steer:+.3f}", (10, y_pos), font, 0.5, (100, 200, 255), 1)
        y_pos += 20
        cv2.putText(info_panel, f"Throttle: {expert_throttle:.3f}", (10, y_pos), font, 0.5, (100, 255, 100), 1)
        y_pos += 20
        cv2.putText(info_panel, f"Brake: {expert_brake:.3f}", (10, y_pos), font, 0.5, (150, 150, 150), 1)
        y_pos += 25
        
        # === 噪声状态显示 ===
        cv2.putText(info_panel, "=== Noise ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        
        if self.noise_enabled:
            # 检查横向噪声状态
            lateral_active = False
            if self.lateral_noise_enabled and self.lateral_noiser is not None:
                lateral_active = self.lateral_noiser.noise_being_set or self.lateral_noiser.remove_noise
            
            # 检查纵向噪声状态
            longitudinal_active = False
            if self.longitudinal_noise_enabled and self.longitudinal_noiser is not None:
                longitudinal_active = self.longitudinal_noiser.noise_being_set or self.longitudinal_noiser.remove_noise
            
            # 横向噪声状态
            lat_status = "ON" if self.lateral_noise_enabled else "OFF"
            lat_color = (0, 165, 255) if lateral_active else ((100, 255, 100) if self.lateral_noise_enabled else (150, 150, 150))
            lat_indicator = " [ACTIVE]" if lateral_active else ""
            cv2.putText(info_panel, f"Lateral: {lat_status}{lat_indicator}", (10, y_pos), font, 0.4, lat_color, 1)
            y_pos += 18
            
            # 纵向噪声状态
            lon_status = "ON" if self.longitudinal_noise_enabled else "OFF"
            lon_color = (0, 165, 255) if longitudinal_active else ((100, 255, 100) if self.longitudinal_noise_enabled else (150, 150, 150))
            lon_indicator = " [ACTIVE]" if longitudinal_active else ""
            cv2.putText(info_panel, f"Longitudinal: {lon_status}{lon_indicator}", (10, y_pos), font, 0.4, lon_color, 1)
            y_pos += 22
            
            # 计算噪声值
            steer_noise = actual_steer - expert_steer
            throttle_noise = actual_throttle - expert_throttle
            
            # 显示噪声计算公式：专家值 + 噪声 = 实际控制
            cv2.putText(info_panel, "--- Steer ---", (10, y_pos), font, 0.35, (180, 180, 180), 1)
            y_pos += 16
            
            # 转向噪声公式
            steer_formula_color = (0, 165, 255) if abs(steer_noise) > 0.01 else (150, 150, 150)
            cv2.putText(info_panel, f"{expert_steer:+.2f} + ({steer_noise:+.2f}) = {actual_steer:+.2f}", 
                       (10, y_pos), font, 0.4, steer_formula_color, 1)
            y_pos += 18
            
            cv2.putText(info_panel, "--- Throttle ---", (10, y_pos), font, 0.35, (180, 180, 180), 1)
            y_pos += 16
            
            # 油门噪声公式
            throttle_formula_color = (0, 165, 255) if abs(throttle_noise) > 0.01 else (150, 150, 150)
            cv2.putText(info_panel, f"{expert_throttle:.2f} + ({throttle_noise:+.2f}) = {actual_throttle:.2f}", 
                       (10, y_pos), font, 0.4, throttle_formula_color, 1)
            y_pos += 20
        else:
            cv2.putText(info_panel, "Noise: OFF", (10, y_pos), font, 0.45, (150, 150, 150), 1)
            y_pos += 20
        
        y_pos += 8
        
        # === 实际控制（车辆执行的值）===
        cv2.putText(info_panel, "=== Actual Control ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        
        # 实际转向（带噪声的）
        actual_steer_color = (100, 100, 255) if (self.noise_enabled and abs(actual_steer - expert_steer) > 0.01) else (100, 200, 255)
        cv2.putText(info_panel, f"Steer: {actual_steer:+.3f}", (10, y_pos), font, 0.45, actual_steer_color, 1)
        y_pos += 18
        
        # 实际油门
        actual_throttle_color = (100, 100, 255) if (self.noise_enabled and abs(actual_throttle - expert_throttle) > 0.01) else (100, 255, 100)
        cv2.putText(info_panel, f"Throttle: {actual_throttle:.3f}", (10, y_pos), font, 0.45, actual_throttle_color, 1)
        y_pos += 18
        
        # 实际刹车
        cv2.putText(info_panel, f"Brake: {actual_brake:.3f}", (10, y_pos), font, 0.45, (150, 150, 150), 1)
        y_pos += 22
        
        # === 统计信息 ===
        cv2.putText(info_panel, "=== Statistics ===", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        y_pos += 22
        cv2.putText(info_panel, f"Saved: {self.total_saved_frames}", (10, y_pos), font, 0.45, (100, 255, 100), 1)
        y_pos += 18
        cv2.putText(info_panel, f"Segments: {self.total_saved_segments}", (10, y_pos), font, 0.45, (200, 200, 200), 1)
        
        combined = np.hstack([display_image, info_panel])
        
        if paused:
            cv2.putText(combined, "PAUSED", (300, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 165, 255), 4)
            cv2.putText(combined, "Waiting for your command...", (150, 360), font, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Data Collection", combined)
        cv2.waitKey(1)
    
    def step_simulation(self):
        """推进一帧模拟（支持噪声注入）
        
        噪声注入逻辑（DAgger风格）：
        1. 获取专家控制信号（用于标签）
        2. 对专家控制添加噪声（用于执行）
        3. 执行带噪声的控制，让车辆产生偏离
        4. 标签记录专家动作，模型学习"如何纠正"
        """
        if AGENTS_AVAILABLE and self.agent is not None:
            # 获取专家控制（始终保存，用于标签）
            expert_control = self.agent.run_step()
            self._expert_control = expert_control
            
            # 根据噪声配置决定执行哪个控制
            if self.noise_enabled and NOISER_AVAILABLE:
                speed_kmh = self._get_vehicle_speed()
                noisy_control = self._apply_noise(expert_control, speed_kmh)
                self.vehicle.apply_control(noisy_control)
            else:
                self.vehicle.apply_control(expert_control)
        
        self.world.tick()
    
    def _apply_noise(self, control, speed_kmh):
        """应用噪声到控制信号
        
        参数:
            control: 专家控制信号
            speed_kmh: 当前车速（km/h），用于调整噪声强度
            
        返回:
            带噪声的控制信号
        """
        # 创建新的控制对象（避免 deepcopy carla.VehicleControl 的 pickle 问题）
        noisy_control = carla.VehicleControl()
        noisy_control.steer = control.steer
        noisy_control.throttle = control.throttle
        noisy_control.brake = control.brake
        noisy_control.hand_brake = control.hand_brake
        noisy_control.reverse = control.reverse
        noisy_control.manual_gear_shift = control.manual_gear_shift
        noisy_control.gear = control.gear
        
        # 纵向噪声（油门/刹车）
        if self.longitudinal_noise_enabled and self.longitudinal_noiser is not None:
            noisy_control, _, _ = self.longitudinal_noiser.compute_noise(noisy_control, speed_kmh)
        
        # 横向噪声（转向）
        if self.lateral_noise_enabled and self.lateral_noiser is not None:
            noisy_control, _, _ = self.lateral_noiser.compute_noise(noisy_control, speed_kmh)
        
        return noisy_control
    
    def wait_for_first_frame(self, timeout=10.0):
        """等待第一帧图像
        
        参数:
            timeout: 超时时间（秒），默认10秒
            
        返回:
            bool: 是否成功获取到第一帧图像
        """
        print("等待第一帧图像...")
        start_time = time.time()
        tick_count = 0
        
        while len(self.image_buffer) == 0:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"⚠️ 等待第一帧图像超时（{timeout}秒），已尝试 {tick_count} 次tick")
                return False
            
            self.step_simulation()
            tick_count += 1
            time.sleep(0.01)
            
            # 每2秒打印一次等待状态
            if tick_count % 200 == 0:
                print(f"  ... 仍在等待图像（已等待 {elapsed:.1f}秒，{tick_count} 次tick）")
        
        print("摄像头就绪！")
        return True
    
    def cleanup(self):
        """清理资源
        
        优先使用资源管理器 V2 进行清理，否则使用传统方式。
        关键：必须先切换到异步模式，再销毁传感器，避免 tick() 死锁
        """
        print("正在清理资源...")
        
        # 1. 清理 agent 引用（不涉及 CARLA actor）
        self.agent = None
        
        # 2. 优先使用资源管理器 V2 清理
        if self._resource_manager is not None:
            self._resource_manager.destroy_all(restore_original_mode=False)
            self._resource_manager = None
            self.vehicle = None
            self.camera = None
            self.collision_sensor = None
        else:
            # 传统清理方式
            # 先切换到异步模式（关键！避免 tick() 死锁）
            if self.world is not None:
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    self.world.apply_settings(settings)
                    time.sleep(0.3)  # 等待模式切换完成
                except:
                    pass
            
            # 按顺序销毁资源（传感器 -> 车辆）
            if self.collision_sensor is not None:
                try:
                    self.collision_sensor.stop()
                    self.collision_sensor.destroy()
                except:
                    pass
                self.collision_sensor = None
            
            if self.camera is not None:
                try:
                    self.camera.stop()
                    self.camera.destroy()
                except:
                    pass
                self.camera = None
            
            if self.vehicle is not None:
                try:
                    self.vehicle.destroy()
                except:
                    pass
                self.vehicle = None
            
            # 等待 CARLA 服务器处理销毁请求
            time.sleep(0.5)
        
        # 3. 清理图像缓冲
        self.image_buffer.clear()
        
        if self.enable_visualization:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print("清理完成！")
