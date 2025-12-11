#!/usr/bin/env python
# coding=utf-8
"""
数据收集器基类

包含CARLA连接、车辆生成、摄像头设置、导航命令获取等共享功能。
"""

import glob
import os
import sys
import time
import numpy as np
import cv2
from collections import deque
from typing import Optional, Dict, Any, List

# 添加CARLA Python API路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("⚠️ CARLA 模块不可用")

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入agents模块
try:
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.local_planner import RoadOption
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    print("⚠️ agents模块不可用")

from ..config import CollectorConfig, COMMAND_NAMES, COMMAND_COLORS, VisualizationInfo
from ..detection import AnomalyDetector, CollisionHandler
from ..noise import Noiser
from .agent_factory import create_basic_agent, is_agents_available
from .sync_mode_manager import SyncModeManager, SyncModeConfig, ResourceLifecycleHelper


class BaseDataCollector:
    """数据收集器基类"""
    
    # RoadOption到命令的映射
    ROAD_OPTION_TO_COMMAND = {}
    if AGENTS_AVAILABLE:
        ROAD_OPTION_TO_COMMAND = {
            RoadOption.LANEFOLLOW: 2.0,
            RoadOption.LEFT: 3.0,
            RoadOption.RIGHT: 4.0,
            RoadOption.STRAIGHT: 5.0,
            RoadOption.CHANGELANELEFT: 2.0,
            RoadOption.CHANGELANERIGHT: 2.0,
            RoadOption.VOID: 2.0
        }
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        """
        初始化基类
        
        参数:
            config: 收集器配置，None则使用默认配置
        """
        self.config = config or CollectorConfig()
        
        # CARLA对象
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None  # 修复：添加初始化
        self.traffic_manager = None
        self.agent = None
        
        # 同步模式管理器和资源生命周期辅助
        self._sync_manager: Optional[SyncModeManager] = None
        self._lifecycle_helper: Optional[ResourceLifecycleHelper] = None
        
        # 数据缓冲
        self.image_buffer = deque(maxlen=1)
        self.current_segment_data = {'rgb': [], 'targets': []}
        
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
        
        # 检测器
        self._anomaly_detector = AnomalyDetector(self.config.anomaly)
        self._collision_handler = CollisionHandler(on_collision=self._on_collision_event)
        
        # 噪声器
        self._lateral_noiser: Optional[Noiser] = None
        self._longitudinal_noiser: Optional[Noiser] = None
        self._init_noisers()
        
        # 专家控制（用于噪声模式）
        self._expert_control = None
        
        # 缓存
        self._cached_vehicle_list = None
    
    def _init_noisers(self, segment_frames: int = None):
        """初始化噪声器
        
        参数:
            segment_frames: segment大小，None则使用 config.auto_save_interval
        """
        noise_cfg = self.config.noise
        
        # 使用配置中的 auto_save_interval 作为默认值
        if segment_frames is None:
            segment_frames = self.config.auto_save_interval
        
        self._lateral_noiser = Noiser(
            'Spike',
            max_offset=noise_cfg.max_steer_offset,
            fps=self.config.simulation_fps,
            mode_config=noise_cfg.mode_config,
            noise_ratio=noise_cfg.noise_ratio,
            segment_frames=segment_frames
        )
        
        self._longitudinal_noiser = Noiser(
            'Throttle',
            max_offset=noise_cfg.max_throttle_offset,
            fps=self.config.simulation_fps,
            mode_config=noise_cfg.mode_config,
            noise_ratio=noise_cfg.noise_ratio,
            segment_frames=segment_frames
        )
    
    def reset_noisers(self):
        """重置噪声器状态"""
        if self._lateral_noiser:
            self._lateral_noiser.reset()
        if self._longitudinal_noiser:
            self._longitudinal_noiser.reset()
    
    def configure_noise(self, enabled: bool = None, lateral_enabled: bool = None,
                        longitudinal_enabled: bool = None, noise_ratio: float = None,
                        max_steer_offset: float = None, max_throttle_offset: float = None,
                        noise_modes: dict = None):
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
        noise_cfg = self.config.noise
        
        if enabled is not None:
            noise_cfg.enabled = enabled
        if lateral_enabled is not None:
            noise_cfg.lateral_enabled = lateral_enabled
        if longitudinal_enabled is not None:
            noise_cfg.longitudinal_enabled = longitudinal_enabled
        if noise_ratio is not None:
            noise_cfg.noise_ratio = noise_ratio
        if max_steer_offset is not None:
            noise_cfg.max_steer_offset = max_steer_offset
        if max_throttle_offset is not None:
            noise_cfg.max_throttle_offset = max_throttle_offset
        if noise_modes is not None:
            noise_cfg.mode_config = noise_modes
        
        # 重新初始化噪声器
        self._init_noisers()
        
        if noise_cfg.enabled:
            print(f"🎲 噪声配置已更新:")
            print(f"  • 噪声占比: {noise_cfg.noise_ratio*100:.0f}%")
            print(f"  • 横向噪声: {'✅' if noise_cfg.lateral_enabled else '❌'} (max_offset={noise_cfg.max_steer_offset})")
            print(f"  • 纵向噪声: {'✅' if noise_cfg.longitudinal_enabled else '❌'} (max_offset={noise_cfg.max_throttle_offset})")
    
    def configure_anomaly_detection(self, enabled: bool = None, spin_enabled: bool = None,
                                     rollover_enabled: bool = None, stuck_enabled: bool = None,
                                     spin_threshold: float = None, spin_time_window: float = None,
                                     rollover_pitch: float = None, rollover_roll: float = None,
                                     stuck_speed: float = None, stuck_time: float = None):
        """
        配置异常检测参数
        
        参数:
            enabled: 总开关
            spin_enabled: 打转检测开关
            rollover_enabled: 翻车检测开关
            stuck_enabled: 卡住检测开关
            spin_threshold: 打转角度阈值（度）
            spin_time_window: 打转检测时间窗口（秒）
            rollover_pitch: 翻车俯仰角阈值（度）
            rollover_roll: 翻车横滚角阈值（度）
            stuck_speed: 卡住速度阈值（m/s）
            stuck_time: 卡住时间阈值（秒）
        """
        self._anomaly_detector.configure(
            enabled=enabled,
            spin_enabled=spin_enabled,
            spin_threshold=spin_threshold,
            spin_time_window=spin_time_window,
            rollover_enabled=rollover_enabled,
            rollover_pitch=rollover_pitch,
            rollover_roll=rollover_roll,
            stuck_enabled=stuck_enabled,
            stuck_speed=stuck_speed,
            stuck_time=stuck_time
        )
    
    def _on_collision_event(self, event):
        """碰撞事件回调"""
        print(f"💥 碰撞检测到！")
    
    # ==================== CARLA 连接 ====================
    
    def connect(self):
        """连接到CARLA服务器"""
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        print(f"正在连接到CARLA服务器 {self.config.host}:{self.config.port}...")
        
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(30.0)
        
        print(f"正在加载地图 {self.config.town}...")
        self.world = self.client.load_world(self.config.town)
        self.blueprint_library = self.world.get_blueprint_library()
        
        # 初始化同步模式管理器和资源生命周期辅助
        sync_config = SyncModeConfig(simulation_fps=self.config.simulation_fps)
        self._sync_manager = SyncModeManager(self.world, sync_config)
        self._lifecycle_helper = ResourceLifecycleHelper(self._sync_manager)
        
        if not self._sync_manager.ensure_sync_mode():
            print("⚠️ 同步模式启用失败，尝试重置...")
            self._sync_manager.reset_sync_mode()
        
        print(f"✅ 已连接！同步模式: {self.config.simulation_fps} FPS")
    
    # ==================== 车辆管理 ====================
    
    def spawn_vehicle(self, spawn_index: int, destination_index: int) -> bool:
        """生成车辆并规划路线
        
        使用 ResourceLifecycleHelper.spawn_vehicle_safe() 安全生成车辆，
        自动处理物理稳定等待。
        """
        print(f"正在生成车辆...")
        
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        if spawn_index >= len(spawn_points) or destination_index >= len(spawn_points):
            print(f"❌ 索引超出范围！最大索引: {len(spawn_points)-1}")
            return False
        
        spawn_point = spawn_points[spawn_index]
        destination = spawn_points[destination_index].location
        
        # 使用 ResourceLifecycleHelper 安全生成车辆
        if self._lifecycle_helper is not None:
            self.vehicle = self._lifecycle_helper.spawn_vehicle_safe(
                vehicle_bp, spawn_point, stabilize_ticks=10
            )
        else:
            # 降级方案：手动生成
            # 注意：推荐使用 ResourceLifecycleHelper，降级方案可能不够安全
            print("⚠️ ResourceLifecycleHelper 未初始化，使用降级方案生成车辆")
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                # 等待车辆稳定
                if self._sync_manager is not None:
                    self._sync_manager.stabilize_tick(10)
                else:
                    # 没有 SyncModeManager 时，等待一段时间让车辆稳定
                    # 不直接调用 world.tick()，避免与 SyncModeManager 职责重叠
                    time.sleep(0.5)
        
        if self.vehicle is None:
            print("❌ 生成车辆失败！")
            return False
        
        print(f"✅ 车辆生成成功！")
        
        # 配置导航
        if is_agents_available():
            self._setup_basic_agent(spawn_point, destination)
        else:
            self._setup_traffic_manager()
        
        # 关键：初始化 vehicle_list 缓存，避免后续 agent.run_step() 中调用 get_actors()
        # 这在同步模式下可能导致死锁
        try:
            self._cached_vehicle_list = self.world.get_actors().filter("*vehicle*")
        except Exception as e:
            print(f"⚠️ 初始化 vehicle_list 缓存失败: {e}")
            self._cached_vehicle_list = []
        
        self.reset_noisers()
        return True
    
    def _setup_basic_agent(self, spawn_point, destination):
        """配置BasicAgent（使用工厂函数）"""
        # 使用 get_effective_* 方法获取实际配置（考虑总开关）
        self.agent = create_basic_agent(
            vehicle=self.vehicle,
            world_map=self.world.get_map(),
            destination=destination,
            start_location=spawn_point.location,
            target_speed=self.config.target_speed,
            simulation_fps=self.config.simulation_fps,
            ignore_traffic_lights=self.config.get_effective_ignore_lights(),
            ignore_signs=self.config.get_effective_ignore_signs(),
            ignore_vehicles_percentage=self.config.get_effective_ignore_vehicles_percentage()
        )
    
    def _setup_traffic_manager(self):
        """配置Traffic Manager（降级方案）"""
        self.traffic_manager = self.client.get_trafficmanager()
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        
        # 使用 get_effective_* 方法获取实际配置（考虑总开关）
        if self.config.get_effective_ignore_lights():
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 100)
        if self.config.get_effective_ignore_signs():
            self.traffic_manager.ignore_signs_percentage(self.vehicle, 100)
        self.traffic_manager.ignore_vehicles_percentage(
            self.vehicle, self.config.get_effective_ignore_vehicles_percentage()
        )
        print(f"✅ Traffic Manager 已配置")
    
    # ==================== 传感器管理 ====================
    
    def setup_camera(self):
        """设置摄像头
        
        使用 ResourceLifecycleHelper.create_sensor_safe() 安全创建传感器，
        自动处理初始化等待。
        """
        cam_cfg = self.config.camera
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(cam_cfg.raw_width))
        camera_bp.set_attribute('image_size_y', str(cam_cfg.raw_height))
        camera_bp.set_attribute('fov', str(cam_cfg.fov))
        
        camera_transform = carla.Transform(
            carla.Location(x=cam_cfg.location[0], y=cam_cfg.location[1], z=cam_cfg.location[2]),
            carla.Rotation(pitch=cam_cfg.rotation[1])
        )
        
        # 使用 ResourceLifecycleHelper 安全创建传感器
        if self._lifecycle_helper is not None:
            self.camera = self._lifecycle_helper.create_sensor_safe(
                camera_bp, camera_transform, self.vehicle, 
                self._on_camera_update, init_ticks=10
            )
        else:
            # 降级方案：手动创建
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform,
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
            if self.camera is not None:
                self.camera.listen(self._on_camera_update)
                # 等待传感器初始化
                if self._sync_manager is not None:
                    self._sync_manager.stabilize_tick(10)
                else:
                    time.sleep(0.5)
        
        if self.camera is None:
            print("❌ 摄像头创建失败！")
            return
        
        print(f"✅ 摄像头设置完成！")
    
    def setup_collision_sensor(self):
        """设置碰撞传感器
        
        使用 ResourceLifecycleHelper.create_sensor_safe() 安全创建传感器。
        """
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        
        # 使用 ResourceLifecycleHelper 安全创建传感器
        if self._lifecycle_helper is not None:
            self.collision_sensor = self._lifecycle_helper.create_sensor_safe(
                collision_bp, collision_transform, self.vehicle,
                self._collision_handler.handle_collision, init_ticks=5
            )
        else:
            # 降级方案：手动创建
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, collision_transform, attach_to=self.vehicle
            )
            if self.collision_sensor is not None:
                self.collision_sensor.listen(self._collision_handler.handle_collision)
        
        if self.collision_sensor is None:
            print("❌ 碰撞传感器创建失败！")
            return
        
        print("✅ 碰撞传感器设置完成！")
    
    def _on_camera_update(self, image):
        """摄像头回调"""
        cam_cfg = self.config.camera
        
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        
        bgr = array[:, :, :3]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        
        # 裁剪
        cropped = rgb[cam_cfg.crop_top:cam_cfg.crop_bottom, :, :]
        
        # 缩放
        processed = cv2.resize(
            cropped, (cam_cfg.output_width, cam_cfg.output_height),
            interpolation=cv2.INTER_CUBIC
        )
        self.image_buffer.append(processed)
    
    # ==================== 导航命令 ====================
    
    def get_navigation_command(self) -> float:
        """获取当前导航命令"""
        if not is_agents_available() or self.agent is None:
            return 2.0
        
        try:
            local_planner = self.agent.get_local_planner()
            if local_planner is None:
                return 2.0
            
            waypoints_queue = local_planner.get_plan()
            if waypoints_queue is None or len(waypoints_queue) == 0:
                return 2.0
            
            # 搜索转弯命令
            search_range = min(5, len(waypoints_queue))
            for i in range(search_range):
                _, direction = waypoints_queue[i]
                if direction in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
                    turn_waypoint = waypoints_queue[i][0]
                    distance = self.vehicle.get_location().distance(turn_waypoint.transform.location)
                    
                    if distance < 15.0:
                        self._last_turn_command = self.ROAD_OPTION_TO_COMMAND.get(direction, 2.0)
                        self._turn_command_frames = 0
                        return self._last_turn_command
            
            # 持久化转弯命令
            if self._last_turn_command is not None and self._last_turn_command != 2.0:
                self._turn_command_frames += 1
                if self._turn_command_frames >= self._max_turn_frames:
                    self._last_turn_command = None
                    self._turn_command_frames = 0
                else:
                    return self._last_turn_command
            
            return 2.0
            
        except Exception as e:
            print(f"⚠️ 获取导航命令失败: {e}")
            return 2.0
    
    def is_route_completed(self) -> bool:
        """检查是否到达目的地"""
        if not is_agents_available() or self.agent is None:
            return False
        try:
            return self.agent.done()
        except:
            return False
    
    # ==================== 车辆状态 ====================
    
    def get_vehicle_speed(self) -> float:
        """获取车辆速度（km/h）"""
        if self.vehicle is None:
            return 0.0
        velocity = self.vehicle.get_velocity()
        return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def check_anomaly(self) -> bool:
        """检测车辆异常"""
        if self.vehicle is None:
            return False
        return self._anomaly_detector.check(self.vehicle)
    
    @property
    def collision_detected(self) -> bool:
        """是否检测到碰撞"""
        return self._collision_handler.collision_detected
    
    def reset_collision_state(self):
        """重置碰撞状态"""
        self._collision_handler.reset()
    
    def reset_anomaly_state(self):
        """重置异常状态"""
        self._anomaly_detector.reset()
    
    # ==================== 模拟控制 ====================
    
    def warmup_actor_cache(self):
        """预热 actor 缓存，避免同步模式下首次调用 agent.run_step() 时死锁
        
        在同步模式下，get_actors() 需要等待服务器响应，但服务器在等待 tick()。
        如果在 tick() 之前调用 get_actors()，会形成死锁。
        
        此方法在进入主循环前调用，确保：
        1. 先执行多次 tick() 推进模拟并稳定
        2. 立即缓存 vehicle_list 供后续 agent.run_step() 使用
        
        注意：推荐确保 _sync_manager 已初始化，否则将使用降级方案。
        """
        if self.world is None:
            return
        
        try:
            # 使用 SyncModeManager 的 warmup_tick（多次 tick 确保稳定）
            if self._sync_manager is not None:
                # 确保同步模式已启用
                if not self._sync_manager.is_sync:
                    print("  🔄 启用同步模式...")
                    self._sync_manager.enable_sync_mode()
                
                success_count = self._sync_manager.warmup_tick(10)
                if success_count < 5:
                    print(f"⚠️ Actor 缓存预热不完整: {success_count}/10 次 tick 成功")
                    # 尝试重置同步模式
                    print("  🔄 尝试重置同步模式...")
                    self._sync_manager.reset_sync_mode()
                    success_count = self._sync_manager.warmup_tick(5)
            else:
                # 降级方案：等待一段时间让模拟稳定
                # 注意：不直接调用 world.tick()，避免与 SyncModeManager 职责重叠
                print("⚠️ SyncModeManager 未初始化，使用降级方案等待")
                time.sleep(0.5)
            
            # 立即缓存 actors - 关键修复：即使没有 agent 也要缓存
            # 因为后续 step_simulation 中会用到
            try:
                self._cached_vehicle_list = self.world.get_actors().filter("*vehicle*")
                print("✅ Actor 缓存预热完成")
            except Exception as e:
                print(f"⚠️ 缓存 vehicle_list 失败: {e}")
                # 设置空列表避免后续 get_actors() 调用
                self._cached_vehicle_list = []
        except RuntimeError as e:
            print(f"⚠️ Actor 缓存预热超时: {e}")
            self._cached_vehicle_list = []
        except Exception as e:
            print(f"⚠️ Actor 缓存预热失败: {e}")
            self._cached_vehicle_list = []
    
    def step_simulation(self, debug: bool = False):
        """推进一帧模拟
        
        使用 SyncModeManager 统一管理 tick 调用，确保：
        - 带超时，避免无限阻塞
        - 带重试，提高稳定性
        - 统一的错误处理
        
        参数:
            debug: 是否打印调试信息
        """
        # 先执行 tick，确保模拟推进
        if debug:
            print("    [DEBUG] 开始 tick...")
        
        tick_success = self._do_tick()
        
        if debug:
            print(f"    [DEBUG] tick 完成: {tick_success}")
        
        # 注意：即使 tick 失败，也要尝试执行 agent 逻辑
        # 因为 tick 失败可能是暂时的，而 agent 逻辑可以帮助恢复
        
        if is_agents_available() and self.agent is not None:
            # 关键修复：确保 cached_vehicles 不为 None
            # 如果为 None，agent.run_step() 内部会调用 world.get_actors()
            # 这在同步模式下可能导致死锁
            cached_vehicles = getattr(self, '_cached_vehicle_list', None)
            if cached_vehicles is None:
                # 提供空列表而不是 None，避免 agent 内部调用 get_actors()
                cached_vehicles = []
            
            if debug:
                print("    [DEBUG] 开始 agent.run_step()...")
            
            try:
                expert_control = self.agent.run_step(vehicle_list=cached_vehicles)
            except Exception as e:
                print(f"⚠️ agent.run_step() 出错: {e}")
                return
            
            if debug:
                print(f"    [DEBUG] agent.run_step() 完成")
            
            if expert_control is None:
                return
            
            self._expert_control = expert_control
            
            # 应用噪声
            if self.config.noise.enabled:
                speed_kmh = self.get_vehicle_speed()
                noisy_control = self._apply_noise(expert_control, speed_kmh)
                self.vehicle.apply_control(noisy_control)
            else:
                self.vehicle.apply_control(expert_control)
        
        # 缓存actors - 每 10 帧更新一次，减少开销
        # 关键：在 tick 之后立即更新缓存，此时服务器已响应
        self._tick_count = getattr(self, '_tick_count', 0) + 1
        if self._tick_count % 10 == 0:  # 每 10 帧更新一次
            try:
                self._cached_vehicle_list = self.world.get_actors().filter("*vehicle*")
            except Exception as e:
                if debug:
                    print(f"    [DEBUG] 更新 vehicle_list 失败: {e}")
                # 保持旧缓存或空列表
                if not hasattr(self, '_cached_vehicle_list'):
                    self._cached_vehicle_list = []
    
    def _do_tick(self) -> bool:
        """执行 tick（统一入口）
        
        优先使用 SyncModeManager，否则使用降级方案。
        
        注意：推荐确保 _sync_manager 已初始化。降级方案仅在异步模式下有效，
        同步模式下必须使用 SyncModeManager。
        
        返回:
            bool: 是否成功
        """
        if self._sync_manager is not None:
            success = self._sync_manager.safe_tick()
            if not success:
                # tick 失败时尝试重置同步模式
                self._tick_fail_count = getattr(self, '_tick_fail_count', 0) + 1
                if self._tick_fail_count >= 5:
                    print(f"⚠️ tick 连续失败 {self._tick_fail_count} 次，尝试重置同步模式...")
                    self._sync_manager.reset_sync_mode()
                    self._tick_fail_count = 0
                    # 重置后再试一次
                    success = self._sync_manager.safe_tick()
            else:
                self._tick_fail_count = 0
            return success
        else:
            # 降级方案：等待一帧时间（适用于异步模式）
            # 注意：不直接调用 world.tick()，避免与 SyncModeManager 职责重叠
            # 如果需要同步模式，必须初始化 _sync_manager
            time.sleep(1.0 / self.config.simulation_fps)
            return True
    
    def _apply_noise(self, control, speed_kmh: float):
        """应用噪声到控制信号"""
        noisy_control = carla.VehicleControl()
        noisy_control.steer = control.steer
        noisy_control.throttle = control.throttle
        noisy_control.brake = control.brake
        noisy_control.hand_brake = control.hand_brake
        noisy_control.reverse = control.reverse
        
        noise_cfg = self.config.noise
        
        if noise_cfg.longitudinal_enabled and self._longitudinal_noiser:
            noisy_control, _, _ = self._longitudinal_noiser.compute_noise(noisy_control, speed_kmh)
        
        if noise_cfg.lateral_enabled and self._lateral_noiser:
            noisy_control, _, _ = self._lateral_noiser.compute_noise(noisy_control, speed_kmh)
        
        return noisy_control
    
    def wait_for_first_frame(self, timeout: float = 10.0) -> bool:
        """等待第一帧图像（使用 SyncModeManager v2.0）"""
        print("等待第一帧图像...")
        start_time = time.time()
        tick_count = 0
        tick_fail_count = 0
        
        while len(self.image_buffer) == 0:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"⚠️ 等待超时 ({elapsed:.1f}s, {tick_count} ticks, {tick_fail_count} 失败)")
                return False
            
            # 执行 tick
            tick_success = self._do_tick()
            tick_count += 1
            
            if not tick_success:
                tick_fail_count += 1
                # 【v2.0】如果连续多次 tick 失败，使用 ensure_sync_mode 自动恢复
                if tick_fail_count >= 5:
                    print(f"⚠️ tick 连续失败 {tick_fail_count} 次，尝试恢复同步模式...")
                    if self._sync_manager is not None:
                        # 使用 v2.0 的 ensure_sync_mode，自动验证和恢复
                        if not self._sync_manager.ensure_sync_mode(warmup=True, verify=True):
                            print("  ⚠️ ensure_sync_mode 失败，尝试完整重置...")
                            self._sync_manager.reset_sync_mode()
                    else:
                        # 降级方案：等待一段时间
                        # 注意：不直接操作同步模式设置，避免与 SyncModeManager 职责重叠
                        print("  ⚠️ SyncModeManager 未初始化，无法恢复同步模式")
                        time.sleep(0.5)
                    tick_fail_count = 0
                    time.sleep(0.5)
            else:
                tick_fail_count = 0  # 重置失败计数
            
            # 防止 tick 过快
            time.sleep(0.01)
            
            # 每 20 次 tick 打印一次进度
            if tick_count % 20 == 0:
                print(f"  等待中... ({tick_count} ticks, {elapsed:.1f}s, buffer={len(self.image_buffer)})")
        
        print(f"✅ 摄像头就绪！({tick_count} ticks)")
        return True
    
    # ==================== 数据构建 ====================
    
    def build_targets(self, speed_kmh: float, command: float) -> np.ndarray:
        """构建targets数组"""
        if self.config.noise.enabled and self._expert_control is not None:
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
    
    def get_visualization_info(self) -> VisualizationInfo:
        """
        获取可视化所需的信息
        
        返回一个VisualizationInfo对象，包含噪声状态、专家控制和实际控制信息。
        这个方法实现了收集器和可视化器之间的解耦。
        """
        info = VisualizationInfo()
        noise_cfg = self.config.noise
        
        # 噪声配置
        info.noise_enabled = noise_cfg.enabled
        info.lateral_enabled = noise_cfg.lateral_enabled
        info.longitudinal_enabled = noise_cfg.longitudinal_enabled
        
        # 检查噪声是否正在激活
        if noise_cfg.enabled:
            if noise_cfg.lateral_enabled and self._lateral_noiser is not None:
                info.lateral_active = (
                    self._lateral_noiser.noise_being_set or 
                    self._lateral_noiser.remove_noise
                )
            if noise_cfg.longitudinal_enabled and self._longitudinal_noiser is not None:
                info.longitudinal_active = (
                    self._longitudinal_noiser.noise_being_set or 
                    self._longitudinal_noiser.remove_noise
                )
        
        # 专家控制（标签值）
        if self._expert_control is not None:
            info.expert_steer = self._expert_control.steer
            info.expert_throttle = self._expert_control.throttle
            info.expert_brake = self._expert_control.brake
        
        # 实际控制（车辆执行的值）
        if self.vehicle is not None:
            actual_control = self.vehicle.get_control()
            info.actual_steer = actual_control.steer
            info.actual_throttle = actual_control.throttle
            info.actual_brake = actual_control.brake
        
        return info
    
    # ==================== 清理 ====================
    
    def cleanup(self):
        """清理资源
        
        使用 ResourceLifecycleHelper.destroy_all_safe() 统一管理资源销毁。
        """
        print("正在清理资源...")
        
        self.agent = None
        
        # 收集需要销毁的传感器
        sensors = []
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            sensors.append(self.collision_sensor)
        if self.camera:
            sensors.append(self.camera)
        
        # 使用 ResourceLifecycleHelper 安全销毁所有资源
        if self._lifecycle_helper is not None:
            self._lifecycle_helper.destroy_all_safe(
                sensors=sensors,
                vehicle=self.vehicle,
                restore_sync=False
            )
        else:
            # 降级方案：手动清理
            # 注意：推荐使用 ResourceLifecycleHelper，降级方案可能不够安全
            print("⚠️ ResourceLifecycleHelper 未初始化，使用降级方案清理资源")
            
            # 尝试切换到异步模式（销毁资源前必须）
            if self._sync_manager is not None:
                self._sync_manager.ensure_async_mode(wait=True)
            else:
                # 没有 SyncModeManager 时，等待一段时间确保稳定
                time.sleep(0.5)
            
            # 销毁传感器
            for sensor in sensors:
                try:
                    sensor.stop()
                except:
                    pass
                try:
                    sensor.destroy()
                except:
                    pass
            
            # 销毁车辆
            if self.vehicle:
                try:
                    self.vehicle.destroy()
                except:
                    pass
            
            time.sleep(0.3)
        
        # 清理引用
        self.collision_sensor = None
        self.camera = None
        self.vehicle = None
        self.image_buffer.clear()
        
        if self.config.enable_visualization:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print("✅ 清理完成！")
