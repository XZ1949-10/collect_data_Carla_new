#!/usr/bin/env python
# coding=utf-8
"""
全自动数据收集器

自动遍历所有生成点组合，收集完整的场景数据。
支持碰撞恢复、异常检测、多天气收集等功能。
"""

import os
import sys
import time
import json
import cv2
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from ..config import CollectorConfig, NPCConfig
from ..core import (
    NPCManager,
    RoutePlanner,
    CollisionRecoveryManager,
    WeatherManager,
    adjust_spawn_transform,
    create_basic_agent,
    is_agents_available,
    get_weather_list,
    # 同步模式管理
    SyncModeManager,
    SyncModeConfig,
    ResourceLifecycleHelper,
    # 红绿灯管理
    TrafficLightManager,
)
from ..utils import FrameVisualizer
from .command_based import CommandBasedCollector


class AutoFullTownCollector:
    """
    全自动数据收集器
    
    特性：
    - 自动遍历所有生成点组合
    - 支持碰撞恢复
    - 支持异常检测
    - 支持多天气收集
    - 支持实时可视化
    """
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        """初始化自动收集器"""
        self.config = config or CollectorConfig()
        
        # CARLA 对象
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.spawn_points = []
        
        # 模块
        self._route_planner: Optional[RoutePlanner] = None
        self._npc_manager: Optional[NPCManager] = None
        self._weather_manager: Optional[WeatherManager] = None
        self._recovery_manager = CollisionRecoveryManager()
        self._inner_collector: Optional[CommandBasedCollector] = None
        self._visualizer: Optional[FrameVisualizer] = None
        
        # 数据保存器（复用实例，避免每次保存都创建新实例）
        self._data_saver = None
        
        # 同步模式管理器
        self._sync_manager: Optional[SyncModeManager] = None
        self._lifecycle_helper: Optional[ResourceLifecycleHelper] = None
        
        # 红绿灯管理器
        self._traffic_light_manager: Optional[TrafficLightManager] = None
        
        # 从配置读取参数
        self.frames_per_route = self.config.frames_per_route
        self.auto_save_interval = self.config.auto_save_interval
        self.route_generation_strategy = self.config.route.strategy
        
        # 高级设置
        self.enable_route_validation = self.config.advanced.enable_route_validation
        self.retry_failed_routes = self.config.advanced.retry_failed_routes
        self.max_retries = self.config.advanced.max_retries
        self.pause_between_routes = self.config.advanced.pause_between_routes
        
        # 统计
        self.total_routes_attempted = 0
        self.total_routes_completed = 0
        self.total_frames_collected = 0
        self.failed_routes: List[Tuple[int, int, str]] = []
    
    def configure_routes(self, min_distance: float = None, max_distance: float = None,
                         overlap_threshold: float = None, turn_priority_ratio: float = None,
                         target_routes_ratio: float = None, max_candidates: int = None):
        """配置路线生成参数"""
        if self._route_planner:
            self._route_planner.configure(
                min_distance=min_distance or self.config.route.min_distance,
                max_distance=max_distance or self.config.route.max_distance,
                overlap_threshold=overlap_threshold or self.config.route.overlap_threshold,
                turn_priority_ratio=turn_priority_ratio or self.config.route.turn_priority_ratio,
                target_routes_ratio=target_routes_ratio or self.config.route.target_routes_ratio,
                max_candidates=max_candidates or self.config.route.max_candidates_to_analyze,
            )
    
    def configure_recovery(self, enabled: bool = None, max_collisions: int = None,
                           min_distance: float = None, skip_distance: float = None):
        """配置碰撞恢复"""
        cfg = self.config.collision_recovery
        self._recovery_manager.configure(
            enabled=enabled if enabled is not None else cfg.enabled,
            max_collisions=max_collisions or cfg.max_collisions_per_route,
            min_distance=min_distance or cfg.min_distance_to_destination,
            skip_distance=skip_distance or cfg.recovery_skip_distance,
        )
    
    def connect(self):
        """连接到CARLA服务器"""
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        print("\n" + "="*70)
        print("🚗 全自动数据收集器")
        print("="*70)
        print(f"正在连接到CARLA服务器 {self.config.host}:{self.config.port}...")
        
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(120.0)
        
        self.world = self.client.get_world()
        current_map = self.world.get_map().name.split('/')[-1]
        
        if current_map != self.config.town:
            print(f"正在加载地图 {self.config.town}...")
            self.world = self.client.load_world(self.config.town)
        else:
            print(f"✅ 已连接到地图 {self.config.town}")
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"✅ 成功连接！共找到 {len(self.spawn_points)} 个生成点")
        
        # 初始化模块
        self._route_planner = RoutePlanner(self.world, self.spawn_points, town=self.config.town)
        self._weather_manager = WeatherManager(self.world)
        self._traffic_light_manager = TrafficLightManager(self.world, verbose=True)
        
        # 初始化同步模式管理器并启用同步模式
        sync_config = SyncModeConfig(simulation_fps=self.config.simulation_fps)
        self._sync_manager = SyncModeManager(self.world, sync_config)
        self._lifecycle_helper = ResourceLifecycleHelper(self._sync_manager)
        
        # 启用同步模式并预热
        # 重要：先确保异步模式，再切换到同步模式，避免状态不一致
        print("🔄 重置并启用同步模式...")
        
        # 先切换到异步模式（清除可能残留的同步状态）
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                print("  ⚠️ 检测到残留的同步模式，先切换到异步...")
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                time.sleep(1.0)  # 等待服务器处理
        except Exception as e:
            print(f"  ⚠️ 重置同步模式时出错: {e}")
        
        # 再启用同步模式
        self._sync_manager.enable_sync_mode()
        time.sleep(0.5)  # 额外等待确保服务器准备好
        
        # 预热 tick（确保同步模式正常工作）
        print("  🔄 预热同步模式...")
        warmup_success = self._sync_manager.warmup_tick(15)
        if warmup_success < 10:
            print(f"  ⚠️ 预热不完整 ({warmup_success}/15)，尝试重置...")
            self._sync_manager.reset_sync_mode()
            self._sync_manager.warmup_tick(10)
        
        print(f"✅ 同步模式已启用 (FPS: {self.config.simulation_fps})")
        
        # 应用路线配置
        self.configure_routes()
        
        # 应用碰撞恢复配置
        self.configure_recovery()
        
        # 应用红绿灯时间配置
        self._configure_traffic_lights()
        
        self._print_config()
    
    def _configure_traffic_lights(self):
        """配置红绿灯时间
        
        根据配置设置所有红绿灯的时间参数。
        使用独立的 TrafficLightManager 模块，安全且不会造成卡顿。
        """
        traffic_light_cfg = self.config.traffic_light
        if not traffic_light_cfg.enabled:
            return
        
        if self._traffic_light_manager is None:
            print("  ⚠️ 红绿灯管理器未初始化")
            return
        
        print(f"🚦 配置红绿灯时间...")
        self._traffic_light_manager.set_timing(
            red=traffic_light_cfg.red_time,
            green=traffic_light_cfg.green_time,
            yellow=traffic_light_cfg.yellow_time
        )
    
    def set_traffic_light_timing(self, red_time: float = None, green_time: float = None, 
                                  yellow_time: float = None) -> bool:
        """手动设置红绿灯时间
        
        参数:
            red_time: 红灯时间（秒），None则不修改
            green_time: 绿灯时间（秒），None则不修改
            yellow_time: 黄灯时间（秒），None则不修改
            
        返回:
            bool: 是否成功
        """
        if self._traffic_light_manager is None:
            print("⚠️ 红绿灯管理器未初始化")
            return False
        
        return self._traffic_light_manager.set_timing(
            red=red_time, green=green_time, yellow=yellow_time
        )
    
    def reset_all_traffic_lights(self) -> bool:
        """重置所有红绿灯状态
        
        让所有红绿灯重新开始计时周期。
        
        返回:
            bool: 是否成功
        """
        if self._traffic_light_manager is None:
            print("⚠️ 红绿灯管理器未初始化")
            return False
        
        return self._traffic_light_manager.reset_all()
    
    @property
    def traffic_light_manager(self) -> Optional[TrafficLightManager]:
        """获取红绿灯管理器实例，供外部直接调用高级功能"""
        return self._traffic_light_manager
    
    def _print_config(self):
        """打印配置信息"""
        print(f"\n📋 配置信息:")
        # 显示总开关状态
        if self.config.obey_traffic_rules:
            print(f"  • 交通规则: ✅ 遵守所有规则（总开关已启用）")
        else:
            print(f"  • 忽略红绿灯: {'✅' if self.config.get_effective_ignore_lights() else '❌'}")
            print(f"  • 忽略停车标志: {'✅' if self.config.get_effective_ignore_signs() else '❌'}")
        
        # 显示红绿灯时间配置
        if self.config.traffic_light.enabled:
            tl_cfg = self.config.traffic_light
            print(f"  • 红绿灯时间: 红{tl_cfg.red_time}s/绿{tl_cfg.green_time}s/黄{tl_cfg.yellow_time}s")
        
        print(f"  • 目标速度: {self.config.target_speed:.1f} km/h")
        print(f"  • 模拟帧率: {self.config.simulation_fps} FPS")
        print(f"  • 每路线帧数: {self.frames_per_route}")
        print(f"  • 自动保存间隔: {self.auto_save_interval}")
        
        npc_cfg = self.config.npc
        if npc_cfg.num_vehicles > 0:
            print(f"  • NPC车辆: {npc_cfg.num_vehicles}")
        if npc_cfg.num_walkers > 0:
            print(f"  • NPC行人: {npc_cfg.num_walkers}")
        
        if self.config.noise.enabled:
            print(f"  • 噪声注入: ✅ (比例: {self.config.noise.noise_ratio:.0%})")
    
    def set_weather(self, weather_name: str) -> bool:
        """设置天气"""
        if self._weather_manager is None:
            return False
        return self._weather_manager.set_weather_preset(weather_name)
    
    def set_weather_from_config(self) -> bool:
        """从配置设置天气"""
        if self._weather_manager is None:
            return False
        
        weather_cfg = self.config.weather
        if weather_cfg.preset:
            return self._weather_manager.set_weather_preset(weather_cfg.preset)
        elif weather_cfg.custom:
            from ..core.weather_manager import CustomWeatherParams
            params = CustomWeatherParams.from_dict(weather_cfg.custom)
            return self._weather_manager.set_custom_weather(params)
        return False
    
    def _spawn_npcs(self):
        """生成NPC"""
        npc_cfg = self.config.npc
        
        if npc_cfg.num_vehicles > 0 or npc_cfg.num_walkers > 0:
            self._npc_manager = NPCManager(
                self.client, self.world, self.blueprint_library,
                sync_manager=self._sync_manager
            )
            self._npc_manager.spawn_all(npc_cfg)
    
    def generate_routes(self, cache_path: Optional[str] = None) -> List[Tuple[int, int, float]]:
        """生成路线"""
        if self._route_planner is None:
            return []
        return self._route_planner.generate_routes(
            strategy=self.route_generation_strategy,
            cache_path=cache_path
        )
    
    def collect_route_data(self, start_idx: int, end_idx: int, save_path: str) -> bool:
        """收集单条路线数据（支持碰撞恢复）"""
        print(f"\n{'='*70}")
        print(f"📊 收集路线: {start_idx} → {end_idx}")
        print(f"{'='*70}")
        
        # 设置恢复管理器
        destination = self.spawn_points[end_idx].location
        route_waypoints = []
        
        if self._route_planner:
            route = self._route_planner.trace_route(
                self.spawn_points[start_idx].location, destination
            )
            if route:
                route_waypoints = list(route)
                print(f"📍 路线waypoints: {len(route_waypoints)} 个点")
        
        self._recovery_manager.set_route(route_waypoints, destination, end_idx)
        
        # 收集循环
        current_spawn_transform = None
        current_start_idx = start_idx
        total_saved_frames = 0
        
        while True:
            result = self._do_single_collection(
                current_start_idx, end_idx, save_path,
                spawn_transform=current_spawn_transform
            )
            
            total_saved_frames += result.get('saved_frames', 0)
            
            if result.get('need_recovery') and self._recovery_manager.can_recover:
                self._recovery_manager.increment_collision()
                
                if self._recovery_manager.collision_count >= \
                   self._recovery_manager.config.max_collisions_per_route:
                    print(f"  ⚠️ 碰撞次数达到上限，终止本路线")
                    break
                
                recovery_transform = result.get('recovery_transform')
                if recovery_transform is not None:
                    print(f"\n🔄 碰撞恢复：从路线waypoint恢复")
                    current_spawn_transform = recovery_transform
                    current_start_idx = None
                    time.sleep(1.0)
                    continue
                else:
                    print(f"  ⚠️ 无法恢复，终止本路线")
                    break
            else:
                break
        
        print(f"\n📊 路线总计: {total_saved_frames} 帧, "
              f"碰撞 {self._recovery_manager.collision_count} 次")
        return result.get('success', False) or total_saved_frames > 0


    def _do_single_collection(self, start_idx: Optional[int], end_idx: int,
                               save_path: str, spawn_transform=None) -> Dict:
        """执行单次收集"""
        result = {'success': False, 'saved_frames': 0, 
                  'need_recovery': False, 'recovery_transform': None}
        
        try:
            self._reset_sync_mode()
            
            # 创建内部收集器
            self._inner_collector = CommandBasedCollector(self.config)
            self._inner_collector.client = self.client
            self._inner_collector.world = self.world
            self._inner_collector.blueprint_library = self.blueprint_library
            # 传递同步模式管理器和资源生命周期辅助
            self._inner_collector._sync_manager = self._sync_manager
            self._inner_collector._lifecycle_helper = self._lifecycle_helper
            
            # 生成车辆
            if spawn_transform is not None:
                if not self._spawn_at_transform(spawn_transform, end_idx):
                    return result
            else:
                if not self._inner_collector.spawn_vehicle(start_idx, end_idx):
                    return result
            
            self._inner_collector.setup_camera()
            self._inner_collector.setup_collision_sensor()
            self._inner_collector.reset_noisers()
            
            result = self._auto_collect(save_path)
            return result
            
        except Exception as e:
            print(f"❌ 收集出错: {e}")
            import traceback
            traceback.print_exc()
            return result
        finally:
            self._cleanup_inner_collector()
    
    def _spawn_at_transform(self, spawn_transform, destination_idx: int) -> bool:
        """在指定位置生成车辆
        
        使用 ResourceLifecycleHelper.spawn_vehicle_safe() 安全生成车辆，
        自动处理物理稳定等待。
        """
        print(f"🚗 在恢复点生成车辆...")
        
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        destination = self.spawn_points[destination_idx].location
        adjusted = adjust_spawn_transform(spawn_transform, 0.5)
        
        # 使用 ResourceLifecycleHelper 安全生成车辆
        if self._lifecycle_helper is not None:
            self._inner_collector.vehicle = self._lifecycle_helper.spawn_vehicle_safe(
                vehicle_bp, adjusted, stabilize_ticks=5
            )
        else:
            # 降级方案：手动生成
            self._inner_collector.vehicle = self.world.try_spawn_actor(vehicle_bp, adjusted)
            if self._inner_collector.vehicle is not None:
                # 等待物理稳定
                if self._sync_manager is not None:
                    self._sync_manager.stabilize_tick(5)
                else:
                    # 没有 SyncModeManager 时，等待一段时间让车辆稳定
                    # 不直接调用 world.tick()，避免与 SyncModeManager 职责重叠
                    time.sleep(0.3)
        
        if self._inner_collector.vehicle is None:
            print("❌ 在恢复点生成车辆失败！")
            return False
        
        print(f"✅ 车辆生成成功！")
        
        if is_agents_available():
            self._setup_recovery_agent(adjusted, destination)
        
        # 关键：初始化 vehicle_list 缓存
        try:
            self._inner_collector._cached_vehicle_list = self.world.get_actors().filter("*vehicle*")
        except Exception as e:
            print(f"⚠️ 初始化 vehicle_list 缓存失败: {e}")
            self._inner_collector._cached_vehicle_list = []
        
        return True
    
    def _setup_recovery_agent(self, spawn_transform, destination):
        """为恢复的车辆配置BasicAgent"""
        # 使用 get_effective_* 方法获取实际配置（考虑总开关）
        self._inner_collector.agent = create_basic_agent(
            vehicle=self._inner_collector.vehicle,
            world_map=self.world.get_map(),
            destination=destination,
            start_location=spawn_transform.location,
            target_speed=self.config.target_speed,
            simulation_fps=self.config.simulation_fps,
            ignore_traffic_lights=self.config.get_effective_ignore_lights(),
            ignore_signs=self.config.get_effective_ignore_signs(),
            ignore_vehicles_percentage=self.config.get_effective_ignore_vehicles_percentage()
        )
        
        if self._inner_collector.agent:
            self._recovery_manager.update_waypoints_from_agent(self._inner_collector.agent)
            print(f"  ✅ BasicAgent 已配置（恢复模式）")

    def _auto_collect(self, save_path: str) -> Dict:
        """自动收集数据"""
        os.makedirs(save_path, exist_ok=True)
        
        result = {'success': False, 'saved_frames': 0,
                  'need_recovery': False, 'recovery_transform': None}
        
        # 初始化可视化器
        if self.config.enable_visualization:
            self._visualizer = FrameVisualizer()
        
        # ⚠️ 重要：必须先预热 actor 缓存，再等待第一帧
        # 因为 wait_for_first_frame() 会调用 step_simulation()，
        # 而 step_simulation() 中的 agent.run_step() 需要 vehicle_list 缓存
        # 否则在同步模式下 get_actors() 可能导致死锁
        self._inner_collector.warmup_actor_cache()
        
        # 等待第一帧
        if not self._inner_collector.wait_for_first_frame(timeout=15.0):
            print("❌ 摄像头初始化失败")
            if self._recovery_manager.config.enabled:
                transform = self._get_recovery_transform()
                if transform:
                    result['need_recovery'] = True
                    result['recovery_transform'] = transform
            return result
        
        saved_frames = 0
        pending_frames = 0
        segment_data = {'rgb': [], 'targets': []}
        segment_start_cmd = None
        loop_count = 0
        
        # 帧率控制 - 基于绝对时间戳，避免累积误差
        target_frame_time = 1.0 / self.config.simulation_fps  # 目标每帧时间
        collection_start_time = time.time()  # 收集开始时间
        next_frame_time = collection_start_time  # 下一帧应该开始的时间
        realtime_sync = getattr(self.config, 'realtime_sync', False)  # 是否启用实时同步
        
        if realtime_sync:
            print(f"🚀 开始数据收集循环... (实时同步模式, 目标帧率: {self.config.simulation_fps} FPS)")
        else:
            print(f"🚀 开始数据收集循环... (最快速度模式)")
        # 【v2.0】移除被动检测逻辑，因为 ensure_sync_mode 已经在 _reset_sync_mode 中验证过
        # 如果仍然出现问题，safe_tick 会自动触发恢复机制
        
        try:
            while (saved_frames + pending_frames) < self.frames_per_route:
                loop_count += 1
                
                # 每 100 帧打印一次调试信息
                if loop_count % 100 == 1:
                    speed = self._inner_collector.get_vehicle_speed()
                    buf_len = len(self._inner_collector.image_buffer)
                    print(f"  [循环 {loop_count}] 速度: {speed:.1f} km/h, 缓冲: {buf_len}, 帧: {saved_frames + pending_frames}")
                
                # 【v2.0】移除被动低速检测
                # 原因：ensure_sync_mode 已经在开始时验证，safe_tick 会自动处理失败
                # 如果需要保留检测作为备用，可以取消下面的注释
                # if loop_count <= 50:
                #     speed = self._inner_collector.get_vehicle_speed()
                #     if speed < 0.5:
                #         consecutive_low_speed += 1
                #         if consecutive_low_speed >= 30:
                #             print(f"  ⚠️ 检测到可能的同步模式问题...")
                #             if self._sync_manager is not None:
                #                 self._sync_manager.ensure_sync_mode()
                #             consecutive_low_speed = 0
                #     else:
                #         consecutive_low_speed = 0
                
                self._inner_collector.step_simulation()
                
                # 获取当前状态（用于可视化和数据收集）
                speed_kmh = self._inner_collector.get_vehicle_speed()
                current_cmd = self._inner_collector.get_navigation_command()
                
                # 安全获取图像（防止竞态条件：len检查和索引访问之间缓冲区可能被清空）
                try:
                    current_image = self._inner_collector.image_buffer[-1].copy()
                except IndexError:
                    current_image = None
                
                # 可视化 - 移到前面，确保即使没有数据也能显示窗口
                if self._visualizer and current_image is not None:
                    vis_info = self._inner_collector.get_visualization_info()
                    self._visualizer.visualize_frame(
                        current_image, speed_kmh, int(current_cmd),
                        saved_frames + pending_frames, self.frames_per_route,
                        pending_frames, is_collecting=True,
                        noise_info=vis_info.to_noise_info(),
                        control_info=vis_info.to_control_info(),
                        expert_control=vis_info.to_expert_control()
                    )
                elif self._visualizer:
                    # 即使没有图像，也调用 waitKey 保持窗口响应
                    cv2.waitKey(1)
                
                if self._inner_collector.is_route_completed():
                    print(f"\n🎯 已到达目的地！")
                    break
                
                # 碰撞和异常检测
                is_collision = self._inner_collector.collision_detected
                is_anomaly = self._inner_collector.check_anomaly()
                
                if is_collision or is_anomaly:
                    if is_collision:
                        print(f"\n💥 检测到碰撞！")
                    
                    if pending_frames > 0:
                        print(f"  🗑️ 丢弃当前 segment（{pending_frames} 帧）")
                    
                    if self._recovery_manager.config.enabled:
                        transform = self._get_recovery_transform()
                        if transform:
                            result['need_recovery'] = True
                            result['recovery_transform'] = transform
                    
                    result['saved_frames'] = saved_frames
                    return result
                
                # 数据收集 - 需要有效图像
                if current_image is None:
                    continue
                
                if current_image.mean() < 5 or speed_kmh > 150:
                    continue
                
                if self._inner_collector.collision_detected:
                    continue
                
                targets = self._inner_collector.build_targets(speed_kmh, current_cmd)
                
                # 如果 targets 为 None，说明噪声启用但专家控制尚未就绪，跳过该帧
                if targets is None:
                    continue
                
                if pending_frames == 0:
                    segment_start_cmd = current_cmd
                
                segment_data['rgb'].append(current_image)
                segment_data['targets'].append(targets)
                pending_frames += 1
                
                # 定期保存
                if pending_frames >= self.auto_save_interval:
                    if not self._inner_collector.collision_detected:
                        self._save_segment(segment_data, save_path, segment_start_cmd)
                        saved_frames += pending_frames
                    segment_data = {'rgb': [], 'targets': []}
                    pending_frames = 0
                    segment_start_cmd = None
                    self._inner_collector.reset_collision_state()
                    self._inner_collector.reset_anomaly_state()
                    self._inner_collector.reset_noisers()
                
                if (saved_frames + pending_frames) % 100 == 0:
                    print(f"  [收集中] 帧数: {saved_frames + pending_frames}/{self.frames_per_route}")
                
                # 帧率限制：仅在启用实时同步时生效
                if realtime_sync:
                    next_frame_time += target_frame_time
                    sleep_time = next_frame_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -target_frame_time:
                        # 如果落后太多（超过一帧），重置时间基准，避免追赶
                        next_frame_time = time.time()
            
            # 保存剩余数据
            if pending_frames > 0 and not self._inner_collector.collision_detected:
                self._save_segment(segment_data, save_path, 
                                   segment_start_cmd if segment_start_cmd else 2.0)
                saved_frames += pending_frames
            
            collection_elapsed = time.time() - collection_start_time
            actual_fps = saved_frames / collection_elapsed if collection_elapsed > 0 else 0
            print(f"\n📊 本次收集: {saved_frames} 帧, 耗时: {collection_elapsed:.1f}秒, 实际帧率: {actual_fps:.1f} FPS")
            self.total_frames_collected += saved_frames
            result['success'] = True
            result['saved_frames'] = saved_frames
            return result
        
        except KeyboardInterrupt:
            # 捕获 Ctrl+C，让上层处理
            print(f"\n⚠️ 收集被中断！已保存 {saved_frames} 帧")
            result['saved_frames'] = saved_frames
            raise  # 重新抛出，让上层的 KeyboardInterrupt 处理器捕获
            
        except Exception as e:
            print(f"❌ 收集出错: {e}")
            result['saved_frames'] = saved_frames
            return result
        finally:
            if self._visualizer:
                self._visualizer.close()
                self._visualizer = None
            cv2.destroyAllWindows()
    
    def _save_segment(self, segment_data: Dict, save_path: str, command: float):
        """保存数据段
        
        复用 DataSaver 实例，避免每次保存都创建新实例。
        """
        if len(segment_data['rgb']) == 0:
            return
        
        # 复用 DataSaver 实例（注意：DataSaver 的属性是 save_path 不是 base_path）
        if self._data_saver is None or self._data_saver.save_path != save_path:
            from ..utils.data_utils import DataSaver
            self._data_saver = DataSaver(save_path)
        
        self._data_saver.save_segment(segment_data['rgb'], segment_data['targets'], command)
    
    def _get_recovery_transform(self):
        """获取恢复点"""
        if self._inner_collector is None or self._inner_collector.vehicle is None:
            return None
        vehicle_location = self._inner_collector.vehicle.get_location()
        return self._recovery_manager.get_recovery_transform(vehicle_location)

    def _reset_sync_mode(self):
        """重置同步模式（使用 SyncModeManager v2.0）
        
        【v2.0 改进】使用 ensure_sync_mode() 代替手动重置，
        自动验证同步模式是否真正生效，失败时自动恢复。
        
        注意：推荐确保 _sync_manager 已初始化，否则将使用降级方案。
        """
        if self._sync_manager is not None:
            # 【v2.0】使用 ensure_sync_mode，自动验证和恢复
            if not self._sync_manager.ensure_sync_mode(warmup=True, verify=True):
                print("  ⚠️ ensure_sync_mode 失败，尝试完整重置...")
                self._sync_manager.reset_sync_mode()
                # 重置后再次验证
                if not self._sync_manager.ensure_sync_mode(warmup=True, verify=True):
                    print("  ❌ 同步模式无法恢复，可能需要重启 CARLA 服务器")
        else:
            # 降级方案：手动设置同步模式
            # 注意：推荐使用 SyncModeManager，降级方案可能不够安全
            print("⚠️ SyncModeManager 未初始化，使用降级方案设置同步模式")
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.config.simulation_fps
                self.world.apply_settings(settings)
                time.sleep(0.5)
            except Exception as e:
                print(f"  ⚠️ 重置同步模式失败: {e}")
    
    def _cleanup_inner_collector(self):
        """清理内部收集器
        
        使用 ResourceLifecycleHelper.destroy_all_safe() 统一管理资源销毁，
        确保在正确的模式下执行清理操作。
        """
        if self._inner_collector is None:
            return
        
        print("  🧹 清理内部收集器...")
        
        try:
            self._inner_collector.agent = None
            self._inner_collector.image_buffer.clear()
            self._inner_collector._cached_vehicle_list = None  # 清理 actor 缓存
        except:
            pass
        
        # 收集需要销毁的传感器
        sensors = []
        if hasattr(self._inner_collector, 'collision_sensor') and \
           self._inner_collector.collision_sensor:
            sensors.append(self._inner_collector.collision_sensor)
        if self._inner_collector.camera:
            sensors.append(self._inner_collector.camera)
        
        # 使用 ResourceLifecycleHelper 安全销毁所有资源
        if self._lifecycle_helper is not None:
            self._lifecycle_helper.destroy_all_safe(
                sensors=sensors,
                vehicle=self._inner_collector.vehicle,
                restore_sync=False  # 不恢复同步模式，后续 _reset_sync_mode 会处理
            )
        else:
            # 降级方案：手动清理
            if self._sync_manager is not None:
                self._sync_manager.ensure_async_mode(wait=True)
            else:
                try:
                    settings = self.world.get_settings()
                    if settings.synchronous_mode:
                        settings.synchronous_mode = False
                        self.world.apply_settings(settings)
                        time.sleep(0.3)
                except:
                    pass
            
            # 批量销毁资源
            for sensor in sensors:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass
            
            try:
                if self._inner_collector.vehicle:
                    self._inner_collector.vehicle.destroy()
            except:
                pass
            
            time.sleep(0.3)
        
        self._inner_collector = None
        print("  ✅ 清理完成")
    
    def _cleanup_npcs(self):
        """清理NPC
        
        注意：必须在异步模式下清理 NPC，否则可能导致死锁或崩溃。
        """
        if self._npc_manager:
            # 确保在异步模式下清理 NPC
            if self._sync_manager is not None:
                try:
                    self._sync_manager.ensure_async_mode(wait=True)
                except Exception as e:
                    print(f"⚠️ 切换异步模式失败: {e}")
            
            try:
                self._npc_manager.cleanup_all()
            except Exception as e:
                print(f"⚠️ NPC 清理过程中出错: {e}")
            finally:
                self._npc_manager = None


    def run(self, save_path: str = None, strategy: str = None, 
            route_cache_path: Optional[str] = None):
        """
        运行全自动收集
        
        参数:
            save_path: 数据保存路径
            strategy: 路线生成策略
            route_cache_path: 路线缓存文件路径
        """
        save_path = save_path or self.config.save_path
        self.route_generation_strategy = strategy or self.config.route.strategy
        
        if route_cache_path is None:
            route_cache_path = os.path.join(
                save_path, f"route_cache_{self.config.town}_{self.route_generation_strategy}.json"
            )
        
        try:
            self.connect()
            
            # 设置天气
            self.set_weather_from_config()
            
            # 生成NPC
            self._spawn_npcs()
            
            # 生成路线
            route_pairs = self.generate_routes(cache_path=route_cache_path)
            
            if not route_pairs:
                print("❌ 没有生成任何路线！")
                return
            
            print("\n" + "="*70)
            print("🚀 开始全自动数据收集")
            print("="*70)
            print(f"总路线数: {len(route_pairs)}")
            print(f"保存路径: {save_path}")
            print("="*70 + "\n")
            
            start_time = time.time()
            
            for idx, (start_idx, end_idx, distance) in enumerate(route_pairs):
                self.total_routes_attempted += 1
                
                print(f"\n📍 路线 {idx+1}/{len(route_pairs)}: "
                      f"{start_idx} → {end_idx} ({distance:.1f}m)")
                
                # 路线验证
                if self.enable_route_validation and self._route_planner:
                    valid, _, _ = self._route_planner.validate_route(start_idx, end_idx)
                    if not valid:
                        self.failed_routes.append((start_idx, end_idx, "不可达"))
                        continue
                
                # 收集数据
                success = False
                retries = 0
                max_retries = self.max_retries if self.retry_failed_routes else 1
                
                while not success and retries <= max_retries:
                    if retries > 0:
                        print(f"  🔄 重试 {retries}/{max_retries}...")
                        self._reset_sync_mode()
                        time.sleep(2.0)
                    
                    try:
                        success = self.collect_route_data(start_idx, end_idx, save_path)
                    except Exception as e:
                        print(f"  ❌ 路线收集异常: {e}")
                        success = False
                    
                    if not success:
                        retries += 1
                
                if success:
                    self.total_routes_completed += 1
                else:
                    self.failed_routes.append((start_idx, end_idx, "收集失败"))
                
                # 路线之间暂停
                if self.pause_between_routes > 0 and idx < len(route_pairs) - 1:
                    time.sleep(self.pause_between_routes)
                
                # 进度显示
                elapsed = time.time() - start_time
                remaining = elapsed / (idx + 1) * (len(route_pairs) - idx - 1)
                print(f"📊 进度: {idx+1}/{len(route_pairs)}, "
                      f"成功: {self.total_routes_completed}, "
                      f"剩余: {remaining/60:.1f}分钟")
            
            self._print_final_statistics(time.time() - start_time, save_path)
            
        except KeyboardInterrupt:
            print("\n⚠️ 收到中断信号，正在清理资源...")
        finally:
            # 清理内部收集器（车辆、传感器等）
            self._cleanup_inner_collector()
            
            # 清理 NPC
            self._cleanup_npcs()
            
            # 【v2.0】使用 ensure_async_mode 恢复异步模式
            if self._sync_manager is not None:
                try:
                    self._sync_manager.ensure_async_mode(wait=True)
                    print("✅ 已恢复异步模式")
                except Exception as e:
                    print(f"⚠️ 恢复异步模式失败: {e}")
            elif self.world:
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    self.world.apply_settings(settings)
                    time.sleep(0.5)
                    print("✅ 已恢复异步模式")
                except Exception as e:
                    print(f"⚠️ 恢复异步模式失败: {e}")
            
            print("✅ 资源清理完成")

    def _print_final_statistics(self, total_time: float, save_path: str):
        """打印最终统计"""
        print("\n" + "="*70)
        print("📊 收集完成 - 最终统计")
        print("="*70)
        print(f"总路线: {self.total_routes_attempted}")
        print(f"成功: {self.total_routes_completed}")
        print(f"失败: {len(self.failed_routes)}")
        print(f"总帧数: {self.total_frames_collected}")
        print(f"耗时: {total_time/60:.1f}分钟")
        print("="*70)
        
        # 保存统计
        stats = {
            'total_routes': self.total_routes_attempted,
            'completed': self.total_routes_completed,
            'frames': self.total_frames_collected,
            'time_seconds': total_time,
            'failed': [{'start': s, 'end': e, 'reason': r} 
                       for s, e, r in self.failed_routes],
            'timestamp': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(save_path, 'collection_statistics.json')
        os.makedirs(save_path, exist_ok=True)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"✅ 统计已保存: {stats_file}")
    
    def reset_statistics(self):
        """重置统计数据（用于多天气收集）"""
        self.total_routes_attempted = 0
        self.total_routes_completed = 0
        self.total_frames_collected = 0
        self.failed_routes = []

    def run_single_weather(self, weather_name: str, save_path: str,
                           strategy: str = None, route_cache_path: Optional[str] = None):
        """
        运行单个天气的数据收集（专为多天气收集设计）
        
        与 run() 方法的区别：
        - 接受天气名称参数，在连接后设置天气
        - 不会在 finally 中恢复异步模式（由调用者处理）
        
        参数:
            weather_name: 天气名称
            save_path: 数据保存路径
            strategy: 路线生成策略
            route_cache_path: 路线缓存文件路径
        """
        save_path = save_path or self.config.save_path
        self.route_generation_strategy = strategy or self.config.route.strategy
        
        if route_cache_path is None:
            route_cache_path = os.path.join(
                save_path, f"route_cache_{self.config.town}_{self.route_generation_strategy}.json"
            )
        
        try:
            # 连接到 CARLA
            self.connect()
            
            # 设置指定的天气
            print(f"🌤️ 设置天气: {weather_name}")
            self.set_weather(weather_name)
            
            # 生成 NPC
            self._spawn_npcs()
            
            # 生成路线
            route_pairs = self.generate_routes(cache_path=route_cache_path)
            
            if not route_pairs:
                print("❌ 没有生成任何路线！")
                return
            
            print("\n" + "="*70)
            print(f"🚀 开始数据收集 - 天气: {weather_name}")
            print("="*70)
            print(f"总路线数: {len(route_pairs)}")
            print(f"保存路径: {save_path}")
            print("="*70 + "\n")
            
            start_time = time.time()
            
            for idx, (start_idx, end_idx, distance) in enumerate(route_pairs):
                self.total_routes_attempted += 1
                
                print(f"\n📍 路线 {idx+1}/{len(route_pairs)}: "
                      f"{start_idx} → {end_idx} ({distance:.1f}m)")
                
                # 路线验证
                if self.enable_route_validation and self._route_planner:
                    valid, _, _ = self._route_planner.validate_route(start_idx, end_idx)
                    if not valid:
                        self.failed_routes.append((start_idx, end_idx, "不可达"))
                        continue
                
                # 收集数据
                success = False
                retries = 0
                max_retries = self.max_retries if self.retry_failed_routes else 1
                
                while not success and retries <= max_retries:
                    if retries > 0:
                        print(f"  🔄 重试 {retries}/{max_retries}...")
                        self._reset_sync_mode()
                        time.sleep(2.0)
                    
                    try:
                        success = self.collect_route_data(start_idx, end_idx, save_path)
                    except Exception as e:
                        print(f"  ❌ 路线收集异常: {e}")
                        success = False
                    
                    if not success:
                        retries += 1
                
                if success:
                    self.total_routes_completed += 1
                else:
                    self.failed_routes.append((start_idx, end_idx, "收集失败"))
                
                # 路线之间暂停
                if self.pause_between_routes > 0 and idx < len(route_pairs) - 1:
                    time.sleep(self.pause_between_routes)
                
                # 进度显示
                elapsed = time.time() - start_time
                remaining = elapsed / (idx + 1) * (len(route_pairs) - idx - 1)
                print(f"📊 进度: {idx+1}/{len(route_pairs)}, "
                      f"成功: {self.total_routes_completed}, "
                      f"剩余: {remaining/60:.1f}分钟")
            
            self._print_final_statistics(time.time() - start_time, save_path)
            
        except KeyboardInterrupt:
            print("\n⚠️ 收到中断信号，正在清理资源...")
            raise  # 重新抛出，让 MultiWeatherCollector 处理
        except Exception as e:
            print(f"\n❌ run_single_weather 发生异常: {e}")
            import traceback
            traceback.print_exc()
            raise  # 重新抛出，让 MultiWeatherCollector 处理
        finally:
            print(f"🧹 [run_single_weather] 开始清理资源 (天气: {weather_name})...")
            # 只清理内部收集器（车辆、传感器等）
            # NPC 清理由 MultiWeatherCollector 统一处理
            self._cleanup_inner_collector()
            
            # 注意：不在这里清理 NPC 和恢复异步模式，由 MultiWeatherCollector 统一处理
            print(f"✅ [run_single_weather] 内部收集器清理完成 (天气: {weather_name})")


# ============================================================================
# 多天气收集器
# ============================================================================

class MultiWeatherCollector:
    """
    多天气数据收集器
    
    自动轮换多个天气进行数据收集，共享路线缓存。
    """
    
    def __init__(self, config: CollectorConfig):
        """
        初始化多天气收集器
        
        参数:
            config: 收集器配置
        """
        self.config = config
        self.total_frames_all_weather = 0
        self.weather_statistics: Dict[str, Dict] = {}
    
    def run(self, weather_list: List[str], base_save_path: str,
            strategy: str = None, route_cache_path: str = None):
        """
        运行多天气收集
        
        参数:
            weather_list: 天气名称列表
            base_save_path: 基础保存路径
            strategy: 路线生成策略
            route_cache_path: 路线缓存路径（所有天气共享）
        """
        strategy = strategy or self.config.route.strategy
        
        # 共享路线缓存
        if route_cache_path is None:
            route_cache_path = os.path.join(
                base_save_path,
                f"route_cache_{self.config.town}_{strategy}.json"
            )
        
        print("\n" + "="*70)
        print("🌤️ 多天气数据收集")
        print("="*70)
        print(f"天气列表: {weather_list}")
        print(f"天气数量: {len(weather_list)}")
        print(f"保存路径: {base_save_path}")
        print(f"路线缓存: {route_cache_path}")
        print("="*70 + "\n")
        
        for idx, weather_name in enumerate(weather_list):
            print(f"\n🔄 开始处理第 {idx+1}/{len(weather_list)} 个天气...")
            print(f"\n{'='*70}")
            print(f"🌤️ [{idx+1}/{len(weather_list)}] 开始收集天气: {weather_name}")
            print(f"{'='*70}")
            
            # 创建天气专属保存路径
            weather_save_path = os.path.join(base_save_path, weather_name)
            
            # 创建收集器
            collector = AutoFullTownCollector(self.config)
            
            try:
                # 直接调用 run_single_weather()，避免重复调用 connect()
                # run_single_weather() 是专门为多天气收集设计的方法
                collector.run_single_weather(
                    weather_name=weather_name,
                    save_path=weather_save_path,
                    strategy=strategy,
                    route_cache_path=route_cache_path
                )
                
                # 记录统计
                self.weather_statistics[weather_name] = {
                    'routes_attempted': collector.total_routes_attempted,
                    'routes_completed': collector.total_routes_completed,
                    'frames_collected': collector.total_frames_collected,
                    'failed_routes': len(collector.failed_routes),
                }
                self.total_frames_all_weather += collector.total_frames_collected
                
            except KeyboardInterrupt:
                print(f"\n⚠️ 用户中断，停止多天气收集")
                # 记录当前天气的统计
                self.weather_statistics[weather_name] = {
                    'routes_attempted': collector.total_routes_attempted,
                    'routes_completed': collector.total_routes_completed,
                    'frames_collected': collector.total_frames_collected,
                    'failed_routes': len(collector.failed_routes),
                    'interrupted': True,
                }
                self.total_frames_all_weather += collector.total_frames_collected
                break  # 退出天气循环
                
            except Exception as e:
                print(f"❌ 天气 {weather_name} 收集失败: {e}")
                import traceback
                traceback.print_exc()
                # 继续下一个天气，不退出循环
                
            finally:
                print(f"🧹 [MultiWeatherCollector] 清理天气 {weather_name} 的资源...")
                # 完整的资源清理（用 try-except 包裹，确保即使清理失败也能继续）
                try:
                    collector._cleanup_inner_collector()
                except Exception as cleanup_error:
                    print(f"⚠️ 清理内部收集器失败: {cleanup_error}")
                
                try:
                    collector._cleanup_npcs()
                except Exception as cleanup_error:
                    print(f"⚠️ 清理 NPC 失败: {cleanup_error}")
                
                # 恢复异步模式
                if collector._sync_manager is not None:
                    try:
                        collector._sync_manager.ensure_async_mode(wait=True)
                    except Exception as cleanup_error:
                        print(f"⚠️ 恢复异步模式失败: {cleanup_error}")
                
                print(f"✅ [MultiWeatherCollector] 天气 {weather_name} 处理完成，继续下一个天气...")
        
        self._print_multi_weather_summary(base_save_path)
    
    def _print_multi_weather_summary(self, save_path: str):
        """打印多天气收集总结"""
        print("\n" + "="*70)
        print("📊 多天气收集完成 - 总结")
        print("="*70)
        
        for weather, stats in self.weather_statistics.items():
            print(f"  {weather}: {stats['frames_collected']} 帧, "
                  f"{stats['routes_completed']}/{stats['routes_attempted']} 路线")
        
        print(f"\n总帧数: {self.total_frames_all_weather}")
        print("="*70)
        
        # 保存总结
        summary = {
            'total_frames': self.total_frames_all_weather,
            'weather_statistics': self.weather_statistics,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(save_path, 'multi_weather_summary.json')
        os.makedirs(save_path, exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"✅ 总结已保存: {summary_file}")


# ============================================================================
# 便捷函数
# ============================================================================

def run_single_weather_collection(config: CollectorConfig, weather_name: str,
                                   save_path: str, strategy: str = None,
                                   route_cache_path: str = None) -> int:
    """
    运行单个天气的数据收集
    
    参数:
        config: 收集器配置
        weather_name: 天气名称
        save_path: 保存路径
        strategy: 路线生成策略
        route_cache_path: 路线缓存路径
        
    返回:
        收集的帧数
    """
    collector = AutoFullTownCollector(config)
    
    try:
        collector.connect()
        collector.set_weather(weather_name)
        collector._spawn_npcs()
        collector.run(
            save_path=save_path,
            strategy=strategy,
            route_cache_path=route_cache_path
        )
        return collector.total_frames_collected
    finally:
        # 完整的资源清理
        collector._cleanup_inner_collector()
        collector._cleanup_npcs()
        
        # 恢复异步模式
        if collector._sync_manager is not None:
            try:
                collector._sync_manager.ensure_async_mode(wait=True)
            except Exception as cleanup_error:
                print(f"⚠️ 恢复异步模式失败: {cleanup_error}")


def run_multi_weather_collection(config: CollectorConfig, weather_list: List[str],
                                  base_save_path: str, strategy: str = None) -> int:
    """
    运行多天气数据收集
    
    参数:
        config: 收集器配置
        weather_list: 天气名称列表
        base_save_path: 基础保存路径
        strategy: 路线生成策略
        
    返回:
        总收集帧数
    """
    collector = MultiWeatherCollector(config)
    collector.run(weather_list, base_save_path, strategy)
    return collector.total_frames_all_weather
