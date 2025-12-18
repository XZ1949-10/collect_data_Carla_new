#!/usr/bin/env python
# coding=utf-8
"""
红绿灯路口数据收集脚本

专门收集经过红绿灯路口的路线数据。

使用方法:
    # 基本使用（使用默认配置）
    python -m collect_data_new.scripts.run_traffic_light_collection
    
    # 指定配置文件
    python -m collect_data_new.scripts.run_traffic_light_collection --config my_config.json
    
    # 命令行参数覆盖
    python -m collect_data_new.scripts.run_traffic_light_collection \
        --town Town01 \
        --save-path ./traffic_light_data \
        --min-lights 1 \
        --max-lights 5 \
        --min-distance 100 \
        --max-distance 300
    
    # 启用可视化
    python -m collect_data_new.scripts.run_traffic_light_collection --visualize
"""

import os
import sys
import json
import signal
import argparse
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 全局变量用于信号处理
_collector = None
_force_exit = False
_interrupt_count = 0


def signal_handler(signum, frame):
    """信号处理器"""
    global _force_exit, _interrupt_count
    _interrupt_count += 1
    
    if _interrupt_count == 1:
        print("\n\n🛑 收到中断信号 (Ctrl+C)，正在安全退出...")
        print("   (再按一次 Ctrl+C 强制退出)")
        _force_exit = True
        raise KeyboardInterrupt()
    else:
        print("\n\n⚠️ 强制退出！")
        os._exit(1)


from collect_data_new.config import (
    CollectorConfig, NoiseConfig, AnomalyConfig, NPCConfig,
    WeatherConfig, MultiWeatherConfig, RouteConfig, TrafficLightRouteConfig,
    CollisionRecoveryConfig, AdvancedConfig, TrafficLightConfig
)
from collect_data_new.core import (
    TrafficLightRoutePlanner,
    TrafficLightRoutePlannerConfig,  # core 中的配置类（包含完整参数）
    SyncModeManager,
    SyncModeConfig,
    ResourceLifecycleHelper,
    NPCManager,
    WeatherManager,
    get_weather_list,
    WEATHER_COLLECTION_PRESETS,
)
from collect_data_new.collectors.auto_collector import AutoFullTownCollector

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


def load_config_file(config_path: str) -> dict:
    """加载配置文件"""
    default_config = {
        'carla_settings': {'host': 'localhost', 'port': 2000, 'town': 'Town01'},
        'traffic_rules': {
            'obey_traffic_rules': False,
            'ignore_traffic_lights': False,  # 红绿灯收集时默认不忽略红绿灯
            'ignore_signs': True,
            'ignore_vehicles_percentage': 0
        },
        'world_settings': {
            'spawn_npc_vehicles': True, 'num_npc_vehicles': 20,
            'spawn_npc_walkers': True, 'num_npc_walkers': 20,
            'npc_behavior': {
                'obey_traffic_rules': False,
                'ignore_traffic_lights': False,  # NPC也遵守红绿灯
                'ignore_signs': True,
                'ignore_walkers': False
            }
        },
        'weather_settings': {'preset': 'ClearNoon', 'custom': {}},
        'route_generation': {
            'strategy': 'traffic_light',  # 使用红绿灯策略
            'min_distance': 100.0,
            'max_distance': 400.0,
            'target_routes_ratio': 1.0,
            'overlap_threshold': 0.5,
            'turn_priority_ratio': 0.7,
            'max_candidates_to_analyze': 0
        },
        'traffic_light_route_settings': {
            'min_traffic_lights': 1,
            'max_traffic_lights': 0,
            'traffic_light_radius': 30.0,
            'prefer_more_lights': True
        },
        'collection_settings': {
            'frames_per_route': 1000,
            'save_path': './traffic_light_data',
            'simulation_fps': 20,
            'target_speed_kmh': 15.0,  # 红绿灯场景适当降速
            'auto_save_interval': 200
        },
        'noise_settings': {
            'enabled': False,
            'lateral_noise': True,
            'longitudinal_noise': False,
            'noise_ratio': 0.4,
            'max_steer_offset': 0.35,
            'max_throttle_offset': 0.2,
            'noise_modes': None
        },
        'collision_recovery': {
            'enabled': True,
            'max_collisions_per_route': 99,
            'min_distance_to_destination': 30.0,
            'recovery_skip_distance': 25.0
        },
        'anomaly_detection': {
            'enabled': True,
            'spin_detection': {'enabled': True, 'threshold_degrees': 270.0, 'time_window': 3.0},
            'rollover_detection': {'enabled': True, 'pitch_threshold': 45.0, 'roll_threshold': 45.0},
            'stuck_detection': {'enabled': True, 'speed_threshold': 0.5, 'time_threshold': 15.0}  # 红绿灯等待时间更长
        },
        'advanced_settings': {
            'enable_route_validation': True,
            'retry_failed_routes': False,
            'max_retries': 3,
            'pause_between_routes': 2
        },
        'traffic_light_settings': {
            'enabled': False,
            'red_time': 5.0,
            'green_time': 10.0,
            'yellow_time': 2.0
        },
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            for section in default_config:
                if section in loaded:
                    if isinstance(default_config[section], dict):
                        default_config[section].update(loaded[section])
                    else:
                        default_config[section] = loaded[section]
            
            print(f"✅ 配置来源: JSON文件 ({config_path})")
        except Exception as e:
            print(f"⚠️ 加载配置失败: {e}")
            print(f"⚠️ 配置来源: 默认配置")
    else:
        print(f"⚠️ 配置文件不存在: {config_path}")
        print(f"⚠️ 配置来源: 默认配置")
    
    return default_config


def create_collector_config(config: dict, args) -> CollectorConfig:
    """从配置字典和命令行参数创建 CollectorConfig"""
    carla = config['carla_settings']
    traffic = config['traffic_rules']
    collection = config['collection_settings']
    noise_cfg = config['noise_settings']
    anomaly_cfg = config.get('anomaly_detection', {})
    npc_cfg = config.get('world_settings', {})
    weather_cfg = config.get('weather_settings', {})
    route_cfg = config.get('route_generation', {})
    tl_route_cfg = config.get('traffic_light_route_settings', {})
    recovery_cfg = config.get('collision_recovery', {})
    advanced_cfg = config.get('advanced_settings', {})
    traffic_light_cfg = config.get('traffic_light_settings', {})
    
    # 命令行参数覆盖
    host = args.host or carla.get('host', 'localhost')
    port = args.port or carla.get('port', 2000)
    town = args.town or carla.get('town', 'Town01')
    target_speed = args.target_speed or collection.get('target_speed_kmh', 15.0)
    fps = args.fps or collection.get('simulation_fps', 20)
    frames_per_route = args.frames_per_route or collection.get('frames_per_route', 1000)
    save_path = args.save_path or collection.get('save_path', './traffic_light_data')
    
    # 噪声配置
    noise_enabled = args.noise or noise_cfg.get('enabled', False)
    noise = NoiseConfig(
        enabled=noise_enabled,
        lateral_enabled=noise_cfg.get('lateral_noise', True),
        longitudinal_enabled=noise_cfg.get('longitudinal_noise', False),
        noise_ratio=noise_cfg.get('noise_ratio', 0.4),
        max_steer_offset=noise_cfg.get('max_steer_offset', 0.35),
        max_throttle_offset=noise_cfg.get('max_throttle_offset', 0.2),
        mode_config=noise_cfg.get('noise_modes')
    )
    
    # 异常检测配置
    spin_cfg = anomaly_cfg.get('spin_detection', {})
    rollover_cfg = anomaly_cfg.get('rollover_detection', {})
    stuck_cfg = anomaly_cfg.get('stuck_detection', {})
    
    anomaly = AnomalyConfig(
        enabled=anomaly_cfg.get('enabled', True),
        spin_enabled=spin_cfg.get('enabled', True),
        spin_threshold_degrees=spin_cfg.get('threshold_degrees', 270.0),
        spin_time_window=spin_cfg.get('time_window', 3.0),
        rollover_enabled=rollover_cfg.get('enabled', True),
        rollover_pitch_threshold=rollover_cfg.get('pitch_threshold', 45.0),
        rollover_roll_threshold=rollover_cfg.get('roll_threshold', 45.0),
        stuck_enabled=stuck_cfg.get('enabled', True),
        stuck_speed_threshold=stuck_cfg.get('speed_threshold', 0.5),
        stuck_time_threshold=stuck_cfg.get('time_threshold', 15.0),  # 红绿灯等待更长
    )
    
    # NPC配置
    npc_behavior = npc_cfg.get('npc_behavior', {})
    npc = NPCConfig(
        num_vehicles=npc_cfg.get('num_npc_vehicles', 20) if npc_cfg.get('spawn_npc_vehicles') else 0,
        num_walkers=npc_cfg.get('num_npc_walkers', 20) if npc_cfg.get('spawn_npc_walkers') else 0,
        vehicles_obey_traffic_rules=npc_behavior.get('obey_traffic_rules', False),
        vehicles_ignore_lights=npc_behavior.get('ignore_traffic_lights', False),
        vehicles_ignore_signs=npc_behavior.get('ignore_signs', True),
        vehicles_ignore_walkers=npc_behavior.get('ignore_walkers', False),
    )
    
    # 天气配置
    weather = WeatherConfig(
        preset=args.weather or weather_cfg.get('preset', 'ClearNoon'),
        custom=weather_cfg.get('custom')
    )
    
    # 红绿灯时间配置
    traffic_light = TrafficLightConfig(
        enabled=traffic_light_cfg.get('enabled', False),
        red_time=traffic_light_cfg.get('red_time', 5.0),
        green_time=traffic_light_cfg.get('green_time', 10.0),
        yellow_time=traffic_light_cfg.get('yellow_time', 2.0),
    )
    
    # 路线配置 - 强制使用 traffic_light 策略
    route = RouteConfig(
        strategy='traffic_light',
        min_distance=args.min_distance or route_cfg.get('min_distance', 100.0),
        max_distance=args.max_distance or route_cfg.get('max_distance', 400.0),
        target_routes_ratio=route_cfg.get('target_routes_ratio', 1.0),
        overlap_threshold=route_cfg.get('overlap_threshold', 0.5),
        turn_priority_ratio=route_cfg.get('turn_priority_ratio', 0.7),
        max_candidates_to_analyze=route_cfg.get('max_candidates_to_analyze', 0),
    )
    
    # 红绿灯路线配置
    traffic_light_route = TrafficLightRouteConfig(
        min_traffic_lights=args.min_lights or tl_route_cfg.get('min_traffic_lights', 1),
        max_traffic_lights=args.max_lights or tl_route_cfg.get('max_traffic_lights', 0),
        traffic_light_radius=args.tl_radius or tl_route_cfg.get('traffic_light_radius', 30.0),
        prefer_more_lights=tl_route_cfg.get('prefer_more_lights', True),
    )
    
    # 碰撞恢复配置
    collision_recovery = CollisionRecoveryConfig(
        enabled=recovery_cfg.get('enabled', True),
        max_collisions_per_route=recovery_cfg.get('max_collisions_per_route', 99),
        min_distance_to_destination=recovery_cfg.get('min_distance_to_destination', 30.0),
        recovery_skip_distance=recovery_cfg.get('recovery_skip_distance', 25.0),
    )
    
    # 高级设置
    advanced = AdvancedConfig(
        enable_route_validation=advanced_cfg.get('enable_route_validation', True),
        retry_failed_routes=advanced_cfg.get('retry_failed_routes', False),
        max_retries=advanced_cfg.get('max_retries', 3),
        pause_between_routes=advanced_cfg.get('pause_between_routes', 2),
    )
    
    # 可视化
    enable_vis = args.visualize or collection.get('enable_visualization', False)
    
    return CollectorConfig(
        host=host,
        port=port,
        town=town,
        obey_traffic_rules=traffic.get('obey_traffic_rules', False),
        ignore_traffic_lights=traffic.get('ignore_traffic_lights', False),
        ignore_signs=traffic.get('ignore_signs', True),
        ignore_vehicles_percentage=traffic.get('ignore_vehicles_percentage', 0),
        target_speed=target_speed,
        simulation_fps=fps,
        save_path=save_path,
        frames_per_route=frames_per_route,
        auto_save_interval=collection.get('auto_save_interval', 200),
        enable_visualization=enable_vis,
        noise=noise,
        anomaly=anomaly,
        npc=npc,
        weather=weather,
        traffic_light=traffic_light,
        route=route,
        traffic_light_route=traffic_light_route,
        collision_recovery=collision_recovery,
        advanced=advanced,
    )


class TrafficLightCollector(AutoFullTownCollector):
    """
    红绿灯路口数据收集器
    
    继承自 AutoFullTownCollector，使用 TrafficLightRoutePlanner 生成路线。
    """
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._tl_route_planner = None
    
    def connect(self):
        """连接到CARLA服务器"""
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        print("\n" + "="*70)
        print("🚦 红绿灯路口数据收集器")
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
        
        # 初始化普通路线规划器（父类方法可能需要）
        from collect_data_new.core import RoutePlanner
        self._route_planner = RoutePlanner(self.world, self.spawn_points, town=self.config.town)
        
        # 初始化红绿灯路线规划器
        self._tl_route_planner = TrafficLightRoutePlanner(
            self.world, self.spawn_points, town=self.config.town
        )
        
        # 配置红绿灯路线参数
        tl_route_cfg = self.config.traffic_light_route
        self._tl_route_planner.configure(
            min_distance=self.config.route.min_distance,
            max_distance=self.config.route.max_distance,
            overlap_threshold=self.config.route.overlap_threshold,
            target_routes_ratio=self.config.route.target_routes_ratio,
            max_candidates=self.config.route.max_candidates_to_analyze,
            min_traffic_lights=tl_route_cfg.min_traffic_lights,
            max_traffic_lights=tl_route_cfg.max_traffic_lights,
            traffic_light_radius=tl_route_cfg.traffic_light_radius,
            prefer_more_lights=tl_route_cfg.prefer_more_lights,
        )
        
        # 初始化天气管理器
        self._weather_manager = WeatherManager(self.world)
        
        # 初始化红绿灯管理器（用于设置红绿灯时间等）
        from collect_data_new.core import TrafficLightManager
        self._traffic_light_manager = TrafficLightManager(self.world, verbose=True)
        
        # 初始化同步模式管理器
        sync_config = SyncModeConfig(simulation_fps=self.config.simulation_fps)
        self._sync_manager = SyncModeManager(self.world, sync_config)
        self._lifecycle_helper = ResourceLifecycleHelper(self._sync_manager)
        
        # 启用同步模式
        print("🔄 启用同步模式...")
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                print("  ⚠️ 检测到残留的同步模式，先切换到异步...")
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                time.sleep(1.0)
        except Exception as e:
            print(f"  ⚠️ 重置同步模式时出错: {e}")
        
        self._sync_manager.enable_sync_mode()
        time.sleep(0.5)
        
        print("  🔄 预热同步模式...")
        warmup_success = self._sync_manager.warmup_tick(15)
        if warmup_success < 10:
            print(f"  ⚠️ 预热不完整 ({warmup_success}/15)，尝试重置...")
            self._sync_manager.reset_sync_mode()
            self._sync_manager.warmup_tick(10)
        
        print(f"✅ 同步模式已启用 (FPS: {self.config.simulation_fps})")
        
        # 应用碰撞恢复配置
        self.configure_recovery()
        
        # 应用红绿灯时间配置
        self._configure_traffic_lights()
        
        self._print_traffic_light_config()
    
    def _print_traffic_light_config(self):
        """打印红绿灯路线配置"""
        tl_cfg = self.config.traffic_light_route
        print(f"\n📋 红绿灯路线配置:")
        print(f"  • 红绿灯数量: {tl_cfg.min_traffic_lights} ~ "
              f"{tl_cfg.max_traffic_lights if tl_cfg.max_traffic_lights > 0 else '不限'}")
        print(f"  • 检测半径: {tl_cfg.traffic_light_radius:.0f}m")
        print(f"  • 优先更多红绿灯: {'✅' if tl_cfg.prefer_more_lights else '❌'}")
        print(f"  • 路线距离: {self.config.route.min_distance:.0f}m ~ {self.config.route.max_distance:.0f}m")
        print(f"  • 目标速度: {self.config.target_speed:.1f} km/h")
        print(f"  • 每路线帧数: {self.frames_per_route}")
        
        # 显示交通规则配置
        if self.config.obey_traffic_rules:
            print(f"  • 交通规则: ✅ 遵守所有规则")
        else:
            print(f"  • 忽略红绿灯: {'✅' if self.config.ignore_traffic_lights else '❌'}")
    
    def generate_routes(self, cache_path=None):
        """生成红绿灯路线"""
        if self._tl_route_planner is None:
            return []
        
        routes = self._tl_route_planner.generate_routes(cache_path=cache_path)
        
        # 转换格式：(start, end, distance, tl_count) -> (start, end, distance)
        # 保持与父类兼容
        return [(s, e, d) for s, e, d, _ in routes]
    
    def run(self, save_path: str = None, route_cache_path: str = None):
        """运行红绿灯数据收集"""
        global _collector
        _collector = self
        
        try:
            self.connect()
            
            # 设置天气
            self.set_weather_from_config()
            
            # 先生成路线（需要在生成 NPC 之前，以便排除路线使用的生成点）
            routes = self.generate_routes(cache_path=route_cache_path)
            
            if not routes:
                print("❌ 没有找到符合条件的红绿灯路线！")
                print("   请尝试:")
                print("   - 降低 min_traffic_lights 参数")
                print("   - 增加 traffic_light_radius 参数")
                print("   - 增加 max_distance 参数")
                return
            
            # 生成NPC（不再排除所有路线的生成点，而是在每条路线开始前动态清除）
            self._spawn_npcs(excluded_spawn_indices=None)
            
            print(f"\n🚦 共找到 {len(routes)} 条红绿灯路线")
            
            # 收集数据
            actual_save_path = save_path or self.config.save_path
            os.makedirs(actual_save_path, exist_ok=True)
            
            for i, (start_idx, end_idx, distance) in enumerate(routes):
                if _force_exit:
                    print("\n⚠️ 收到退出信号，停止收集")
                    break
                
                print(f"\n{'='*70}")
                print(f"📍 路线 {i+1}/{len(routes)}: {start_idx} → {end_idx} ({distance:.0f}m)")
                print(f"{'='*70}")
                
                self.total_routes_attempted += 1
                
                try:
                    success = self.collect_route_data(start_idx, end_idx, actual_save_path)
                    if success:
                        self.total_routes_completed += 1
                except KeyboardInterrupt:
                    print("\n⚠️ 用户中断，保存当前进度...")
                    break
                except Exception as e:
                    print(f"❌ 路线收集失败: {e}")
                    self.failed_routes.append((start_idx, end_idx, str(e)))
                
                # 路线间暂停
                if i < len(routes) - 1:
                    time.sleep(self.pause_between_routes)
            
            self._print_final_statistics()
            
        except KeyboardInterrupt:
            print("\n\n🛑 收集被中断")
        finally:
            self.cleanup()
    
    def _print_final_statistics(self):
        """打印最终统计"""
        print("\n" + "="*70)
        print("📊 红绿灯数据收集统计")
        print("="*70)
        print(f"  • 尝试路线数: {self.total_routes_attempted}")
        print(f"  • 完成路线数: {self.total_routes_completed}")
        print(f"  • 总收集帧数: {self.total_frames_collected}")
        if self.failed_routes:
            print(f"  • 失败路线数: {len(self.failed_routes)}")
        print("="*70)
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        
        # 清理NPC
        if self._npc_manager is not None:
            try:
                self._npc_manager.cleanup_all()
            except:
                pass
        
        # 清理内部收集器
        self._cleanup_inner_collector()
        
        # 恢复异步模式
        if self._sync_manager is not None:
            try:
                self._sync_manager.ensure_async_mode(wait=True)
            except:
                pass
        
        print("✅ 清理完成")


def find_config_file(config_path: str) -> str:
    """查找配置文件"""
    if os.path.isabs(config_path) and os.path.exists(config_path):
        return config_path
    
    search_paths = [
        config_path,
        os.path.join(os.path.dirname(__file__), '..', 'config', config_path),
        os.path.join(os.path.dirname(__file__), config_path),
        os.path.join(os.getcwd(), config_path),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return config_path


def main():
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    parser = argparse.ArgumentParser(
        description='红绿灯路口数据收集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python -m collect_data_new.scripts.run_traffic_light_collection
  
  # 指定红绿灯数量范围
  python -m collect_data_new.scripts.run_traffic_light_collection --min-lights 2 --max-lights 5
  
  # 指定路线距离范围
  python -m collect_data_new.scripts.run_traffic_light_collection --min-distance 150 --max-distance 300
  
  # 启用可视化
  python -m collect_data_new.scripts.run_traffic_light_collection --visualize
        """
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, default='auto_collection_config.json',
                        help='配置文件路径')
    
    # CARLA连接
    parser.add_argument('--host', type=str, help='CARLA服务器地址')
    parser.add_argument('--port', type=int, help='CARLA服务器端口')
    parser.add_argument('--town', type=str, help='地图名称')
    
    # 收集参数
    parser.add_argument('--save-path', type=str, help='数据保存路径')
    parser.add_argument('--frames-per-route', type=int, help='每条路线最大帧数')
    parser.add_argument('--target-speed', type=float, help='目标速度 (km/h)')
    parser.add_argument('--fps', type=int, help='模拟帧率')
    
    # 路线参数
    parser.add_argument('--min-distance', type=float, help='最小路线距离')
    parser.add_argument('--max-distance', type=float, help='最大路线距离')
    parser.add_argument('--route-cache', type=str, help='路线缓存文件路径')
    
    # 红绿灯路线参数
    parser.add_argument('--min-lights', type=int, help='路线最少经过的红绿灯数量')
    parser.add_argument('--max-lights', type=int, help='路线最多经过的红绿灯数量 (0=不限)')
    parser.add_argument('--tl-radius', type=float, help='红绿灯检测半径 (米)')
    
    # 噪声
    parser.add_argument('--noise', action='store_true', help='启用噪声注入')
    
    # 可视化
    parser.add_argument('--visualize', action='store_true', help='启用实时可视化')
    
    # 天气
    parser.add_argument('--weather', type=str, help='天气预设名称')
    
    args = parser.parse_args()
    
    # 查找配置文件
    config_path = find_config_file(args.config)
    
    # 加载配置
    config_dict = load_config_file(config_path)
    
    # 创建收集器配置
    collector_config = create_collector_config(config_dict, args)
    
    # 运行收集
    collector = TrafficLightCollector(collector_config)
    collector.run(
        save_path=collector_config.save_path,
        route_cache_path=args.route_cache
    )


if __name__ == '__main__':
    main()
