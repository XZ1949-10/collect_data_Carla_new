#!/usr/bin/env python
# coding=utf-8
"""
全自动数据收集脚本

使用方法:
    # 基本使用（使用默认配置）
    python -m collect_data_new.scripts.run_auto_collection
    
    # 指定配置文件
    python -m collect_data_new.scripts.run_auto_collection --config my_config.json
    
    # 命令行参数覆盖
    python -m collect_data_new.scripts.run_auto_collection \
        --town Town01 \
        --save-path ./my_data \
        --strategy smart \
        --frames-per-route 500 \
        --target-speed 15.0
    
    # 启用噪声
    python -m collect_data_new.scripts.run_auto_collection --noise --noise-ratio 0.4
    
    # 启用可视化
    python -m collect_data_new.scripts.run_auto_collection --visualize
    
    # 单天气收集
    python -m collect_data_new.scripts.run_auto_collection --weather ClearNoon
    
    # 多天气收集（使用预设）
    python -m collect_data_new.scripts.run_auto_collection --multi-weather basic
    
    # 多天气收集（自定义列表）
    python -m collect_data_new.scripts.run_auto_collection \
        --weather-list ClearNoon CloudyNoon WetNoon
"""

import os
import sys
import json
import signal
import argparse
import threading

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 全局变量用于信号处理
_collector = None
_force_exit = False
_interrupt_count = 0


def signal_handler(signum, frame):
    """信号处理器，用于优雅地处理 Ctrl+C"""
    global _force_exit, _interrupt_count
    _interrupt_count += 1
    
    if _interrupt_count == 1:
        print("\n\n🛑 收到中断信号 (Ctrl+C)，正在安全退出...")
        print("   (再按一次 Ctrl+C 强制退出)")
        _force_exit = True
        # 抛出 KeyboardInterrupt 让程序正常处理
        raise KeyboardInterrupt()
    else:
        print("\n\n⚠️ 强制退出！")
        os._exit(1)  # 强制退出，不等待清理

from collect_data_new.config import (
    CollectorConfig, NoiseConfig, AnomalyConfig, NPCConfig,
    WeatherConfig, MultiWeatherConfig, RouteConfig, TrafficLightRouteConfig,
    CollisionRecoveryConfig, AdvancedConfig, TrafficLightConfig
)
from collect_data_new.collectors.auto_collector import (
    AutoFullTownCollector, MultiWeatherCollector,
    run_single_weather_collection, run_multi_weather_collection
)
from collect_data_new.core import get_weather_list, WEATHER_COLLECTION_PRESETS


def load_config_file(config_path: str) -> dict:
    """加载配置文件"""
    default_config = {
        'carla_settings': {'host': 'localhost', 'port': 2000, 'town': 'Town01'},
        'traffic_rules': {
            'obey_traffic_rules': False,  # 总开关
            'ignore_traffic_lights': True, 
            'ignore_signs': True, 
            'ignore_vehicles_percentage': 80
        },
        'world_settings': {
            'spawn_npc_vehicles': False, 'num_npc_vehicles': 0,
            'spawn_npc_walkers': False, 'num_npc_walkers': 0,
            'npc_behavior': {
                'obey_traffic_rules': False,  # NPC总开关
                'ignore_traffic_lights': True,
                'ignore_signs': True,
                'ignore_walkers': False
            }
        },
        'weather_settings': {'preset': 'ClearNoon', 'custom': {}},
        'route_generation': {
            'strategy': 'smart', 'min_distance': 50.0, 'max_distance': 500.0,
            'target_routes_ratio': 1.0, 'overlap_threshold': 0.5,
            'turn_priority_ratio': 0.7, 'max_candidates_to_analyze': 0
        },
        'collection_settings': {
            'frames_per_route': 1000, 'save_path': './auto_collected_data',
            'simulation_fps': 20, 'target_speed_kmh': 10.0, 'auto_save_interval': 200
        },
        'noise_settings': {
            'enabled': False, 'lateral_noise': True, 'longitudinal_noise': False,
            'noise_ratio': 0.4, 'max_steer_offset': 0.35, 'max_throttle_offset': 0.2,
            'noise_modes': None
        },
        'collision_recovery': {
            'enabled': True, 'max_collisions_per_route': 99,
            'min_distance_to_destination': 30.0, 'recovery_skip_distance': 25.0
        },
        'anomaly_detection': {
            'enabled': True,
            'spin_detection': {'enabled': True, 'threshold_degrees': 270.0, 'time_window': 3.0},
            'rollover_detection': {'enabled': True, 'pitch_threshold': 45.0, 'roll_threshold': 45.0},
            'stuck_detection': {'enabled': True, 'speed_threshold': 0.5, 'time_threshold': 5.0}
        },
        'advanced_settings': {
            'enable_route_validation': True, 'retry_failed_routes': False,
            'max_retries': 3, 'pause_between_routes': 2
        },
        'multi_weather_settings': {
            'enabled': False, 'weather_preset': 'basic', 'custom_weather_list': []
        },
        'traffic_light_route_settings': {
            'min_traffic_lights': 1, 'max_traffic_lights': 0,
            'traffic_light_radius': 30.0, 'prefer_more_lights': True
        },
        'traffic_light_settings': {
            'enabled': False, 'red_time': 5.0, 'green_time': 10.0, 'yellow_time': 2.0
        },
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            # 深度合并配置
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
    multi_weather_cfg = config.get('multi_weather_settings', {})
    route_cfg = config.get('route_generation', {})
    recovery_cfg = config.get('collision_recovery', {})
    advanced_cfg = config.get('advanced_settings', {})
    
    # 命令行参数覆盖
    host = args.host or carla.get('host', 'localhost')
    port = args.port or carla.get('port', 2000)
    town = args.town or carla.get('town', 'Town01')
    target_speed = args.target_speed or collection.get('target_speed_kmh', 10.0)
    fps = args.fps or collection.get('simulation_fps', 20)
    realtime_sync = collection.get('realtime_sync', False)  # 是否启用实时同步
    frames_per_route = args.frames_per_route or collection.get('frames_per_route', 1000)
    save_path = args.save_path or collection.get('save_path', './auto_collected_data')
    
    # 噪声配置
    noise_enabled = args.noise or noise_cfg.get('enabled', False)
    noise_ratio = args.noise_ratio or noise_cfg.get('noise_ratio', 0.4)
    
    noise = NoiseConfig(
        enabled=noise_enabled,
        lateral_enabled=noise_cfg.get('lateral_noise', True),
        longitudinal_enabled=noise_cfg.get('longitudinal_noise', False),
        noise_ratio=noise_ratio,
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
        stuck_time_threshold=stuck_cfg.get('time_threshold', 5.0),
    )
    
    # NPC配置
    npc_behavior = npc_cfg.get('npc_behavior', {})
    npc = NPCConfig(
        num_vehicles=npc_cfg.get('num_npc_vehicles', 0) if npc_cfg.get('spawn_npc_vehicles') else 0,
        num_walkers=npc_cfg.get('num_npc_walkers', 0) if npc_cfg.get('spawn_npc_walkers') else 0,
        vehicles_obey_traffic_rules=npc_behavior.get('obey_traffic_rules', False),  # NPC总开关
        vehicles_ignore_lights=npc_behavior.get('ignore_traffic_lights', True),
        vehicles_ignore_signs=npc_behavior.get('ignore_signs', True),
        vehicles_ignore_walkers=npc_behavior.get('ignore_walkers', False),
        vehicle_distance=npc_behavior.get('vehicle_distance', 3.0),
        vehicle_speed_difference=npc_behavior.get('vehicle_speed_difference', 30.0),
    )
    
    # 天气配置
    weather_preset = args.weather or weather_cfg.get('preset', 'ClearNoon')
    weather = WeatherConfig(
        preset=weather_preset,
        custom=weather_cfg.get('custom')
    )
    
    # 红绿灯时间配置
    traffic_light_cfg = config.get('traffic_light_settings', {})
    traffic_light = TrafficLightConfig(
        enabled=traffic_light_cfg.get('enabled', False),
        red_time=traffic_light_cfg.get('red_time', 5.0),
        green_time=traffic_light_cfg.get('green_time', 10.0),
        yellow_time=traffic_light_cfg.get('yellow_time', 2.0),
    )
    
    # 多天气配置
    multi_weather = MultiWeatherConfig(
        enabled=multi_weather_cfg.get('enabled', False),
        weather_preset=multi_weather_cfg.get('weather_preset', 'basic'),
        custom_weather_list=multi_weather_cfg.get('custom_weather_list', [])
    )
    
    # 路线配置
    route = RouteConfig(
        strategy=args.strategy or route_cfg.get('strategy', 'smart'),
        min_distance=args.min_distance or route_cfg.get('min_distance', 50.0),
        max_distance=args.max_distance or route_cfg.get('max_distance', 500.0),
        target_routes_ratio=route_cfg.get('target_routes_ratio', 1.0),
        overlap_threshold=route_cfg.get('overlap_threshold', 0.5),
        turn_priority_ratio=route_cfg.get('turn_priority_ratio', 0.7),
        max_candidates_to_analyze=route_cfg.get('max_candidates_to_analyze', 0),
    )
    
    # 红绿灯路线配置（仅当 strategy='traffic_light' 时生效）
    tl_route_cfg = config.get('traffic_light_route_settings', {})
    traffic_light_route = TrafficLightRouteConfig(
        min_traffic_lights=tl_route_cfg.get('min_traffic_lights', 1),
        max_traffic_lights=tl_route_cfg.get('max_traffic_lights', 0),
        traffic_light_radius=tl_route_cfg.get('traffic_light_radius', 30.0),
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
    
    # 可视化：命令行参数优先，否则使用配置文件
    enable_vis = args.visualize or collection.get('enable_visualization', False)
    
    return CollectorConfig(
        host=host,
        port=port,
        town=town,
        obey_traffic_rules=traffic.get('obey_traffic_rules', False),  # 自车总开关
        ignore_traffic_lights=traffic.get('ignore_traffic_lights', True),
        ignore_signs=traffic.get('ignore_signs', True),
        ignore_vehicles_percentage=traffic.get('ignore_vehicles_percentage', 80),
        target_speed=target_speed,
        simulation_fps=fps,
        realtime_sync=realtime_sync,
        save_path=save_path,
        frames_per_route=frames_per_route,
        auto_save_interval=collection.get('auto_save_interval', 200),
        enable_visualization=enable_vis,
        noise=noise,
        anomaly=anomaly,
        npc=npc,
        weather=weather,
        traffic_light=traffic_light,
        multi_weather=multi_weather,
        route=route,
        traffic_light_route=traffic_light_route,
        collision_recovery=collision_recovery,
        advanced=advanced,
    )


def find_config_file(config_path: str) -> str:
    """查找配置文件"""
    if os.path.isabs(config_path) and os.path.exists(config_path):
        return config_path
    
    # 尝试多个位置
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
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    # Windows 上 SIGBREAK 对应 Ctrl+Break
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    parser = argparse.ArgumentParser(
        description='全自动CARLA数据收集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
天气预设列表:
  basic      - 基础组合（4种）：ClearNoon, CloudyNoon, ClearSunset, ClearNight
  all_noon   - 所有正午天气（7种）
  all_sunset - 所有日落天气（7种）
  all_night  - 所有夜晚天气（7种）
  clear_all  - 所有晴朗天气（3种）
  rain_all   - 所有雨天（9种）
  full       - 完整组合（13种）
  complete   - 所有天气（22种）
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
    parser.add_argument('--strategy', type=str, choices=['smart', 'exhaustive', 'traffic_light'],
                        help='路线生成策略 (traffic_light=红绿灯路口优先)')
    parser.add_argument('--frames-per-route', type=int, help='每条路线最大帧数')
    parser.add_argument('--target-speed', type=float, help='目标速度 (km/h)')
    parser.add_argument('--fps', type=int, help='模拟帧率')
    
    # 路线参数
    parser.add_argument('--min-distance', type=float, help='最小路线距离')
    parser.add_argument('--max-distance', type=float, help='最大路线距离')
    parser.add_argument('--route-cache', type=str, help='路线缓存文件路径')
    
    # 噪声
    parser.add_argument('--noise', action='store_true', help='启用噪声注入')
    parser.add_argument('--noise-ratio', type=float, help='噪声时间占比')
    
    # 可视化
    parser.add_argument('--visualize', action='store_true', help='启用实时可视化')
    
    # 天气
    parser.add_argument('--weather', type=str, help='单一天气预设名称')
    parser.add_argument('--multi-weather', type=str, 
                        choices=list(WEATHER_COLLECTION_PRESETS.keys()),
                        help='多天气收集预设')
    parser.add_argument('--weather-list', nargs='+', 
                        help='自定义天气列表，如: ClearNoon CloudyNoon WetNoon')
    
    args = parser.parse_args()
    
    # 查找配置文件
    config_path = find_config_file(args.config)
    
    # 加载配置
    config_dict = load_config_file(config_path)
    
    # 创建收集器配置
    collector_config = create_collector_config(config_dict, args)
    
    # 确定天气列表
    weather_list = None
    
    # 打印多天气配置状态
    print(f"\n📋 多天气配置状态:")
    print(f"   - multi_weather.enabled = {collector_config.multi_weather.enabled}")
    print(f"   - multi_weather.weather_preset = '{collector_config.multi_weather.weather_preset}'")
    print(f"   - multi_weather.custom_weather_list = {collector_config.multi_weather.custom_weather_list}")
    print(f"   - 命令行 --multi-weather = {args.multi_weather}")
    print(f"   - 命令行 --weather-list = {args.weather_list}")
    
    # 优先级: 命令行 --weather-list > 命令行 --multi-weather > 配置文件
    if args.weather_list:
        weather_list = args.weather_list
        print(f"\n🌤️ 天气来源: 命令行 --weather-list")
        print(f"   天气列表: {weather_list}")
    elif args.multi_weather:
        weather_list = get_weather_list(args.multi_weather)
        print(f"\n🌤️ 天气来源: 命令行 --multi-weather (预设: {args.multi_weather})")
        print(f"   天气列表: {weather_list}")
    elif collector_config.multi_weather.enabled:
        weather_list = collector_config.multi_weather.get_weather_list()
        print(f"\n🌤️ 天气来源: JSON配置文件 (multi_weather_settings)")
        print(f"   天气列表: {weather_list}")
    else:
        print(f"\n🌤️ 天气来源: 单天气模式 (multi_weather.enabled=False)")
        print(f"   使用天气: {collector_config.weather.preset}")
    
    # 运行收集
    save_path = collector_config.save_path
    
    # 调试信息
    print(f"\n🔍 调试信息:")
    print(f"   - weather_list = {weather_list}")
    print(f"   - weather_list 长度 = {len(weather_list) if weather_list else 0}")
    print(f"   - save_path = {save_path}")
    
    if weather_list and len(weather_list) > 1:
        # 多天气收集
        print(f"\n✅ 收集模式: 多天气轮换 ({len(weather_list)} 种天气)")
        print(f"   天气列表: {weather_list}")
        run_multi_weather_collection(
            config=collector_config,
            weather_list=weather_list,
            base_save_path=save_path,
            strategy=collector_config.route.strategy
        )
    else:
        # 单天气收集
        print(f"\n✅ 收集模式: 单天气")
        if weather_list:
            print(f"   ⚠️ weather_list 只有 {len(weather_list)} 个元素: {weather_list}")
        collector = AutoFullTownCollector(collector_config)
        collector.run(
            save_path=save_path,
            strategy=collector_config.route.strategy,
            route_cache_path=args.route_cache
        )


if __name__ == '__main__':
    main()
