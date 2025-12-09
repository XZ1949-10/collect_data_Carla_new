#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant
日期: 2025-12-01
说明: 全自动Town01场景数据收集器
      自动遍历所有生成点组合，收集完整的Town01场景数据
      无需人工干预，智能选择路线并自动保存
'''

import os
import sys
import time
import random
import numpy as np
import json
import cv2
from datetime import datetime

# 导入基类
from base_collector import BaseDataCollector, AGENTS_AVAILABLE

import carla

# 导入agents模块
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
except ImportError:
    pass


class AutoFullTownCollector(BaseDataCollector):
    """全自动Town01数据收集器"""
    
    def __init__(self, host='localhost', port=2000, town='Town01',
                 ignore_traffic_lights=True, ignore_signs=True,
                 ignore_vehicles_percentage=80, target_speed=10.0,
                 simulation_fps=20, spawn_npc_vehicles=False, num_npc_vehicles=0,
                 spawn_npc_walkers=False, num_npc_walkers=0, weather_config=None):
        
        super().__init__(host, port, town, ignore_traffic_lights, ignore_signs,
                        ignore_vehicles_percentage, target_speed, simulation_fps)
        
        # NPC配置
        self.spawn_npc_vehicles = spawn_npc_vehicles
        self.num_npc_vehicles = num_npc_vehicles
        self.spawn_npc_walkers = spawn_npc_walkers
        self.num_npc_walkers = num_npc_walkers
        self.weather_config = weather_config or {}
        
        # NPC列表
        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []
        
        # 路线规划
        self.spawn_points = []
        self.route_planner = None
        
        # 收集策略
        self.min_distance = 50.0
        self.max_distance = 500.0
        self.frames_per_route = 1000
        self.target_routes_ratio = 1.0  # 路线选择比例（0-1），1.0=全选
        self.overlap_threshold = 0.5
        self.turn_priority_ratio = 0.7  # 转弯路线占比（0-1），0.7=70%转弯路线+30%直行路线
        self.auto_save_interval = 200   # 自动保存间隔（帧数）
        
        # 高级设置
        self.enable_route_validation = True   # 是否启用路线验证
        self.retry_failed_routes = False      # 是否重试失败的路线
        self.max_retries = 3                  # 最大重试次数
        self.pause_between_routes = 2         # 路线之间的暂停时间（秒）
        
        # 路线分析参数
        self.max_candidates_to_analyze = 0    # 最多分析的候选路线数（0=不限制）
        
        # 统计
        self.total_routes_attempted = 0
        self.total_routes_completed = 0
        self.total_frames_collected = 0
        self.failed_routes = []
        
        self.route_generation_strategy = 'smart'
        
        # 内部收集器引用
        self._inner_collector = None
        
        # 噪声配置（会传递给内部收集器）
        self.noise_enabled = False
        self.lateral_noise_enabled = True
        self.longitudinal_noise_enabled = False
        
        # 噪声参数（直观参数）
        self.noise_ratio = 0.4           # 噪声时间占比
        self.max_steer_offset = 0.35     # 最大转向偏移
        self.max_throttle_offset = 0.2   # 最大油门偏移
        self.noise_mode_config = None
        
        # 碰撞恢复配置
        self.collision_recovery_enabled = True      # 是否启用碰撞恢复
        self.max_collisions_per_route = 99          # 单条路线最大碰撞次数（99=基本不限制）
        self.min_distance_to_destination = 30.0     # 距终点小于此距离不恢复
        self.recovery_skip_distance = 25.0          # 恢复时跳过的距离（米），跳过碰撞区域
        
        # 当前路线的waypoints（用于碰撞恢复）
        self._current_route_waypoints = []          # 当前路线的完整waypoints列表
        
        # 异常检测配置
        self.anomaly_detection_enabled = True       # 是否启用异常检测
        self.spin_detection_enabled = True          # 是否检测打转
        self.spin_threshold_degrees = 270.0         # 打转角度阈值
        self.spin_time_window = 3.0                 # 打转检测时间窗口
        self.rollover_detection_enabled = True      # 是否检测翻车
        self.rollover_pitch_threshold = 45.0        # 翻车俯仰角阈值
        self.rollover_roll_threshold = 45.0         # 翻车横滚角阈值
        self.stuck_detection_enabled = True         # 是否检测卡住
        self.stuck_speed_threshold = 0.5            # 卡住速度阈值
        self.stuck_time_threshold = 5.0             # 卡住时间阈值
        
        # 当前路线的终点（用于碰撞恢复时重新规划）
        self._current_destination = None
        self._current_destination_index = None
    
    def connect(self):
        """连接到CARLA服务器（扩展版）"""
        print("\n" + "="*70)
        print("🚗 全自动Town01数据收集器")
        print("="*70)
        print(f"正在连接到CARLA服务器 {self.host}:{self.port}...")
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(120.0)  # 增加超时时间到120秒，避免路线切换时超时
        
        self.world = self.client.get_world()
        current_map_name = self.world.get_map().name.split('/')[-1]
        
        if current_map_name != self.town:
            print(f"正在加载地图 {self.town}...")
            self.world = self.client.load_world(self.town)
        else:
            print(f"✅ 已连接到地图 {self.town}")
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"✅ 成功连接！共找到 {len(self.spawn_points)} 个生成点")
        
        self._print_config()
        self._set_weather()
        
        if self.spawn_npc_vehicles and self.num_npc_vehicles > 0:
            self._spawn_npc_vehicles()
        if self.spawn_npc_walkers and self.num_npc_walkers > 0:
            self._spawn_npc_walkers()
        
        if AGENTS_AVAILABLE:
            try:
                self.route_planner = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2.0)
                print("✅ 路径规划器初始化成功")
            except Exception as e:
                print(f"⚠️  路径规划器初始化失败: {e}")
        print()
    
    def _print_config(self):
        """打印配置信息"""
        print(f"\n📋 配置信息:")
        print(f"  • 忽略红绿灯: {'✅' if self.ignore_traffic_lights else '❌'}")
        print(f"  • 忽略停车标志: {'✅' if self.ignore_signs else '❌'}")
        print(f"  • 目标速度: {self.target_speed:.1f} km/h")
        print(f"  • 模拟帧率: {self.simulation_fps} FPS")
        if self.spawn_npc_vehicles:
            print(f"  • NPC车辆: {self.num_npc_vehicles}")
        if self.spawn_npc_walkers:
            print(f"  • NPC行人: {self.num_npc_walkers}")
    
    def _set_weather(self):
        """设置天气"""
        if not self.weather_config:
            return
        
        preset = self.weather_config.get('preset')
        weather_presets = {
            # 正午天气
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            # 日落天气
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
            'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
            'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
            'MidRainSunset': carla.WeatherParameters.MidRainSunset,
            'HardRainSunset': carla.WeatherParameters.HardRainSunset,
            # 夜晚天气
            'ClearNight': carla.WeatherParameters.ClearNight,
            'CloudyNight': carla.WeatherParameters.CloudyNight,
            'WetNight': carla.WeatherParameters.WetNight,
            'WetCloudyNight': carla.WeatherParameters.WetCloudyNight,
            'SoftRainNight': carla.WeatherParameters.SoftRainNight,
            'MidRainyNight': carla.WeatherParameters.MidRainyNight,
            'HardRainNight': carla.WeatherParameters.HardRainNight,
            # 特殊天气
            'DustStorm': carla.WeatherParameters.DustStorm,
        }
        
        if preset and preset in weather_presets:
            self.world.set_weather(weather_presets[preset])
            print(f"  🌤️ 天气: {preset}")
        elif preset is None or preset == '':
            # 使用自定义天气参数
            custom = self.weather_config.get('custom', {})
            if custom:
                weather = carla.WeatherParameters(
                    cloudiness=custom.get('cloudiness', 0.0),
                    precipitation=custom.get('precipitation', 0.0),
                    precipitation_deposits=custom.get('precipitation_deposits', 0.0),
                    wind_intensity=custom.get('wind_intensity', 0.0),
                    sun_azimuth_angle=custom.get('sun_azimuth_angle', 0.0),
                    sun_altitude_angle=custom.get('sun_altitude_angle', 75.0),
                    fog_density=custom.get('fog_density', 0.0),
                    fog_distance=custom.get('fog_distance', 0.0),
                    wetness=custom.get('wetness', 0.0)
                )
                self.world.set_weather(weather)
                print(f"  🌤️ 天气: 自定义参数")
                print(f"     云量: {custom.get('cloudiness', 0.0)}, 降水: {custom.get('precipitation', 0.0)}")
        elif preset:
            print(f"  ⚠️ 未知天气预设: {preset}，使用默认天气")
    
    def _spawn_npc_vehicles(self):
        """生成NPC车辆
        
        注意：为避免NPC车辆占用数据收集车辆的生成点，
        NPC车辆从生成点列表的后半部分开始生成。
        """
        print(f"\n🚗 正在生成 {self.num_npc_vehicles} 辆NPC车辆...")
        
        blueprints = [x for x in self.blueprint_library.filter('vehicle.*')
                      if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.world.get_map().get_spawn_points()
        
        # 从后半部分生成点开始，避免占用常用的起点/终点
        # 保留前半部分给数据收集车辆使用
        half_idx = len(spawn_points) // 2
        npc_spawn_points = spawn_points[half_idx:]
        random.shuffle(npc_spawn_points)
        
        for i in range(min(self.num_npc_vehicles, len(npc_spawn_points))):
            bp = random.choice(blueprints)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            
            npc = self.world.try_spawn_actor(bp, npc_spawn_points[i])
            if npc:
                npc.set_autopilot(True)
                self.npc_vehicles.append(npc)
        
        print(f"✅ 成功生成 {len(self.npc_vehicles)} 辆NPC车辆（使用后半部分生成点）")
    
    def _spawn_npc_walkers(self):
        """生成NPC行人"""
        print(f"\n🚶 正在生成 {self.num_npc_walkers} 个NPC行人...")
        
        walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        spawn_points = []
        
        for _ in range(self.num_npc_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(location=loc))
        
        batch = [carla.command.SpawnActor(random.choice(walker_bps), sp) for sp in spawn_points]
        results = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in results if not r.error]
        
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
        results = self.client.apply_batch_sync(batch, True)
        self.walker_controllers = [r.actor_id for r in results if not r.error]
        
        self.world.tick()
        for ctrl in self.world.get_actors(self.walker_controllers):
            ctrl.start()
            ctrl.go_to_location(self.world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.0 + random.random())
        
        self.npc_walkers = list(self.world.get_actors(walker_ids))
        print(f"✅ 成功生成 {len(self.npc_walkers)} 个NPC行人")
    
    def _cleanup_npcs(self):
        """清理NPC"""
        for ctrl_id in self.walker_controllers:
            try:
                ctrl = self.world.get_actor(ctrl_id)
                if ctrl:
                    ctrl.stop()
                    ctrl.destroy()
            except:
                pass
        
        for walker in self.npc_walkers:
            try:
                walker.destroy()
            except:
                pass
        
        for vehicle in self.npc_vehicles:
            try:
                vehicle.destroy()
            except:
                pass
        
        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []
    
    def generate_route_pairs(self, cache_path=None):
        """生成路线对（支持缓存）
        
        参数:
            cache_path: 缓存文件路径，如果存在则直接读取，否则生成后保存
        """
        print("\n" + "="*70)
        print("🛣️ 生成路线对")
        print("="*70)
        
        # 尝试从缓存加载
        if cache_path and os.path.exists(cache_path):
            route_pairs = self._load_routes_from_cache(cache_path)
            if route_pairs:
                print(f"✅ 从缓存加载了 {len(route_pairs)} 条路线")
                self._print_route_statistics(route_pairs)
                return route_pairs
        
        # 生成新路线
        if self.route_generation_strategy == 'smart':
            route_pairs = self._generate_smart_routes()
        else:
            route_pairs = self._generate_exhaustive_routes()
        
        if route_pairs:
            self._print_route_statistics(route_pairs)
            # 保存到缓存
            if cache_path:
                self._save_routes_to_cache(route_pairs, cache_path)
        
        return route_pairs
    
    def _load_routes_from_cache(self, cache_path):
        """从缓存文件加载路线"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证缓存是否匹配当前配置
            config = data.get('config', {})
            if (config.get('town') != self.town or
                config.get('min_distance') != self.min_distance or
                config.get('max_distance') != self.max_distance or
                config.get('strategy') != self.route_generation_strategy):
                print(f"⚠️ 缓存配置不匹配，重新生成路线")
                return None
            
            routes = data.get('routes', [])
            # 转换为元组列表
            return [(r['start'], r['end'], r['distance']) for r in routes]
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
            return None
    
    def _save_routes_to_cache(self, route_pairs, cache_path):
        """保存路线到缓存文件"""
        try:
            data = {
                'config': {
                    'town': self.town,
                    'min_distance': self.min_distance,
                    'max_distance': self.max_distance,
                    'strategy': self.route_generation_strategy,
                    'overlap_threshold': self.overlap_threshold,
                    'turn_priority_ratio': self.turn_priority_ratio,
                    'target_routes_ratio': self.target_routes_ratio
                },
                'routes': [
                    {'start': s, 'end': e, 'distance': d}
                    for s, e, d in route_pairs
                ],
                'generated_at': datetime.now().isoformat(),
                'total_routes': len(route_pairs)
            }
            
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 路线已缓存到: {cache_path}")
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
    
    def _generate_smart_routes(self):
        """
        智能路线生成
        
        处理顺序：
        1. _analyze_candidate_routes: 按距离筛选候选路线
        2. _deduplicate_routes: 按 overlap_threshold 去重（先去重）
        3. _select_balanced_routes: 按 turn_priority_ratio 和 target_routes_ratio 选择（后选择）
        """
        print(f"策略: 🧠 智能选择")
        
        if not AGENTS_AVAILABLE or self.route_planner is None:
            return self._generate_basic_routes()
        
        # 1. 按距离筛选候选路线
        candidates = self._analyze_candidate_routes()
        if not candidates:
            return []
        
        # 2. 先去重（保证路线多样性）
        deduplicated = self._deduplicate_routes(candidates)
        if not deduplicated:
            return []
        
        # 3. 后按比例选择（在去重后的路线中选择）
        selected = self._select_balanced_routes(deduplicated)
        return selected
    
    def _analyze_candidate_routes(self):
        """
        分析候选路线
        
        使用真实路径距离进行筛选，不使用直线距离预筛选（因为直线距离与实际路径距离关系不稳定）
        """
        print("\n🔍 分析候选路线...")
        print(f"  📏 路径距离范围: {self.min_distance:.0f}m ~ {self.max_distance:.0f}m")
        
        candidates = []
        command_map = {'LANEFOLLOW': 2, 'LEFT': 3, 'RIGHT': 4, 'STRAIGHT': 5,
                       'CHANGELANELEFT': 2, 'CHANGELANERIGHT': 2}
        
        num_spawns = len(self.spawn_points)
        total_pairs = num_spawns * (num_spawns - 1)
        
        # 从配置获取采样参数（0=不限制）
        max_candidates_to_check = getattr(self, 'max_candidates_to_analyze', 0)
        use_sampling = max_candidates_to_check > 0 and total_pairs > max_candidates_to_check
        
        if use_sampling:
            print(f"  ⚡ 组合数过多 ({total_pairs})，随机采样 {max_candidates_to_check} 条进行分析...")
            all_pairs = [(i, j) for i in range(num_spawns) for j in range(num_spawns) if i != j]
            random.shuffle(all_pairs)
            pairs_to_check = all_pairs[:max_candidates_to_check]
        else:
            pairs_to_check = [(i, j) for i in range(num_spawns) for j in range(num_spawns) if i != j]
            print(f"  📋 共 {len(pairs_to_check)} 个起点-终点组合待分析")
        
        checked = 0
        filtered_by_distance = 0
        last_progress = 0
        
        for start_idx, end_idx in pairs_to_check:
            checked += 1
            
            # 每10%显示进度
            progress = int(checked / len(pairs_to_check) * 100)
            if progress >= last_progress + 10:
                print(f"  📊 进度: {progress}% ({checked}/{len(pairs_to_check)}), "
                      f"有效: {len(candidates)}, 距离不符: {filtered_by_distance}")
                last_progress = progress
            
            start_loc = self.spawn_points[start_idx].location
            end_loc = self.spawn_points[end_idx].location
            
            try:
                # 直接使用路径规划获取真实路径
                route = self.route_planner.trace_route(start_loc, end_loc)
                if not route or len(route) < 2:
                    continue
                
                # 计算真实路径距离和命令
                commands = {2: 0, 3: 0, 4: 0, 5: 0}
                waypoints = []
                route_distance = 0.0
                prev_cmd = None
                
                for i, (wp, road_option) in enumerate(route):
                    if i > 0:
                        route_distance += wp.transform.location.distance(route[i-1][0].transform.location)
                    waypoints.append((wp.transform.location.x, wp.transform.location.y))
                    
                    cmd_name = road_option.name if hasattr(road_option, 'name') else str(road_option)
                    cmd = command_map.get(cmd_name, 2)
                    if cmd != prev_cmd:
                        commands[cmd] += 1
                        prev_cmd = cmd
                
                # 使用真实路径距离筛选
                if route_distance < self.min_distance or route_distance > self.max_distance:
                    filtered_by_distance += 1
                    continue
                
                candidates.append({
                    'start_idx': start_idx, 'end_idx': end_idx,
                    'route_distance': route_distance,
                    'commands': commands, 'waypoints': waypoints,
                    'turn_count': commands[3] + commands[4]
                })
                
                # 注意：不再提前结束，因为 target_routes_ratio 是基于最终筛选结果的比例
                # 需要分析所有候选路线才能正确计算比例
                    
            except Exception:
                pass
        
        print(f"  ✅ 分析完成: 有效路线 {len(candidates)} 条, 距离不符 {filtered_by_distance} 条")
        return candidates
    
    def _select_balanced_routes(self, candidates):
        """
        按转弯比例和选择比例筛选路线
        
        参数说明：
        - turn_priority_ratio: 转弯路线占比（0-1），0.7=70%转弯+30%直行
        - target_routes_ratio: 路线选择比例（0-1），从符合条件的路线中选择多少比例
        
        返回:
            元组列表 [(start_idx, end_idx, distance), ...]
        """
        # 分离转弯路线和非转弯路线
        turn_routes = [c for c in candidates if c.get('turn_count', 0) > 0]
        straight_routes = [c for c in candidates if c.get('turn_count', 0) == 0]
        
        # 按转弯次数排序（转弯多的优先）
        turn_routes.sort(key=lambda x: (-x.get('turn_count', 0), -x.get('route_distance', 0)))
        # 按距离排序（距离长的优先，数据更多）
        straight_routes.sort(key=lambda x: -x.get('route_distance', 0))
        
        print(f"  📊 去重后路线: 转弯 {len(turn_routes)} 条, 直行 {len(straight_routes)} 条")
        
        turn_ratio = max(0.0, min(1.0, self.turn_priority_ratio))
        
        # 先按 turn_priority_ratio 计算转弯和直行的目标数量
        # 以数量较多的一方为基准，按比例计算另一方
        if turn_ratio >= 0.5:
            # 转弯优先：选择所有转弯路线，按比例计算直行路线数量
            max_turn = len(turn_routes)
            if turn_ratio < 1.0:
                max_straight = int(max_turn * (1 - turn_ratio) / turn_ratio)
                max_straight = min(max_straight, len(straight_routes))
            else:
                max_straight = 0
        else:
            # 直行优先：选择所有直行路线，按比例计算转弯路线数量
            max_straight = len(straight_routes)
            if turn_ratio > 0.0:
                max_turn = int(max_straight * turn_ratio / (1 - turn_ratio))
                max_turn = min(max_turn, len(turn_routes))
            else:
                max_turn = 0
        
        # 应用 target_routes_ratio 进一步筛选
        select_ratio = max(0.0, min(1.0, self.target_routes_ratio))
        actual_turn_count = int(max_turn * select_ratio)
        actual_straight_count = int(max_straight * select_ratio)
        
        # 确保至少选择1条（如果有的话）
        if select_ratio > 0:
            if max_turn > 0 and actual_turn_count == 0:
                actual_turn_count = 1
            if max_straight > 0 and actual_straight_count == 0:
                actual_straight_count = 1
        
        # 选择路线（字典列表）
        selected_dicts = turn_routes[:actual_turn_count] + straight_routes[:actual_straight_count]
        
        if selected_dicts:
            actual_turn_ratio = actual_turn_count / len(selected_dicts)
            print(f"  ✅ 最终选择: 转弯 {actual_turn_count} 条 ({actual_turn_ratio:.1%}), "
                  f"直行 {actual_straight_count} 条 ({1-actual_turn_ratio:.1%})")
            print(f"     选择比例: {select_ratio:.0%} (共 {len(selected_dicts)} 条)")
        
        # 转换为元组列表并随机打乱
        result = [(r['start_idx'], r['end_idx'], r.get('route_distance', 0)) for r in selected_dicts]
        random.shuffle(result)
        return result
    
    def _deduplicate_routes(self, routes):
        """
        路径去重
        
        参数:
            routes: 路线字典列表
        返回:
            去重后的路线字典列表（保留完整信息供后续处理）
        """
        if len(routes) <= 1:
            return routes
        
        # 按转弯次数排序（转弯多的优先保留）
        routes_copy = routes.copy()
        routes_copy.sort(key=lambda x: (-x.get('turn_count', 0), -x.get('route_distance', 0)))
        
        deduplicated = []
        removed_count = 0
        
        for route in routes_copy:
            is_overlapping = False
            route_wps = route.get('waypoints', [])
            
            if route_wps:
                for selected in deduplicated:
                    sel_wps = selected.get('waypoints', [])
                    if sel_wps and self._calculate_overlap(route_wps, sel_wps) > self.overlap_threshold:
                        is_overlapping = True
                        removed_count += 1
                        break
            
            if not is_overlapping:
                deduplicated.append(route)
        
        print(f"  🔄 去重完成: {len(routes)} → {len(deduplicated)} 条 (移除 {removed_count} 条重叠路线)")
        return deduplicated
    
    def _calculate_overlap(self, wps1, wps2, grid_size=10.0):
        """计算路径重叠度"""
        def to_grid(wps):
            return set((int(x / grid_size), int(y / grid_size)) for x, y in wps)
        
        g1, g2 = to_grid(wps1), to_grid(wps2)
        if not g1 or not g2:
            return 0.0
        return len(g1 & g2) / len(g1 | g2)
    
    def _generate_basic_routes(self):
        """基础路线生成（降级方案，仅在 agents 模块不可用时使用）
        
        警告：此方法使用直线距离估算，而非实际路径距离。
        直线距离通常是实际路径距离的 0.6-0.8 倍，筛选结果可能与预期有偏差。
        建议安装 agents 模块以获得准确的路径距离筛选。
        """
        print("  ⚠️  agents模块不可用，使用直线距离估算（建议安装agents模块以获得准确筛选）")
        route_pairs = []
        for start_idx, sp in enumerate(self.spawn_points):
            valid_ends = []
            for end_idx, ep in enumerate(self.spawn_points):
                if start_idx != end_idx:
                    d = self._calculate_distance(sp.location, ep.location)
                    # 直线距离通常是路径距离的0.6-0.8倍，适当放宽筛选范围
                    # 最小距离乘0.6（允许更短的直线距离），最大距离乘0.8（避免实际路径超限）
                    if self.min_distance * 0.6 <= d <= self.max_distance * 0.8:
                        valid_ends.append((end_idx, d))
            
            if valid_ends:
                valid_ends.sort(key=lambda x: x[1])
                for idx in [0, len(valid_ends)//2, len(valid_ends)-1]:
                    if idx < len(valid_ends):
                        route_pairs.append((start_idx, valid_ends[idx][0], valid_ends[idx][1]))
        
        random.shuffle(route_pairs)
        return route_pairs
    
    def _generate_exhaustive_routes(self):
        """穷举路线生成 - 生成所有满足距离条件的起点-终点组合
        
        使用路径规划器计算真实路径距离进行筛选，不使用直线距离预筛选。
        当路径规划器不可用时，降级使用直线距离估算。
        """
        print(f"策略: 📋 穷举模式")
        
        route_pairs = []
        num_spawns = len(self.spawn_points)
        total_pairs = num_spawns * (num_spawns - 1)
        
        print(f"  正在分析 {total_pairs} 个起点-终点组合...")
        print(f"  📏 路径距离范围: {self.min_distance:.0f}m ~ {self.max_distance:.0f}m")
        
        checked = 0
        unreachable = 0
        filtered_by_distance = 0
        
        for start_idx, sp in enumerate(self.spawn_points):
            for end_idx, ep in enumerate(self.spawn_points):
                if start_idx == end_idx:
                    continue
                
                checked += 1
                
                # 优先使用路径规划器计算真实路径距离
                if AGENTS_AVAILABLE and self.route_planner is not None:
                    try:
                        route = self.route_planner.trace_route(sp.location, ep.location)
                        if route and len(route) >= 2:
                            # 计算实际路径距离
                            route_distance = sum(
                                route[i][0].transform.location.distance(route[i-1][0].transform.location)
                                for i in range(1, len(route))
                            )
                            if self.min_distance <= route_distance <= self.max_distance:
                                route_pairs.append((start_idx, end_idx, route_distance))
                            else:
                                filtered_by_distance += 1
                        else:
                            unreachable += 1
                    except:
                        unreachable += 1
                else:
                    # 降级方案：使用直线距离估算（放宽范围）
                    d = self._calculate_distance(sp.location, ep.location)
                    if self.min_distance * 0.6 <= d <= self.max_distance * 0.8:
                        route_pairs.append((start_idx, end_idx, d))
                    else:
                        filtered_by_distance += 1
            
            # 显示进度
            if (start_idx + 1) % 50 == 0 or start_idx == num_spawns - 1:
                print(f"  进度: {start_idx + 1}/{num_spawns}, "
                      f"有效: {len(route_pairs)}, 距离不符: {filtered_by_distance}, 不可达: {unreachable}")
        
        print(f"  ✅ 穷举完成，共找到 {len(route_pairs)} 条有效路线")
        
        # 按 target_routes_ratio 比例选择
        select_ratio = max(0.0, min(1.0, self.target_routes_ratio))
        if select_ratio < 1.0:
            random.shuffle(route_pairs)
            target_count = max(1, int(len(route_pairs) * select_ratio))
            route_pairs = route_pairs[:target_count]
            print(f"  📊 按比例选择 {select_ratio:.0%}，共 {len(route_pairs)} 条路线")
        else:
            random.shuffle(route_pairs)
        
        return route_pairs
    
    def _calculate_distance(self, loc1, loc2):
        """计算两点距离"""
        return np.sqrt((loc2.x - loc1.x)**2 + (loc2.y - loc1.y)**2)
    
    def _print_route_statistics(self, route_pairs):
        """打印路线统计"""
        distances = [d for _, _, d in route_pairs]
        print(f"\n📊 路线统计:")
        print(f"  • 总路线数: {len(route_pairs)}")
        print(f"  • 平均距离: {np.mean(distances):.1f}m")
        print(f"  • 预计耗时: {len(route_pairs) * 2:.0f}分钟")

    def collect_route_data(self, start_idx, end_idx, save_path):
        """收集单条路线数据（支持碰撞恢复）
        
        碰撞恢复逻辑：
        1. 碰撞后完全清理所有资源
        2. 从路线waypoints中找恢复点（而非全局spawn_points）
        3. 在恢复点位置重新生成车辆，继续沿原路线行驶
        """
        print(f"\n{'='*70}")
        print(f"📊 收集路线: {start_idx} → {end_idx}")
        print(f"{'='*70}")
        
        # 保存终点信息
        self._current_destination_index = end_idx
        self._current_destination = self.spawn_points[end_idx].location if end_idx < len(self.spawn_points) else None
        
        # 预先计算并保存完整路线的waypoints（用于碰撞恢复）
        self._current_route_waypoints = []
        if AGENTS_AVAILABLE and self.route_planner is not None:
            try:
                route = self.route_planner.trace_route(
                    self.spawn_points[start_idx].location,
                    self.spawn_points[end_idx].location
                )
                if route:
                    self._current_route_waypoints = list(route)
                    print(f"📍 路线waypoints: {len(self._current_route_waypoints)} 个点")
            except Exception as e:
                print(f"⚠️ 获取路线waypoints失败: {e}")
        
        # 第一次收集使用spawn_index
        current_spawn_transform = None  # None表示使用spawn_index
        current_start_idx = start_idx
        collision_count = 0
        total_saved_frames = 0
        
        while True:
            # 每次循环都是一个完整的收集周期（从创建到销毁）
            result = self._do_single_collection(
                current_start_idx, end_idx, save_path,
                spawn_transform=current_spawn_transform
            )
            
            total_saved_frames += result.get('saved_frames', 0)
            
            # 检查是否需要恢复
            if result.get('need_recovery') and self.collision_recovery_enabled:
                collision_count += 1
                
                if collision_count >= self.max_collisions_per_route:
                    print(f"  ⚠️ 碰撞次数达到上限（{self.max_collisions_per_route}次），终止本路线")
                    break
                
                recovery_transform = result.get('recovery_transform')
                if recovery_transform is not None:
                    print(f"\n🔄 碰撞恢复：从路线waypoint恢复（终点不变）")
                    current_spawn_transform = recovery_transform
                    current_start_idx = None  # 使用transform而非index
                    time.sleep(1.0)  # 等待资源完全释放
                    continue
                else:
                    print(f"  ⚠️ 无法恢复，终止本路线")
                    break
            else:
                # 正常完成或无法恢复
                break
        
        print(f"\n📊 路线总计: {total_saved_frames} 帧, 碰撞 {collision_count} 次")
        return result.get('success', False) or total_saved_frames > 0
    
    def _do_single_collection(self, start_idx, end_idx, save_path, spawn_transform=None):
        """执行单次收集（从创建车辆到销毁）
        
        这是一个完整的收集周期，包括：
        1. 创建内部收集器
        2. 生成车辆和传感器（支持从transform或spawn_index生成）
        3. 收集数据
        4. 清理所有资源
        
        参数:
            start_idx: 起点spawn_index（spawn_transform为None时使用）
            end_idx: 终点spawn_index
            save_path: 数据保存路径
            spawn_transform: 车辆生成位置的transform（用于碰撞恢复，None则使用start_idx）
        
        返回:
            dict: 收集结果
        """
        result = {'success': False, 'saved_frames': 0, 'need_recovery': False, 'recovery_transform': None}
        
        try:
            # 创建内部收集器
            from command_based_data_collection import CommandBasedDataCollector
            self._inner_collector = CommandBasedDataCollector(
                host=self.host, port=self.port, town=self.town,
                ignore_traffic_lights=self.ignore_traffic_lights,
                ignore_signs=self.ignore_signs,
                ignore_vehicles_percentage=self.ignore_vehicles_percentage,
                target_speed=self.target_speed,
                simulation_fps=self.simulation_fps
            )
            
            # 复用连接
            self._inner_collector.client = self.client
            self._inner_collector.world = self.world
            self._inner_collector.blueprint_library = self.blueprint_library
            
            # 设置同步模式
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.simulation_fps
                self.world.apply_settings(settings)
            
            # 生成车辆（支持从transform或spawn_index生成）
            if spawn_transform is not None:
                # 碰撞恢复：从指定transform生成车辆
                if not self._spawn_vehicle_at_transform(spawn_transform, end_idx):
                    return result
            else:
                # 正常启动：从spawn_index生成车辆
                if not self._inner_collector.spawn_vehicle(start_idx, end_idx):
                    return result
            
            # 设置传感器
            self._inner_collector.setup_camera()
            self._inner_collector.setup_collision_sensor()
            
            # 等待传感器初始化
            time.sleep(0.5)
            for _ in range(10):
                self.world.tick()
            time.sleep(0.3)
            
            # 配置噪声
            self._inner_collector.configure_noise(
                enabled=self.noise_enabled,
                lateral_enabled=self.lateral_noise_enabled,
                longitudinal_enabled=self.longitudinal_noise_enabled,
                noise_ratio=self.noise_ratio,
                max_steer_offset=self.max_steer_offset,
                max_throttle_offset=self.max_throttle_offset,
                noise_modes=self.noise_mode_config
            )
            self._inner_collector.reset_noisers()
            
            # 配置异常检测
            self._inner_collector.configure_anomaly_detection(
                enabled=self.anomaly_detection_enabled,
                spin_enabled=self.spin_detection_enabled,
                rollover_enabled=self.rollover_detection_enabled,
                stuck_enabled=self.stuck_detection_enabled,
                spin_threshold=self.spin_threshold_degrees,
                spin_time_window=self.spin_time_window,
                rollover_pitch=self.rollover_pitch_threshold,
                rollover_roll=self.rollover_roll_threshold,
                stuck_speed=self.stuck_speed_threshold,
                stuck_time=self.stuck_time_threshold
            )
            self._inner_collector.reset_anomaly_state()
            
            # 执行收集
            result = self._auto_collect(save_path)
            return result
            
        except Exception as e:
            print(f"❌ 收集出错: {e}")
            import traceback
            traceback.print_exc()
            return result
        finally:
            # 无论如何都要清理资源
            self._cleanup_inner_collector()
    
    def _reset_sync_mode(self):
        """重置同步模式（用于错误恢复）"""
        try:
            # 先关闭同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            time.sleep(3.0)  # 等待CARLA完全切换到异步模式（增加到3秒）
            
            # 重新开启同步模式
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.simulation_fps
            self.world.apply_settings(settings)
            time.sleep(1.0)  # 增加等待时间到1秒
            
            # 注意：不在这里调用tick()，因为可能没有actor监听
            # tick()会在新车辆和传感器创建后自动执行
            
            print("✅ 同步模式已重置")
        except Exception as e:
            print(f"⚠️  重置同步模式失败: {e}")
    
    def _cleanup_inner_collector(self):
        """清理内部收集器"""
        if self._inner_collector:
            # 先清理agent引用
            try:
                self._inner_collector.agent = None
            except:
                pass
            
            # 停止并销毁碰撞传感器
            try:
                if self._inner_collector.collision_sensor:
                    self._inner_collector.collision_sensor.stop()
                    self._inner_collector.collision_sensor.destroy()
                    self._inner_collector.collision_sensor = None
            except:
                pass
            
            # 停止并销毁摄像头
            try:
                if self._inner_collector.camera:
                    self._inner_collector.camera.stop()
                    self._inner_collector.camera.destroy()
                    self._inner_collector.camera = None
            except:
                pass
            
            # 销毁车辆
            try:
                if self._inner_collector.vehicle:
                    self._inner_collector.vehicle.destroy()
                    self._inner_collector.vehicle = None
            except:
                pass
            
            self._inner_collector = None
            
            # 等待CARLA处理销毁请求（不要在这里调用tick，因为没有actor监听会导致问题）
            time.sleep(1.0)
    
    def _get_recovery_transform(self):
        """
        从当前路线的waypoints中找恢复点
        
        返回:
            carla.Transform: 恢复点的transform，如果找不到合适的返回 None
        """
        if self._inner_collector is None or self._inner_collector.vehicle is None:
            return None
        
        if self._current_destination is None:
            return None
        
        # 优先从当前路线waypoints中查找
        if self._current_route_waypoints and len(self._current_route_waypoints) > 0:
            return self._get_recovery_from_route_waypoints()
        
        # 如果没有路线waypoints，尝试从agent的local_planner获取
        if (self._inner_collector.agent is not None and 
            hasattr(self._inner_collector.agent, 'get_local_planner')):
            try:
                local_planner = self._inner_collector.agent.get_local_planner()
                plan = list(local_planner.get_plan())
                if plan and len(plan) > 0:
                    self._current_route_waypoints = plan
                    return self._get_recovery_from_route_waypoints()
            except Exception as e:
                print(f"  ⚠️ 从agent获取路线失败: {e}")
        
        # 都失败了，返回None
        print(f"  ⚠️ 无法获取路线waypoints")
        return None
    
    def _get_recovery_from_route_waypoints(self):
        """
        从路线waypoints中查找恢复点
        
        逻辑：
        1. 找到当前位置最近的waypoint
        2. 沿路线向前跳过一定距离（跳过碰撞区域）
        3. 返回该waypoint的transform
        """
        if not self._current_route_waypoints:
            return None
        
        vehicle_location = self._inner_collector.vehicle.get_location()
        
        # 计算到终点的距离
        dist_to_dest = vehicle_location.distance(self._current_destination)
        
        # 如果已经很接近终点，不需要恢复
        if dist_to_dest < self.min_distance_to_destination:
            print(f"  ⚠️ 距终点仅 {dist_to_dest:.1f}m，不需要恢复")
            return None
        
        # 找到当前位置最近的waypoint索引
        min_dist = float('inf')
        current_idx = 0
        for i, (wp, _) in enumerate(self._current_route_waypoints):
            dist = vehicle_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                current_idx = i
        
        # 沿路线向前累积距离，跳过碰撞区域
        recovery_idx = current_idx
        accumulated_dist = 0.0
        
        while recovery_idx < len(self._current_route_waypoints) - 1:
            wp1 = self._current_route_waypoints[recovery_idx][0]
            wp2 = self._current_route_waypoints[recovery_idx + 1][0]
            segment_dist = wp1.transform.location.distance(wp2.transform.location)
            accumulated_dist += segment_dist
            recovery_idx += 1
            
            if accumulated_dist >= self.recovery_skip_distance:
                break
        
        # 检查是否还有足够的路线剩余
        if recovery_idx >= len(self._current_route_waypoints) - 1:
            print(f"  ⚠️ 路线剩余不足，无法恢复")
            return None
        
        # 获取恢复点的transform
        recovery_wp = self._current_route_waypoints[recovery_idx][0]
        recovery_transform = recovery_wp.transform
        
        # 检查恢复点到终点的距离
        recovery_to_dest = recovery_transform.location.distance(self._current_destination)
        if recovery_to_dest < self.min_distance_to_destination:
            print(f"  ⚠️ 恢复点距终点仅 {recovery_to_dest:.1f}m，不需要恢复")
            return None
        
        print(f"  📍 恢复点: waypoint[{recovery_idx}], 跳过 {accumulated_dist:.1f}m, 距终点 {recovery_to_dest:.1f}m")
        
        # 更新waypoints列表，移除已经走过的部分
        self._current_route_waypoints = self._current_route_waypoints[recovery_idx:]
        
        return recovery_transform
    
    def _spawn_vehicle_at_transform(self, spawn_transform, destination_idx):
        """在指定transform位置生成车辆（用于碰撞恢复）
        
        参数:
            spawn_transform: 车辆生成位置的carla.Transform
            destination_idx: 终点的spawn_point索引
        
        返回:
            bool: 是否成功生成车辆
        """
        print(f"🚗 在恢复点生成车辆...")
        
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        destination = self.spawn_points[destination_idx].location
        
        # 稍微抬高生成位置，避免与地面碰撞
        adjusted_transform = carla.Transform(
            carla.Location(
                x=spawn_transform.location.x,
                y=spawn_transform.location.y,
                z=spawn_transform.location.z + 0.5  # 抬高0.5米
            ),
            spawn_transform.rotation
        )
        
        self._inner_collector.vehicle = self.world.try_spawn_actor(vehicle_bp, adjusted_transform)
        
        if self._inner_collector.vehicle is None:
            print("❌ 在恢复点生成车辆失败！")
            return False
        
        print(f"✅ 车辆生成成功！")
        
        # 等待车辆稳定
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)
        
        # 配置BasicAgent
        if AGENTS_AVAILABLE:
            self._setup_recovery_agent(adjusted_transform, destination)
        else:
            self._setup_recovery_traffic_manager()
        
        # 重置噪声器状态
        self._inner_collector.reset_noisers()
        
        return True
    
    def _setup_recovery_agent(self, spawn_transform, destination):
        """为恢复的车辆配置BasicAgent"""
        from agents.navigation.basic_agent import BasicAgent
        
        ignore_vehicles = self._inner_collector.ignore_vehicles_percentage > 50
        
        opt_dict = {
            'target_speed': self._inner_collector.target_speed,
            'ignore_traffic_lights': self._inner_collector.ignore_traffic_lights,
            'ignore_stop_signs': self._inner_collector.ignore_signs,
            'ignore_vehicles': ignore_vehicles,
            'sampling_resolution': 1.0,
            'base_tlight_threshold': 5.0,
            'lateral_control_dict': {
                'K_P': 1.5, 'K_I': 0.0, 'K_D': 0.05,
                'dt': 1.0 / self._inner_collector.simulation_fps
            },
            'longitudinal_control_dict': {
                'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0,
                'dt': 1.0 / self._inner_collector.simulation_fps
            },
            'max_steering': 0.8,
            'max_throttle': 0.75,
            'max_brake': 0.5,
            'base_min_distance': 2.0,
            'distance_ratio': 0.3
        }
        
        self._inner_collector.agent = BasicAgent(
            self._inner_collector.vehicle,
            target_speed=self._inner_collector.target_speed,
            opt_dict=opt_dict,
            map_inst=self.world.get_map()
        )
        
        self._inner_collector.agent.set_destination(destination, start_location=spawn_transform.location)
        
        # 更新路线waypoints为新agent规划的路线（确保一致性）
        try:
            local_planner = self._inner_collector.agent.get_local_planner()
            new_plan = list(local_planner.get_plan())
            if new_plan and len(new_plan) > 0:
                self._current_route_waypoints = new_plan
                print(f"  ✅ BasicAgent 已配置（恢复模式），路线 {len(new_plan)} 个waypoints")
            else:
                print(f"  ✅ BasicAgent 已配置（恢复模式）")
        except Exception as e:
            print(f"  ✅ BasicAgent 已配置（恢复模式），获取路线失败: {e}")
    
    def _setup_recovery_traffic_manager(self):
        """为恢复的车辆配置Traffic Manager（降级方案）"""
        traffic_manager = self.client.get_trafficmanager()
        self._inner_collector.vehicle.set_autopilot(True, traffic_manager.get_port())
        
        if self._inner_collector.ignore_traffic_lights:
            traffic_manager.ignore_lights_percentage(self._inner_collector.vehicle, 100)
        if self._inner_collector.ignore_signs:
            traffic_manager.ignore_signs_percentage(self._inner_collector.vehicle, 100)
        traffic_manager.ignore_vehicles_percentage(
            self._inner_collector.vehicle, 
            self._inner_collector.ignore_vehicles_percentage
        )
        traffic_manager.auto_lane_change(self._inner_collector.vehicle, False)
        print(f"  ✅ Traffic Manager 已配置（恢复模式）")
    
    def _auto_collect(self, save_path):
        """自动收集数据
        
        功能说明：
        1. 每 auto_save_interval 帧（默认200帧）保存一个 segment
        2. 如果在 segment 内发生碰撞，丢弃整个 segment
        3. 碰撞后返回恢复点transform，由上层处理恢复逻辑
        
        返回:
            dict: {
                'success': bool,                    # 是否正常完成
                'saved_frames': int,                # 已保存帧数
                'need_recovery': bool,              # 是否需要恢复
                'recovery_transform': carla.Transform,  # 恢复点transform（如果需要恢复）
            }
        """
        os.makedirs(save_path, exist_ok=True)
        
        result = {
            'success': False,
            'saved_frames': 0,
            'need_recovery': False,
            'recovery_transform': None,
        }
        
        self._inner_collector.enable_visualization = True
        self._inner_collector.wait_for_first_frame()
        
        saved_frames = 0
        pending_frames = 0
        segment_data = {'rgb': [], 'targets': []}
        segment_start_cmd = None
        
        try:
            while (saved_frames + pending_frames) < self.frames_per_route:
                self._inner_collector.step_simulation()
                
                if self._inner_collector._is_route_completed():
                    print(f"\n🎯 已到达目的地！")
                    break
                
                # === 碰撞和异常检测 ===
                is_collision = self._inner_collector.collision_detected
                is_anomaly = self._inner_collector.check_vehicle_anomaly()
                
                if is_collision or is_anomaly:
                    if is_collision:
                        print(f"\n💥 检测到碰撞！")
                    # 异常类型已在 check_vehicle_anomaly 中打印
                    
                    # 丢弃当前 segment
                    if pending_frames > 0:
                        print(f"  🗑️ 丢弃当前 segment（{pending_frames} 帧）")
                    
                    # 尝试找恢复点（基于路线waypoints）
                    if self.collision_recovery_enabled:
                        recovery_transform = self._get_recovery_transform()
                        if recovery_transform is not None:
                            print(f"  🔄 找到恢复点（基于路线waypoints）")
                            result['need_recovery'] = True
                            result['recovery_transform'] = recovery_transform
                        else:
                            print(f"  ⚠️ 未找到合适的恢复点")
                    
                    result['saved_frames'] = saved_frames
                    return result
                
                # === 正常数据收集 ===
                if len(self._inner_collector.image_buffer) == 0:
                    continue
                
                current_image = self._inner_collector.image_buffer[-1].copy()
                speed_kmh = self._inner_collector._get_vehicle_speed()
                current_cmd = self._inner_collector._get_navigation_command()
                
                # 跳过无效帧
                if current_image.mean() < 5 or speed_kmh > 150:
                    continue
                
                # 再次检查碰撞和异常
                if self._inner_collector.collision_detected or self._inner_collector.anomaly_detected:
                    continue
                
                targets = self._inner_collector._build_targets(speed_kmh, current_cmd)
                
                if pending_frames == 0:
                    segment_start_cmd = current_cmd
                
                segment_data['rgb'].append(current_image)
                segment_data['targets'].append(targets)
                pending_frames += 1
                
                # 可视化
                if self._inner_collector.enable_visualization:
                    self._inner_collector.segment_count = pending_frames
                    total_progress = saved_frames + pending_frames
                    self._inner_collector._visualize_frame(
                        current_image, speed_kmh, current_cmd,
                        total_progress, self.frames_per_route, is_collecting=True
                    )
                
                # 定期保存
                if pending_frames >= self.auto_save_interval:
                    if not self._inner_collector.collision_detected and not self._inner_collector.anomaly_detected:
                        self._save_segment_auto(segment_data, save_path, segment_start_cmd)
                        saved_frames += pending_frames
                    segment_data = {'rgb': [], 'targets': []}
                    pending_frames = 0
                    segment_start_cmd = None
                    self._inner_collector.reset_collision_state()
                    self._inner_collector.reset_anomaly_state()
                    # 重置噪声器，为下一个segment重新规划噪声
                    self._inner_collector.reset_noisers()
                
                # 进度显示
                if (saved_frames + pending_frames) % 100 == 0:
                    print(f"  [收集中] 帧数: {saved_frames + pending_frames}/{self.frames_per_route}")
            
            # 保存剩余数据
            if pending_frames > 0 and not self._inner_collector.collision_detected and not self._inner_collector.anomaly_detected:
                self._save_segment_auto(segment_data, save_path,
                                        segment_start_cmd if segment_start_cmd else 2.0)
                saved_frames += pending_frames
            
            print(f"\n📊 本次收集: {saved_frames} 帧")
            self.total_frames_collected += saved_frames
            result['success'] = True
            result['saved_frames'] = saved_frames
            return result
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "time-out" in error_msg:
                print(f"❌ 收集出错: CARLA服务器超时")
            else:
                print(f"❌ 收集出错: {e}")
            result['saved_frames'] = saved_frames
            return result
        except Exception as e:
            print(f"❌ 收集出错: {e}")
            import traceback
            traceback.print_exc()
            result['saved_frames'] = saved_frames
            return result
        finally:
            cv2.destroyAllWindows()
    
    def _save_segment_auto(self, segment_data, save_path, command):
        """自动保存数据段"""
        if len(segment_data['rgb']) == 0:
            return
        
        self._inner_collector._save_data_to_h5(
            segment_data['rgb'], segment_data['targets'],
            save_path, command
        )
    
    def validate_route(self, start_idx, end_idx):
        """验证路线可行性"""
        if not AGENTS_AVAILABLE or self.route_planner is None:
            return True, None, 0.0
        
        try:
            route = self.route_planner.trace_route(
                self.spawn_points[start_idx].location,
                self.spawn_points[end_idx].location
            )
            
            if not route:
                return False, None, 0.0
            
            route_distance = sum(
                route[i][0].transform.location.distance(route[i-1][0].transform.location)
                for i in range(1, len(route))
            )
            return True, route, route_distance
        except:
            return False, None, 0.0
    
    def run(self, save_path='./auto_collected_data', strategy='smart', route_cache_path=None):
        """运行全自动收集
        
        参数:
            save_path: 数据保存路径
            strategy: 路线生成策略 ('smart' 或 'exhaustive')
            route_cache_path: 路线缓存文件路径，None则自动生成
        """
        self.route_generation_strategy = strategy
        
        # 自动生成缓存路径（基于地图和配置）
        if route_cache_path is None:
            route_cache_path = os.path.join(
                save_path, 
                f"route_cache_{self.town}_{strategy}_{int(self.min_distance)}_{int(self.max_distance)}.json"
            )
        
        try:
            self.connect()
            route_pairs = self.generate_route_pairs(cache_path=route_cache_path)
            
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
                
                print(f"\n📍 路线 {idx+1}/{len(route_pairs)}: {start_idx} → {end_idx} ({distance:.1f}m)")
                
                # 路线验证（可通过配置禁用）
                if self.enable_route_validation:
                    valid, _, route_dist = self.validate_route(start_idx, end_idx)
                    if not valid:
                        self.failed_routes.append((start_idx, end_idx, "不可达"))
                        continue
                
                # 收集数据（支持重试）
                success = False
                retries = 0
                max_retries = self.max_retries if self.retry_failed_routes else 1  # 至少重试1次
                while not success and retries <= max_retries:
                    if retries > 0:
                        print(f"  🔄 重试 {retries}/{max_retries}...")
                        # 重试前重置同步模式
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
                
                # 路线之间暂停（用于清理资源）
                if self.pause_between_routes > 0 and idx < len(route_pairs) - 1:
                    time.sleep(self.pause_between_routes)
                
                # 进度
                elapsed = time.time() - start_time
                remaining = elapsed / (idx + 1) * (len(route_pairs) - idx - 1)
                print(f"📊 进度: {idx+1}/{len(route_pairs)}, 成功: {self.total_routes_completed}, "
                      f"剩余: {remaining/60:.1f}分钟")
            
            self._print_final_statistics(time.time() - start_time, save_path)
            
        except KeyboardInterrupt:
            print("\n⚠️  收到中断信号...")
        finally:
            self._cleanup_npcs()
            if self.world:
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    self.world.apply_settings(settings)
                except:
                    pass
    
    def _print_final_statistics(self, total_time, save_path):
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
            'failed': [{'start': s, 'end': e, 'reason': r} for s, e, r in self.failed_routes],
            'timestamp': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(save_path, 'collection_statistics.json')
        os.makedirs(save_path, exist_ok=True)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"✅ 统计已保存: {stats_file}")


def load_config(config_path='auto_collection_config.json'):
    """加载配置文件"""
    default_config = {
        'carla_settings': {'host': 'localhost', 'port': 2000, 'town': 'Town01'},
        'traffic_rules': {'ignore_traffic_lights': True, 'ignore_signs': True, 'ignore_vehicles_percentage': 80},
        'world_settings': {'spawn_npc_vehicles': False, 'num_npc_vehicles': 0,
                          'spawn_npc_walkers': False, 'num_npc_walkers': 0},
        'weather_settings': {'preset': 'ClearNoon', 'custom': {}},
        'route_generation': {'strategy': 'smart', 'min_distance': 50.0, 'max_distance': 500.0,
                            'target_routes_ratio': 1.0, 'overlap_threshold': 0.5, 'turn_priority_ratio': 0.7,
                            'max_candidates_to_analyze': 0},
        'collection_settings': {'frames_per_route': 1000, 'save_path': './auto_collected_data',
                               'simulation_fps': 20, 'target_speed_kmh': 10.0, 'auto_save_interval': 200},
        'noise_settings': {'enabled': False, 'lateral_noise': True, 'longitudinal_noise': False,
                          'noise_ratio': 0.4, 'max_steer_offset': 0.35, 'max_throttle_offset': 0.2,
                          'noise_modes': {
                              'impulse': {'duration_seconds': [0.5, 1.0], 'strength_percent': 100, 'probability_percent': 25},
                              'smooth': {'duration_seconds': [1.5, 2.5], 'strength_percent': 80, 'probability_percent': 35},
                              'drift': {'duration_seconds': [2.5, 4.0], 'strength_percent': 40, 'probability_percent': 20},
                              'jitter': {'duration_seconds': [0.8, 1.5], 'strength_percent': 50, 'probability_percent': 20}
                          }},
        'collision_recovery': {'enabled': True, 'max_collisions_per_route': 99,
                              'min_distance_to_destination': 30.0},
        'anomaly_detection': {'enabled': True,
                             'spin_detection': {'enabled': True, 'threshold_degrees': 270.0, 'time_window': 3.0},
                             'rollover_detection': {'enabled': True, 'pitch_threshold': 45.0, 'roll_threshold': 45.0},
                             'stuck_detection': {'enabled': True, 'speed_threshold': 0.5, 'time_threshold': 5.0}},
        'advanced_settings': {'enable_route_validation': True, 'retry_failed_routes': False,
                             'max_retries': 3, 'pause_between_routes': 2},
        'multi_weather_settings': {'enabled': False, 'weather_preset': 'basic', 'custom_weather_list': []}
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_path)
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            for section in default_config:
                if section in loaded:
                    default_config[section].update(loaded[section])
            print(f"✅ 已加载配置: {config_file}")
        except Exception as e:
            print(f"⚠️  加载配置失败: {e}")
    
    return default_config


def get_weather_list(preset):
    """根据预设名称获取天气列表"""
    weather_presets = {
        'basic': ['ClearNoon', 'CloudyNoon', 'ClearSunset', 'ClearNight'],
        'all_noon': ['ClearNoon', 'CloudyNoon', 'WetNoon', 'SoftRainNoon', 'HardRainNoon'],
        'all_sunset': ['ClearSunset', 'CloudySunset', 'WetSunset', 'SoftRainSunset', 'HardRainSunset'],
        'all_night': ['ClearNight', 'CloudyNight', 'WetNight', 'SoftRainNight', 'HardRainNight'],
        'clear_all': ['ClearNoon', 'ClearSunset', 'ClearNight'],
        'rain_all': ['SoftRainNoon', 'HardRainNoon', 'SoftRainSunset', 'SoftRainNight'],
        'full': ['ClearNoon', 'CloudyNoon', 'WetNoon', 'SoftRainNoon', 'HardRainNoon',
                 'ClearSunset', 'CloudySunset', 'WetSunset',
                 'ClearNight', 'CloudyNight', 'WetNight']
    }
    return weather_presets.get(preset, ['ClearNoon'])


def run_single_weather_collection(config, weather_name, base_save_path):
    """运行单个天气的数据收集"""
    # 更新天气配置
    config['weather_settings'] = {'preset': weather_name}
    
    # 创建天气专属保存路径
    weather_save_path = os.path.join(base_save_path, weather_name)
    
    print(f"\n{'='*70}")
    print(f"🌤️  开始收集天气: {weather_name}")
    print(f"📁 保存路径: {weather_save_path}")
    print(f"{'='*70}")
    
    collector = AutoFullTownCollector(
        host=config['carla_settings']['host'],
        port=config['carla_settings']['port'],
        town=config['carla_settings']['town'],
        ignore_traffic_lights=config['traffic_rules']['ignore_traffic_lights'],
        ignore_signs=config['traffic_rules']['ignore_signs'],
        ignore_vehicles_percentage=config['traffic_rules']['ignore_vehicles_percentage'],
        target_speed=config['collection_settings']['target_speed_kmh'],
        simulation_fps=config['collection_settings']['simulation_fps'],
        spawn_npc_vehicles=config['world_settings']['spawn_npc_vehicles'],
        num_npc_vehicles=config['world_settings']['num_npc_vehicles'],
        spawn_npc_walkers=config['world_settings']['spawn_npc_walkers'],
        num_npc_walkers=config['world_settings']['num_npc_walkers'],
        weather_config=config.get('weather_settings', {})
    )
    
    collector.min_distance = config['route_generation']['min_distance']
    collector.max_distance = config['route_generation']['max_distance']
    collector.frames_per_route = config['collection_settings']['frames_per_route']
    collector.target_routes_ratio = config['route_generation'].get('target_routes_ratio', 1.0)
    collector.overlap_threshold = config['route_generation']['overlap_threshold']
    collector.turn_priority_ratio = config['route_generation'].get('turn_priority_ratio', 0.7)
    collector.auto_save_interval = config['collection_settings'].get('auto_save_interval', 200)
    
    # 路线分析参数
    collector.max_candidates_to_analyze = config['route_generation'].get('max_candidates_to_analyze', 0)
    
    # 高级设置
    advanced_config = config.get('advanced_settings', {})
    collector.enable_route_validation = advanced_config.get('enable_route_validation', True)
    collector.retry_failed_routes = advanced_config.get('retry_failed_routes', False)
    collector.max_retries = advanced_config.get('max_retries', 3)
    collector.pause_between_routes = advanced_config.get('pause_between_routes', 2)
    
    # 噪声配置
    noise_config = config.get('noise_settings', {})
    collector.noise_enabled = noise_config.get('enabled', False)
    collector.lateral_noise_enabled = noise_config.get('lateral_noise', True)
    collector.longitudinal_noise_enabled = noise_config.get('longitudinal_noise', False)
    collector.noise_ratio = noise_config.get('noise_ratio', 0.4)
    collector.max_steer_offset = noise_config.get('max_steer_offset', 0.35)
    collector.max_throttle_offset = noise_config.get('max_throttle_offset', 0.2)
    collector.noise_mode_config = noise_config.get('noise_modes', None)
    collector._init_noisers()
    
    # 碰撞恢复配置
    collision_config = config.get('collision_recovery', {})
    collector.collision_recovery_enabled = collision_config.get('enabled', True)
    collector.max_collisions_per_route = collision_config.get('max_collisions_per_route', 99)
    collector.min_distance_to_destination = collision_config.get('min_distance_to_destination', 30.0)
    collector.recovery_skip_distance = collision_config.get('recovery_skip_distance', 25.0)
    
    # 异常检测配置
    anomaly_config = config.get('anomaly_detection', {})
    collector.anomaly_detection_enabled = anomaly_config.get('enabled', True)
    spin_config = anomaly_config.get('spin_detection', {})
    collector.spin_detection_enabled = spin_config.get('enabled', True)
    collector.spin_threshold_degrees = spin_config.get('threshold_degrees', 270.0)
    collector.spin_time_window = spin_config.get('time_window', 3.0)
    rollover_config = anomaly_config.get('rollover_detection', {})
    collector.rollover_detection_enabled = rollover_config.get('enabled', True)
    collector.rollover_pitch_threshold = rollover_config.get('pitch_threshold', 45.0)
    collector.rollover_roll_threshold = rollover_config.get('roll_threshold', 45.0)
    stuck_config = anomaly_config.get('stuck_detection', {})
    collector.stuck_detection_enabled = stuck_config.get('enabled', True)
    collector.stuck_speed_threshold = stuck_config.get('speed_threshold', 0.5)
    collector.stuck_time_threshold = stuck_config.get('time_threshold', 5.0)
    
    # 路线缓存路径（放在 base_save_path 下，所有天气共享同一份路线缓存）
    route_cache_path = os.path.join(
        base_save_path,
        f"route_cache_{config['carla_settings']['town']}_{config['route_generation']['strategy']}.json"
    )
    
    collector.run(
        save_path=weather_save_path,
        strategy=config['route_generation']['strategy'],
        route_cache_path=route_cache_path
    )
    
    return collector.total_frames_collected


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='全自动数据收集器')
    parser.add_argument('--config', default='auto_collection_config.json')
    parser.add_argument('--host', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, help='CARLA服务器端口')
    parser.add_argument('--save-path', help='保存路径')
    parser.add_argument('--strategy', choices=['smart', 'exhaustive'])
    parser.add_argument('--target-routes-ratio', type=float, help='路线选择比例(0-1)')
    parser.add_argument('--frames-per-route', type=int)
    # 多天气支持参数
    parser.add_argument('--multi-weather', type=str, 
                        help='多天气轮换预设: basic/all_noon/all_sunset/all_night/clear_all/rain_all/full')
    parser.add_argument('--weather-list', nargs='+', 
                        help='自定义天气列表，如: ClearNoon CloudyNoon WetNoon')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 命令行覆盖
    if args.host:
        config['carla_settings']['host'] = args.host
    if args.port:
        config['carla_settings']['port'] = args.port
    if args.save_path:
        config['collection_settings']['save_path'] = args.save_path
    if args.strategy:
        config['route_generation']['strategy'] = args.strategy
    if args.target_routes_ratio is not None:
        config['route_generation']['target_routes_ratio'] = args.target_routes_ratio
    if args.frames_per_route:
        config['collection_settings']['frames_per_route'] = args.frames_per_route
    
    # 确定天气列表
    weather_list = None
    
    # 优先级: 命令行 --weather-list > 命令行 --multi-weather > 配置文件
    if args.weather_list:
        weather_list = args.weather_list
        print(f"\n🌤️  使用命令行指定的天气列表: {weather_list}")
    elif args.multi_weather:
        weather_list = get_weather_list(args.multi_weather)
        print(f"\n🌤️  使用天气预设 '{args.multi_weather}': {weather_list}")
    else:
        # 检查配置文件中的多天气设置
        multi_weather_config = config.get('multi_weather_settings', {})
        if multi_weather_config.get('enabled', False):
            custom_list = multi_weather_config.get('custom_weather_list', [])
            if custom_list:
                weather_list = custom_list
                print(f"\n🌤️  使用配置文件自定义天气列表: {weather_list}")
            else:
                preset = multi_weather_config.get('weather_preset', 'basic')
                weather_list = get_weather_list(preset)
                print(f"\n🌤️  使用配置文件天气预设 '{preset}': {weather_list}")
    
    # 多天气轮换模式
    if weather_list and len(weather_list) > 1:
        base_save_path = config['collection_settings']['save_path']
        total_frames_all_weathers = 0
        
        print(f"\n{'='*70}")
        print(f"🌈 多天气轮换收集模式")
        print(f"{'='*70}")
        print(f"天气数量: {len(weather_list)}")
        print(f"天气列表: {', '.join(weather_list)}")
        print(f"策略: {config['route_generation']['strategy']}")
        print(f"基础保存路径: {base_save_path}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for idx, weather_name in enumerate(weather_list):
            print(f"\n{'#'*70}")
            print(f"# 天气 {idx+1}/{len(weather_list)}: {weather_name}")
            print(f"{'#'*70}")
            
            try:
                frames = run_single_weather_collection(config, weather_name, base_save_path)
                total_frames_all_weathers += frames
                print(f"\n✅ 天气 {weather_name} 收集完成，帧数: {frames}")
            except Exception as e:
                print(f"\n❌ 天气 {weather_name} 收集失败: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"🎉 多天气轮换收集完成！")
        print(f"{'='*70}")
        print(f"总天气数: {len(weather_list)}")
        print(f"总帧数: {total_frames_all_weathers}")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"{'='*70}")
    
    # 单天气模式
    else:
        collector = AutoFullTownCollector(
            host=config['carla_settings']['host'],
            port=config['carla_settings']['port'],
            town=config['carla_settings']['town'],
            ignore_traffic_lights=config['traffic_rules']['ignore_traffic_lights'],
            ignore_signs=config['traffic_rules']['ignore_signs'],
            ignore_vehicles_percentage=config['traffic_rules']['ignore_vehicles_percentage'],
            target_speed=config['collection_settings']['target_speed_kmh'],
            simulation_fps=config['collection_settings']['simulation_fps'],
            spawn_npc_vehicles=config['world_settings']['spawn_npc_vehicles'],
            num_npc_vehicles=config['world_settings']['num_npc_vehicles'],
            spawn_npc_walkers=config['world_settings']['spawn_npc_walkers'],
            num_npc_walkers=config['world_settings']['num_npc_walkers'],
            weather_config=config.get('weather_settings', {})
        )
        
        collector.min_distance = config['route_generation']['min_distance']
        collector.max_distance = config['route_generation']['max_distance']
        collector.frames_per_route = config['collection_settings']['frames_per_route']
        collector.target_routes_ratio = config['route_generation'].get('target_routes_ratio', 1.0)
        collector.overlap_threshold = config['route_generation']['overlap_threshold']
        collector.turn_priority_ratio = config['route_generation'].get('turn_priority_ratio', 0.7)
        collector.auto_save_interval = config['collection_settings'].get('auto_save_interval', 200)
        
        # 路线分析参数
        collector.max_candidates_to_analyze = config['route_generation'].get('max_candidates_to_analyze', 0)
        
        # 高级设置
        advanced_config = config.get('advanced_settings', {})
        collector.enable_route_validation = advanced_config.get('enable_route_validation', True)
        collector.retry_failed_routes = advanced_config.get('retry_failed_routes', False)
        collector.max_retries = advanced_config.get('max_retries', 3)
        collector.pause_between_routes = advanced_config.get('pause_between_routes', 2)
        
        # 噪声配置
        noise_config = config.get('noise_settings', {})
        collector.noise_enabled = noise_config.get('enabled', False)
        collector.lateral_noise_enabled = noise_config.get('lateral_noise', True)
        collector.longitudinal_noise_enabled = noise_config.get('longitudinal_noise', False)
        collector.noise_ratio = noise_config.get('noise_ratio', 0.4)
        collector.max_steer_offset = noise_config.get('max_steer_offset', 0.35)
        collector.max_throttle_offset = noise_config.get('max_throttle_offset', 0.2)
        collector.noise_mode_config = noise_config.get('noise_modes', None)
        collector._init_noisers()
        
        if collector.noise_enabled:
            print(f"\n🎲 噪声注入已启用:")
            print(f"  • 噪声占比: {collector.noise_ratio*100:.0f}%")
            print(f"  • 横向噪声: {'✅' if collector.lateral_noise_enabled else '❌'} (max_offset={collector.max_steer_offset})")
            print(f"  • 纵向噪声: {'✅' if collector.longitudinal_noise_enabled else '❌'} (max_offset={collector.max_throttle_offset})")
        
        # 碰撞恢复配置
        collision_config = config.get('collision_recovery', {})
        collector.collision_recovery_enabled = collision_config.get('enabled', True)
        collector.max_collisions_per_route = collision_config.get('max_collisions_per_route', 99)
        collector.min_distance_to_destination = collision_config.get('min_distance_to_destination', 30.0)
        collector.recovery_skip_distance = collision_config.get('recovery_skip_distance', 25.0)
        
        # 异常检测配置
        anomaly_config = config.get('anomaly_detection', {})
        collector.anomaly_detection_enabled = anomaly_config.get('enabled', True)
        spin_config = anomaly_config.get('spin_detection', {})
        collector.spin_detection_enabled = spin_config.get('enabled', True)
        collector.spin_threshold_degrees = spin_config.get('threshold_degrees', 270.0)
        collector.spin_time_window = spin_config.get('time_window', 3.0)
        rollover_config = anomaly_config.get('rollover_detection', {})
        collector.rollover_detection_enabled = rollover_config.get('enabled', True)
        collector.rollover_pitch_threshold = rollover_config.get('pitch_threshold', 45.0)
        collector.rollover_roll_threshold = rollover_config.get('roll_threshold', 45.0)
        stuck_config = anomaly_config.get('stuck_detection', {})
        collector.stuck_detection_enabled = stuck_config.get('enabled', True)
        collector.stuck_speed_threshold = stuck_config.get('speed_threshold', 0.5)
        collector.stuck_time_threshold = stuck_config.get('time_threshold', 5.0)
        
        collector.run(
            save_path=config['collection_settings']['save_path'],
            strategy=config['route_generation']['strategy']
        )


if __name__ == '__main__':
    main()
