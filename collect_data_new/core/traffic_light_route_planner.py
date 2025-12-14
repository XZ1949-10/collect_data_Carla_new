#!/usr/bin/env python
# coding=utf-8
"""
红绿灯路口路线规划模块

专门生成经过红绿灯路口的路线，用于收集红绿灯场景数据。

使用示例:
    from collect_data_new.core import TrafficLightRoutePlanner
    
    # 创建规划器
    planner = TrafficLightRoutePlanner(world, spawn_points, town='Town01')
    
    # 配置参数
    planner.configure(
        min_distance=100.0,
        max_distance=300.0,
        min_traffic_lights=1,  # 路线至少经过1个红绿灯
        max_traffic_lights=5,  # 路线最多经过5个红绿灯
        traffic_light_radius=30.0,  # 红绿灯检测半径
    )
    
    # 生成路线
    routes = planner.generate_routes(cache_path='./tl_routes_cache.json')
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any, Set

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


class TrafficLightRoutePlanner:
    """
    红绿灯路口路线规划器
    
    专门生成经过红绿灯路口的路线，支持：
    - 筛选经过指定数量红绿灯的路线
    - 配置红绿灯检测半径
    - 路线去重和平衡选择
    - 缓存机制
    """
    
    # 命令映射
    COMMAND_MAP = {
        'LANEFOLLOW': 2, 'LEFT': 3, 'RIGHT': 4, 'STRAIGHT': 5,
        'CHANGELANELEFT': 2, 'CHANGELANERIGHT': 2
    }
    
    def __init__(self, world, spawn_points: List, town: str = None):
        """
        初始化红绿灯路线规划器
        
        参数:
            world: CARLA world 对象
            spawn_points: 生成点列表
            town: 地图名称（用于缓存验证）
        """
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.world = world
        self.spawn_points = spawn_points
        self._route_planner = None
        
        # 地图名称
        self.town = town or self._get_map_name()
        
        # 路线生成参数
        self.min_distance = 100.0
        self.max_distance = 400.0
        self.overlap_threshold = 0.5
        self.target_routes_ratio = 1.0
        self.max_candidates_to_analyze = 0
        
        # 红绿灯相关参数
        self.min_traffic_lights = 1      # 路线最少经过的红绿灯数量
        self.max_traffic_lights = 10     # 路线最多经过的红绿灯数量（0=不限制）
        self.traffic_light_radius = 30.0  # 红绿灯检测半径（米）
        self.prefer_more_lights = True    # 是否优先选择经过更多红绿灯的路线
        
        # 缓存红绿灯位置
        self._traffic_light_locations: List[Tuple[float, float, float]] = []
        
        # 去重后的分组信息（供选择步骤使用）
        self._deduplicated_groups: Dict[int, List[Dict]] = {}
        
        self._init_route_planner()
        self._cache_traffic_lights()
    
    def _get_map_name(self) -> str:
        """获取当前地图名称"""
        try:
            return self.world.get_map().name.split('/')[-1]
        except:
            return 'Unknown'
    
    def _init_route_planner(self):
        """初始化 GlobalRoutePlanner"""
        if not AGENTS_AVAILABLE:
            raise RuntimeError(
                "❌ agents 模块不可用！\n"
                "请确保 CARLA PythonAPI 的 agents 模块已正确安装。"
            )
        
        try:
            self._route_planner = GlobalRoutePlanner(
                self.world.get_map(), sampling_resolution=2.0
            )
            print("✅ 红绿灯路线规划器初始化成功")
        except Exception as e:
            raise RuntimeError(f"❌ 路径规划器初始化失败: {e}")
    
    def _cache_traffic_lights(self):
        """缓存所有红绿灯位置"""
        try:
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
            self._traffic_light_locations = []
            
            for tl in traffic_lights:
                loc = tl.get_location()
                self._traffic_light_locations.append((loc.x, loc.y, loc.z))
            
            print(f"🚦 已缓存 {len(self._traffic_light_locations)} 个红绿灯位置")
        except Exception as e:
            print(f"⚠️ 缓存红绿灯位置失败: {e}")
            self._traffic_light_locations = []
    
    @property
    def traffic_light_count(self) -> int:
        """红绿灯总数"""
        return len(self._traffic_light_locations)
    
    def configure(self, 
                  min_distance: float = None,
                  max_distance: float = None,
                  overlap_threshold: float = None,
                  target_routes_ratio: float = None,
                  max_candidates: int = None,
                  min_traffic_lights: int = None,
                  max_traffic_lights: int = None,
                  traffic_light_radius: float = None,
                  prefer_more_lights: bool = None,
                  town: str = None):
        """
        配置路线生成参数
        
        参数:
            min_distance: 最小路线距离（米）
            max_distance: 最大路线距离（米）
            overlap_threshold: 路线重叠阈值（0-1）
            target_routes_ratio: 目标路线数量比例（0-1）
            max_candidates: 最大候选路线分析数量
            min_traffic_lights: 路线最少经过的红绿灯数量
            max_traffic_lights: 路线最多经过的红绿灯数量（0=不限制）
            traffic_light_radius: 红绿灯检测半径（米）
            prefer_more_lights: 是否优先选择经过更多红绿灯的路线
            town: 地图名称
        """
        if min_distance is not None:
            self.min_distance = min_distance
        if max_distance is not None:
            self.max_distance = max_distance
        if overlap_threshold is not None:
            self.overlap_threshold = overlap_threshold
        if target_routes_ratio is not None:
            self.target_routes_ratio = target_routes_ratio
        if max_candidates is not None:
            self.max_candidates_to_analyze = max_candidates
        if min_traffic_lights is not None:
            self.min_traffic_lights = min_traffic_lights
        if max_traffic_lights is not None:
            self.max_traffic_lights = max_traffic_lights
        if traffic_light_radius is not None:
            self.traffic_light_radius = traffic_light_radius
        if prefer_more_lights is not None:
            self.prefer_more_lights = prefer_more_lights
        if town is not None:
            self.town = town
    
    def _count_traffic_lights_on_route(self, waypoints: List[Tuple[float, float]]) -> int:
        """
        计算路线经过的红绿灯数量
        
        参数:
            waypoints: 路线路点列表 [(x, y), ...]
            
        返回:
            经过的红绿灯数量
        """
        if not self._traffic_light_locations or not waypoints:
            return 0
        
        # 使用集合记录已经计数的红绿灯，避免重复计数
        counted_lights: Set[int] = set()
        radius_sq = self.traffic_light_radius ** 2
        
        for wp_x, wp_y in waypoints:
            for i, (tl_x, tl_y, tl_z) in enumerate(self._traffic_light_locations):
                if i in counted_lights:
                    continue
                
                # 计算2D距离（忽略高度）
                dist_sq = (wp_x - tl_x) ** 2 + (wp_y - tl_y) ** 2
                if dist_sq <= radius_sq:
                    counted_lights.add(i)
        
        return len(counted_lights)
    
    def _get_traffic_lights_on_route(self, waypoints: List[Tuple[float, float]]) -> List[int]:
        """
        获取路线经过的红绿灯索引列表
        
        参数:
            waypoints: 路线路点列表 [(x, y), ...]
            
        返回:
            红绿灯索引列表
        """
        if not self._traffic_light_locations or not waypoints:
            return []
        
        counted_lights: Set[int] = set()
        radius_sq = self.traffic_light_radius ** 2
        
        for wp_x, wp_y in waypoints:
            for i, (tl_x, tl_y, tl_z) in enumerate(self._traffic_light_locations):
                if i in counted_lights:
                    continue
                
                dist_sq = (wp_x - tl_x) ** 2 + (wp_y - tl_y) ** 2
                if dist_sq <= radius_sq:
                    counted_lights.add(i)
        
        return list(counted_lights)

    def generate_routes(self, cache_path: Optional[str] = None) -> List[Tuple[int, int, float, int]]:
        """
        生成经过红绿灯的路线
        
        参数:
            cache_path: 缓存文件路径
            
        返回:
            路线列表 [(start_idx, end_idx, distance, traffic_light_count), ...]
        """
        print("\n" + "="*70)
        print("🚦 生成红绿灯路口路线")
        print("="*70)
        
        if not self._traffic_light_locations:
            print("⚠️ 未找到红绿灯，无法生成红绿灯路线")
            return []
        
        print(f"📍 地图红绿灯总数: {len(self._traffic_light_locations)}")
        print(f"📏 路线距离范围: {self.min_distance:.0f}m ~ {self.max_distance:.0f}m")
        print(f"🚦 红绿灯数量要求: {self.min_traffic_lights} ~ "
              f"{self.max_traffic_lights if self.max_traffic_lights > 0 else '不限'}")
        print(f"📐 红绿灯检测半径: {self.traffic_light_radius:.0f}m")
        
        # 尝试从缓存加载
        if cache_path and os.path.exists(cache_path):
            routes = self._load_from_cache(cache_path)
            if routes:
                print(f"✅ 从缓存加载了 {len(routes)} 条红绿灯路线")
                self._print_statistics(routes)
                return routes
        
        # 生成新路线
        routes = self._generate_traffic_light_routes()
        
        if routes:
            self._print_statistics(routes)
            if cache_path:
                self._save_to_cache(routes, cache_path)
        
        return routes
    
    def _generate_traffic_light_routes(self) -> List[Tuple[int, int, float, int]]:
        """生成经过红绿灯的路线"""
        print(f"\n🔍 分析候选路线...")
        
        if not AGENTS_AVAILABLE or self._route_planner is None:
            raise RuntimeError("❌ 路径规划器不可用")
        
        # 1. 分析所有候选路线
        candidates = self._analyze_candidates()
        if not candidates:
            print("⚠️ 未找到符合条件的路线")
            return []
        
        # 2. 去重
        deduplicated = self._deduplicate(candidates)
        if not deduplicated:
            return []
        
        # 3. 按比例选择
        selected = self._select_routes(deduplicated)
        
        # 转换为返回格式
        result = [
            (r['start_idx'], r['end_idx'], r['distance'], r['traffic_light_count'])
            for r in selected
        ]
        random.shuffle(result)
        return result
    
    def _analyze_candidates(self) -> List[Dict]:
        """分析候选路线，筛选经过红绿灯的路线"""
        candidates = []
        num_spawns = len(self.spawn_points)
        total_pairs = num_spawns * (num_spawns - 1)
        
        # 先打印总组合数
        print(f"  📋 总组合数: {total_pairs} 条 (生成点: {num_spawns} 个)")
        
        # 采样
        use_sampling = (self.max_candidates_to_analyze > 0 and 
                        total_pairs > self.max_candidates_to_analyze)
        
        if use_sampling:
            print(f"  ⚡ 随机采样 {self.max_candidates_to_analyze} 条进行分析...")
            all_pairs = [(i, j) for i in range(num_spawns) 
                         for j in range(num_spawns) if i != j]
            random.shuffle(all_pairs)
            pairs_to_check = all_pairs[:self.max_candidates_to_analyze]
        else:
            print(f"  📋 将分析全部 {total_pairs} 个组合")
            pairs_to_check = [(i, j) for i in range(num_spawns) 
                              for j in range(num_spawns) if i != j]
        
        checked = 0
        filtered_distance = 0
        filtered_no_lights = 0
        filtered_too_many_lights = 0
        last_progress = 0
        
        for start_idx, end_idx in pairs_to_check:
            checked += 1
            
            # 进度显示
            progress = int(checked / len(pairs_to_check) * 100)
            if progress >= last_progress + 10:
                print(f"  📊 进度: {progress}% ({checked}/{len(pairs_to_check)}), "
                      f"有效: {len(candidates)}, "
                      f"距离不符: {filtered_distance}, "
                      f"无红绿灯: {filtered_no_lights}")
                last_progress = progress
            
            start_loc = self.spawn_points[start_idx].location
            end_loc = self.spawn_points[end_idx].location
            
            try:
                route = self._route_planner.trace_route(start_loc, end_loc)
                if not route or len(route) < 2:
                    continue
                
                # 分析路线
                route_info = self._analyze_single_route(route, start_idx, end_idx)
                
                # 距离筛选
                if route_info['distance'] < self.min_distance or \
                   route_info['distance'] > self.max_distance:
                    filtered_distance += 1
                    continue
                
                # 红绿灯数量筛选
                tl_count = route_info['traffic_light_count']
                if tl_count < self.min_traffic_lights:
                    filtered_no_lights += 1
                    continue
                
                if self.max_traffic_lights > 0 and tl_count > self.max_traffic_lights:
                    filtered_too_many_lights += 1
                    continue
                
                candidates.append(route_info)
                
            except Exception:
                pass
        
        print(f"\n  ✅ 分析完成:")
        print(f"     有效路线: {len(candidates)} 条")
        print(f"     距离不符: {filtered_distance} 条")
        print(f"     无红绿灯: {filtered_no_lights} 条")
        if filtered_too_many_lights > 0:
            print(f"     红绿灯过多: {filtered_too_many_lights} 条")
        
        return candidates
    
    def _analyze_single_route(self, route, start_idx: int, end_idx: int) -> Dict:
        """分析单条路线"""
        commands = {2: 0, 3: 0, 4: 0, 5: 0}
        waypoints = []
        distance = 0.0
        prev_cmd = None
        
        for i, (wp, road_option) in enumerate(route):
            if i > 0:
                distance += wp.transform.location.distance(
                    route[i-1][0].transform.location
                )
            waypoints.append((wp.transform.location.x, wp.transform.location.y))
            
            cmd_name = road_option.name if hasattr(road_option, 'name') else str(road_option)
            cmd = self.COMMAND_MAP.get(cmd_name, 2)
            if cmd != prev_cmd:
                commands[cmd] += 1
                prev_cmd = cmd
        
        # 计算经过的红绿灯数量
        traffic_light_count = self._count_traffic_lights_on_route(waypoints)
        traffic_light_indices = self._get_traffic_lights_on_route(waypoints)
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'distance': distance,
            'commands': commands,
            'waypoints': waypoints,
            'turn_count': commands[3] + commands[4],
            'traffic_light_count': traffic_light_count,
            'traffic_light_indices': traffic_light_indices,
        }
    
    def _deduplicate(self, routes: List[Dict]) -> List[Dict]:
        """
        路径去重（改进版）
        
        改进点：
        1. 按红绿灯数量分组去重，保证各组都有代表
        2. 使用更细的网格(5m)提高精度
        3. 组内按质量排序，组间轮流选择
        """
        if len(routes) <= 1:
            return routes
        
        # 1. 按红绿灯数量分组
        groups: Dict[int, List[Dict]] = {}
        for route in routes:
            tl_count = route.get('traffic_light_count', 0)
            if tl_count not in groups:
                groups[tl_count] = []
            groups[tl_count].append(route)
        
        print(f"  📊 去重前分组: {{{', '.join(f'{k}个灯:{len(v)}条' for k, v in sorted(groups.items()))}}}")
        
        # 2. 每组内部按质量排序
        for tl_count, group in groups.items():
            if self.prefer_more_lights:
                group.sort(key=lambda x: (
                    -x.get('turn_count', 0),
                    -x.get('distance', 0)
                ))
            else:
                group.sort(key=lambda x: (
                    -x.get('turn_count', 0),
                    x.get('distance', 0)
                ))
        
        # 3. 每组内部去重
        deduplicated_groups: Dict[int, List[Dict]] = {}
        total_removed = 0
        
        for tl_count, group in groups.items():
            deduped = []
            removed = 0
            
            for route in group:
                is_overlapping = False
                route_wps = route.get('waypoints', [])
                
                if route_wps:
                    for selected in deduped:
                        sel_wps = selected.get('waypoints', [])
                        if sel_wps and self._calc_overlap(route_wps, sel_wps) > self.overlap_threshold:
                            is_overlapping = True
                            removed += 1
                            break
                
                if not is_overlapping:
                    deduped.append(route)
            
            deduplicated_groups[tl_count] = deduped
            total_removed += removed
        
        # 4. 合并所有组（保持分组信息用于后续平衡选择）
        deduplicated = []
        for tl_count in sorted(deduplicated_groups.keys()):
            deduplicated.extend(deduplicated_groups[tl_count])
        
        # 保存分组信息供 _select_routes 使用
        self._deduplicated_groups = deduplicated_groups
        
        print(f"  🔄 去重完成: {len(routes)} → {len(deduplicated)} 条 (移除 {total_removed} 条)")
        print(f"  📊 去重后分组: {{{', '.join(f'{k}个灯:{len(v)}条' for k, v in sorted(deduplicated_groups.items()))}}}")
        
        return deduplicated
    
    def _calc_overlap(self, wps1: List, wps2: List, grid_size: float = 5.0) -> float:
        """
        计算路径重叠度
        
        改进：使用 5m 网格（原来是 10m），提高精度
        """
        def to_grid(wps):
            return set((int(x / grid_size), int(y / grid_size)) for x, y in wps)
        
        g1, g2 = to_grid(wps1), to_grid(wps2)
        if not g1 or not g2:
            return 0.0
        return len(g1 & g2) / len(g1 | g2)
    
    def _select_routes(self, candidates: List[Dict]) -> List[Dict]:
        """
        选择路线（改进版）
        
        改进点：
        1. 按红绿灯数量分层采样，保证各类路线都有代表
        2. 支持两种模式：均匀分布 / 按原比例
        """
        # 使用去重时保存的分组信息
        if hasattr(self, '_deduplicated_groups') and self._deduplicated_groups:
            groups = self._deduplicated_groups
        else:
            # 如果没有分组信息，重新分组
            groups: Dict[int, List[Dict]] = {}
            for c in candidates:
                count = c.get('traffic_light_count', 0)
                if count not in groups:
                    groups[count] = []
                groups[count].append(c)
        
        # 打印分布统计
        print(f"\n  📊 红绿灯数量分布:")
        for count in sorted(groups.keys()):
            print(f"     {count} 个红绿灯: {len(groups[count])} 条路线")
        
        total_candidates = sum(len(g) for g in groups.values())
        select_ratio = max(0.0, min(1.0, self.target_routes_ratio))
        
        if select_ratio >= 1.0:
            # 选择全部
            selected = candidates
            print(f"  ✅ 选择全部 {len(selected)} 条路线")
        else:
            # 分层采样：每组按比例选择，但保证每组至少选1条
            target_total = max(1, int(total_candidates * select_ratio))
            selected = []
            
            # 计算每组应选数量（按比例，但至少1条）
            group_targets = {}
            remaining = target_total
            
            for tl_count in sorted(groups.keys()):
                group_size = len(groups[tl_count])
                # 按比例计算，但至少选1条（如果该组有路线）
                if group_size > 0:
                    proportional = max(1, int(group_size * select_ratio))
                    group_targets[tl_count] = min(proportional, group_size)
                    remaining -= group_targets[tl_count]
            
            # 如果还有剩余配额，按组大小分配
            if remaining > 0:
                for tl_count in sorted(groups.keys(), key=lambda x: -len(groups[x])):
                    can_add = len(groups[tl_count]) - group_targets.get(tl_count, 0)
                    add = min(remaining, can_add)
                    if add > 0:
                        group_targets[tl_count] = group_targets.get(tl_count, 0) + add
                        remaining -= add
                    if remaining <= 0:
                        break
            
            # 从每组选择
            for tl_count in sorted(groups.keys()):
                target = group_targets.get(tl_count, 0)
                group = groups[tl_count]
                # 随机选择，增加多样性
                if len(group) > target:
                    selected.extend(random.sample(group, target))
                else:
                    selected.extend(group)
            
            print(f"  ✅ 分层采样 {select_ratio:.0%}:")
            for tl_count in sorted(groups.keys()):
                actual = len([s for s in selected if s.get('traffic_light_count', 0) == tl_count])
                print(f"     {tl_count} 个红绿灯: 选择 {actual}/{len(groups[tl_count])} 条")
            print(f"  📦 共选择 {len(selected)} 条路线")
        
        return selected

    def validate_route(self, start_idx: int, end_idx: int) -> Tuple[bool, Any, float, int]:
        """
        验证路线可行性
        
        返回:
            (是否有效, 路线, 距离, 红绿灯数量)
        """
        if self._route_planner is None:
            raise RuntimeError("❌ 路径规划器不可用")
        
        try:
            route = self._route_planner.trace_route(
                self.spawn_points[start_idx].location,
                self.spawn_points[end_idx].location
            )
            
            if not route:
                return False, None, 0.0, 0
            
            distance = sum(
                route[i][0].transform.location.distance(
                    route[i-1][0].transform.location
                )
                for i in range(1, len(route))
            )
            
            # 计算红绿灯数量
            waypoints = [(wp.transform.location.x, wp.transform.location.y) 
                         for wp, _ in route]
            tl_count = self._count_traffic_lights_on_route(waypoints)
            
            return True, route, distance, tl_count
        except Exception as e:
            print(f"⚠️ 路线验证失败: {e}")
            return False, None, 0.0, 0
    
    def trace_route(self, start_location, end_location):
        """规划路线"""
        if self._route_planner is None:
            raise RuntimeError("❌ 路径规划器不可用")
        return self._route_planner.trace_route(start_location, end_location)
    
    def _load_from_cache(self, cache_path: str) -> Optional[List[Tuple[int, int, float, int]]]:
        """从缓存加载路线"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            
            # 验证配置是否匹配
            mismatches = []
            
            if config.get('town') != self.town:
                mismatches.append(f"town: 缓存={config.get('town')}, 当前={self.town}")
            if config.get('min_distance') != self.min_distance:
                mismatches.append(f"min_distance: 缓存={config.get('min_distance')}, 当前={self.min_distance}")
            if config.get('max_distance') != self.max_distance:
                mismatches.append(f"max_distance: 缓存={config.get('max_distance')}, 当前={self.max_distance}")
            if config.get('min_traffic_lights') != self.min_traffic_lights:
                mismatches.append(f"min_traffic_lights: 缓存={config.get('min_traffic_lights')}, 当前={self.min_traffic_lights}")
            if config.get('max_traffic_lights') != self.max_traffic_lights:
                mismatches.append(f"max_traffic_lights: 缓存={config.get('max_traffic_lights')}, 当前={self.max_traffic_lights}")
            if config.get('traffic_light_radius') != self.traffic_light_radius:
                mismatches.append(f"traffic_light_radius: 缓存={config.get('traffic_light_radius')}, 当前={self.traffic_light_radius}")
            
            # 验证 spawn_points 数量
            cached_spawn_count = config.get('num_spawn_points', 0)
            if cached_spawn_count != len(self.spawn_points):
                mismatches.append(f"spawn_points: 缓存={cached_spawn_count}, 当前={len(self.spawn_points)}")
            
            # 验证红绿灯数量
            cached_tl_count = config.get('num_traffic_lights', 0)
            if cached_tl_count != len(self._traffic_light_locations):
                mismatches.append(f"traffic_lights: 缓存={cached_tl_count}, 当前={len(self._traffic_light_locations)}")
            
            if mismatches:
                print(f"⚠️ 缓存配置不匹配，重新生成:")
                for m in mismatches:
                    print(f"   - {m}")
                return None
            
            routes = data.get('routes', [])
            return [(r['start'], r['end'], r['distance'], r['traffic_light_count']) 
                    for r in routes]
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
            return None
    
    def _save_to_cache(self, routes: List[Tuple[int, int, float, int]], cache_path: str):
        """保存路线到缓存"""
        try:
            data = {
                'config': {
                    'town': self.town,
                    'num_spawn_points': len(self.spawn_points),
                    'num_traffic_lights': len(self._traffic_light_locations),
                    'min_distance': self.min_distance,
                    'max_distance': self.max_distance,
                    'min_traffic_lights': self.min_traffic_lights,
                    'max_traffic_lights': self.max_traffic_lights,
                    'traffic_light_radius': self.traffic_light_radius,
                    'overlap_threshold': self.overlap_threshold,
                    'target_routes_ratio': self.target_routes_ratio,
                },
                'routes': [
                    {'start': s, 'end': e, 'distance': d, 'traffic_light_count': tl}
                    for s, e, d, tl in routes
                ],
                'generated_at': datetime.now().isoformat(),
                'total_routes': len(routes)
            }
            
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 红绿灯路线已缓存到: {cache_path}")
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
    
    def _print_statistics(self, routes: List[Tuple[int, int, float, int]]):
        """打印路线统计"""
        if not routes:
            return
        
        distances = [d for _, _, d, _ in routes]
        tl_counts = [tl for _, _, _, tl in routes]
        
        print(f"\n📊 红绿灯路线统计:")
        print(f"  • 总路线数: {len(routes)}")
        print(f"  • 平均距离: {np.mean(distances):.1f}m")
        print(f"  • 平均红绿灯数: {np.mean(tl_counts):.1f}")
        print(f"  • 红绿灯范围: {min(tl_counts)} ~ {max(tl_counts)}")
        print(f"  • 预计耗时: {len(routes) * 2:.0f}分钟")
    
    def get_traffic_light_locations(self) -> List[Tuple[float, float, float]]:
        """获取所有红绿灯位置"""
        return self._traffic_light_locations.copy()
    
    def refresh_traffic_lights(self):
        """刷新红绿灯缓存"""
        self._cache_traffic_lights()


# ==================== 配置类 ====================

from dataclasses import dataclass


@dataclass
class TrafficLightRouteConfig:
    """红绿灯路线配置"""
    # 基础路线参数
    min_distance: float = 100.0
    max_distance: float = 400.0
    overlap_threshold: float = 0.5
    target_routes_ratio: float = 1.0
    max_candidates_to_analyze: int = 0
    
    # 红绿灯相关参数
    min_traffic_lights: int = 1
    max_traffic_lights: int = 0  # 0 = 不限制
    traffic_light_radius: float = 30.0
    prefer_more_lights: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrafficLightRouteConfig':
        """从字典创建"""
        return cls(
            min_distance=data.get('min_distance', 100.0),
            max_distance=data.get('max_distance', 400.0),
            overlap_threshold=data.get('overlap_threshold', 0.5),
            target_routes_ratio=data.get('target_routes_ratio', 1.0),
            max_candidates_to_analyze=data.get('max_candidates_to_analyze', 0),
            min_traffic_lights=data.get('min_traffic_lights', 1),
            max_traffic_lights=data.get('max_traffic_lights', 0),
            traffic_light_radius=data.get('traffic_light_radius', 30.0),
            prefer_more_lights=data.get('prefer_more_lights', True),
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'min_distance': self.min_distance,
            'max_distance': self.max_distance,
            'overlap_threshold': self.overlap_threshold,
            'target_routes_ratio': self.target_routes_ratio,
            'max_candidates_to_analyze': self.max_candidates_to_analyze,
            'min_traffic_lights': self.min_traffic_lights,
            'max_traffic_lights': self.max_traffic_lights,
            'traffic_light_radius': self.traffic_light_radius,
            'prefer_more_lights': self.prefer_more_lights,
        }


# ==================== 便捷函数 ====================

def create_traffic_light_route_planner(world, spawn_points: List, 
                                        town: str = None,
                                        config: TrafficLightRouteConfig = None) -> TrafficLightRoutePlanner:
    """
    创建红绿灯路线规划器的便捷函数
    
    参数:
        world: CARLA world 对象
        spawn_points: 生成点列表
        town: 地图名称
        config: 配置对象
        
    返回:
        TrafficLightRoutePlanner 实例
    """
    planner = TrafficLightRoutePlanner(world, spawn_points, town)
    
    if config:
        planner.configure(
            min_distance=config.min_distance,
            max_distance=config.max_distance,
            overlap_threshold=config.overlap_threshold,
            target_routes_ratio=config.target_routes_ratio,
            max_candidates=config.max_candidates_to_analyze,
            min_traffic_lights=config.min_traffic_lights,
            max_traffic_lights=config.max_traffic_lights,
            traffic_light_radius=config.traffic_light_radius,
            prefer_more_lights=config.prefer_more_lights,
        )
    
    return planner
