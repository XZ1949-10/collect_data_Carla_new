#!/usr/bin/env python
# coding=utf-8
"""
路线规划模块

负责路线生成、分析、去重和选择策略。
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


class RoutePlanner:
    """路线规划器"""
    
    # 命令映射
    COMMAND_MAP = {
        'LANEFOLLOW': 2, 'LEFT': 3, 'RIGHT': 4, 'STRAIGHT': 5,
        'CHANGELANELEFT': 2, 'CHANGELANERIGHT': 2
    }
    
    def __init__(self, world, spawn_points: List, town: str = None):
        """
        初始化路线规划器
        
        参数:
            world: CARLA world 对象
            spawn_points: 生成点列表
            town: 地图名称（用于缓存验证）
        """
        self.world = world
        self.spawn_points = spawn_points
        self._route_planner = None
        
        # 地图名称（用于缓存验证）
        self.town = town or self._get_map_name()
        
        # 路线生成参数
        self.min_distance = 50.0
        self.max_distance = 500.0
        self.overlap_threshold = 0.5
        self.turn_priority_ratio = 0.7
        self.target_routes_ratio = 1.0
        self.max_candidates_to_analyze = 0
        
        self._init_route_planner()
    
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
                "请确保 CARLA PythonAPI 的 agents 模块已正确安装。\n"
                "通常位于: CARLA_ROOT/PythonAPI/carla/agents/\n"
                "需要将 CARLA_ROOT/PythonAPI/carla 添加到 PYTHONPATH"
            )
        
        try:
            self._route_planner = GlobalRoutePlanner(
                self.world.get_map(), sampling_resolution=2.0
            )
            print("✅ 路径规划器初始化成功")
        except Exception as e:
            raise RuntimeError(f"❌ 路径规划器初始化失败: {e}")
    
    def configure(self, min_distance: float = 50.0, max_distance: float = 500.0,
                  overlap_threshold: float = 0.5, turn_priority_ratio: float = 0.7,
                  target_routes_ratio: float = 1.0, max_candidates: int = 0,
                  town: str = None):
        """配置路线生成参数"""
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.overlap_threshold = overlap_threshold
        self.turn_priority_ratio = turn_priority_ratio
        self.target_routes_ratio = target_routes_ratio
        self.max_candidates_to_analyze = max_candidates
        if town is not None:
            self.town = town
    
    def generate_routes(self, strategy: str = 'smart', 
                        cache_path: Optional[str] = None) -> List[Tuple[int, int, float]]:
        """
        生成路线对
        
        参数:
            strategy: 生成策略 ('smart' 或 'exhaustive')
            cache_path: 缓存文件路径
            
        返回:
            路线列表 [(start_idx, end_idx, distance), ...]
        """
        print("\n" + "="*70)
        print("🛣️ 生成路线对")
        print("="*70)
        
        # 尝试从缓存加载
        if cache_path and os.path.exists(cache_path):
            routes = self._load_from_cache(cache_path)
            if routes:
                print(f"✅ 从缓存加载了 {len(routes)} 条路线")
                self._print_statistics(routes)
                return routes
        
        # 生成新路线
        if strategy == 'smart':
            routes = self._generate_smart_routes()
        else:
            routes = self._generate_exhaustive_routes()
        
        if routes:
            self._print_statistics(routes)
            if cache_path:
                self._save_to_cache(routes, cache_path, strategy)
        
        return routes
    
    def _generate_smart_routes(self) -> List[Tuple[int, int, float]]:
        """智能路线生成"""
        print(f"策略: 🧠 智能选择")
        
        if not AGENTS_AVAILABLE or self._route_planner is None:
            raise RuntimeError("❌ 路径规划器不可用，无法生成路线")
        
        # 1. 分析候选路线
        candidates = self._analyze_candidates()
        if not candidates:
            return []
        
        # 2. 去重
        deduplicated = self._deduplicate(candidates)
        if not deduplicated:
            return []
        
        # 3. 按比例选择
        selected = self._select_balanced(deduplicated)
        return selected
    
    def _analyze_candidates(self) -> List[Dict]:
        """分析候选路线"""
        print("\n🔍 分析候选路线...")
        print(f"  📏 路径距离范围: {self.min_distance:.0f}m ~ {self.max_distance:.0f}m")
        
        candidates = []
        num_spawns = len(self.spawn_points)
        total_pairs = num_spawns * (num_spawns - 1)
        
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
            pairs_to_check = [(i, j) for i in range(num_spawns) 
                              for j in range(num_spawns) if i != j]
            print(f"  📋 共 {len(pairs_to_check)} 个组合待分析")
        
        checked = 0
        filtered = 0
        last_progress = 0
        
        for start_idx, end_idx in pairs_to_check:
            checked += 1
            
            # 进度显示
            progress = int(checked / len(pairs_to_check) * 100)
            if progress >= last_progress + 10:
                print(f"  📊 进度: {progress}% ({checked}/{len(pairs_to_check)}), "
                      f"有效: {len(candidates)}, 距离不符: {filtered}")
                last_progress = progress
            
            start_loc = self.spawn_points[start_idx].location
            end_loc = self.spawn_points[end_idx].location
            
            try:
                route = self._route_planner.trace_route(start_loc, end_loc)
                if not route or len(route) < 2:
                    continue
                
                # 分析路线
                route_info = self._analyze_single_route(route, start_idx, end_idx)
                
                if route_info['distance'] < self.min_distance or \
                   route_info['distance'] > self.max_distance:
                    filtered += 1
                    continue
                
                candidates.append(route_info)
                
            except Exception:
                pass
        
        print(f"  ✅ 分析完成: 有效 {len(candidates)} 条, 距离不符 {filtered} 条")
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
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'distance': distance,
            'commands': commands,
            'waypoints': waypoints,
            'turn_count': commands[3] + commands[4]
        }
    
    def _deduplicate(self, routes: List[Dict]) -> List[Dict]:
        """路径去重"""
        if len(routes) <= 1:
            return routes
        
        # 按转弯次数排序
        routes_copy = routes.copy()
        routes_copy.sort(key=lambda x: (-x.get('turn_count', 0), -x.get('distance', 0)))
        
        deduplicated = []
        removed = 0
        
        for route in routes_copy:
            is_overlapping = False
            route_wps = route.get('waypoints', [])
            
            if route_wps:
                for selected in deduplicated:
                    sel_wps = selected.get('waypoints', [])
                    if sel_wps and self._calc_overlap(route_wps, sel_wps) > self.overlap_threshold:
                        is_overlapping = True
                        removed += 1
                        break
            
            if not is_overlapping:
                deduplicated.append(route)
        
        print(f"  🔄 去重完成: {len(routes)} → {len(deduplicated)} 条 (移除 {removed} 条)")
        return deduplicated
    
    def _calc_overlap(self, wps1: List, wps2: List, grid_size: float = 10.0) -> float:
        """计算路径重叠度"""
        def to_grid(wps):
            return set((int(x / grid_size), int(y / grid_size)) for x, y in wps)
        
        g1, g2 = to_grid(wps1), to_grid(wps2)
        if not g1 or not g2:
            return 0.0
        return len(g1 & g2) / len(g1 | g2)
    
    def _select_balanced(self, candidates: List[Dict]) -> List[Tuple[int, int, float]]:
        """按比例选择路线"""
        turn_routes = [c for c in candidates if c.get('turn_count', 0) > 0]
        straight_routes = [c for c in candidates if c.get('turn_count', 0) == 0]
        
        turn_routes.sort(key=lambda x: (-x.get('turn_count', 0), -x.get('distance', 0)))
        straight_routes.sort(key=lambda x: -x.get('distance', 0))
        
        print(f"  📊 去重后: 转弯 {len(turn_routes)} 条, 直行 {len(straight_routes)} 条")
        
        turn_ratio = max(0.0, min(1.0, self.turn_priority_ratio))
        
        if turn_ratio >= 0.5:
            max_turn = len(turn_routes)
            max_straight = int(max_turn * (1 - turn_ratio) / turn_ratio) if turn_ratio < 1.0 else 0
            max_straight = min(max_straight, len(straight_routes))
        else:
            max_straight = len(straight_routes)
            max_turn = int(max_straight * turn_ratio / (1 - turn_ratio)) if turn_ratio > 0.0 else 0
            max_turn = min(max_turn, len(turn_routes))
        
        select_ratio = max(0.0, min(1.0, self.target_routes_ratio))
        actual_turn = int(max_turn * select_ratio)
        actual_straight = int(max_straight * select_ratio)
        
        if select_ratio > 0:
            if max_turn > 0 and actual_turn == 0:
                actual_turn = 1
            if max_straight > 0 and actual_straight == 0:
                actual_straight = 1
        
        selected = turn_routes[:actual_turn] + straight_routes[:actual_straight]
        
        if selected:
            ratio = actual_turn / len(selected)
            print(f"  ✅ 最终选择: 转弯 {actual_turn} ({ratio:.1%}), "
                  f"直行 {actual_straight} ({1-ratio:.1%})")
        
        result = [(r['start_idx'], r['end_idx'], r.get('distance', 0)) for r in selected]
        random.shuffle(result)
        return result

    def _generate_exhaustive_routes(self) -> List[Tuple[int, int, float]]:
        """穷举路线生成"""
        print(f"策略: 📋 穷举模式")
        
        routes = []
        num_spawns = len(self.spawn_points)
        total_pairs = num_spawns * (num_spawns - 1)
        
        print(f"  正在分析 {total_pairs} 个组合...")
        print(f"  📏 路径距离范围: {self.min_distance:.0f}m ~ {self.max_distance:.0f}m")
        
        checked = 0
        unreachable = 0
        filtered = 0
        
        for start_idx, sp in enumerate(self.spawn_points):
            for end_idx, ep in enumerate(self.spawn_points):
                if start_idx == end_idx:
                    continue
                
                checked += 1
                
                try:
                    route = self._route_planner.trace_route(sp.location, ep.location)
                    if route and len(route) >= 2:
                        distance = sum(
                            route[i][0].transform.location.distance(
                                route[i-1][0].transform.location
                            )
                            for i in range(1, len(route))
                        )
                        if self.min_distance <= distance <= self.max_distance:
                            routes.append((start_idx, end_idx, distance))
                        else:
                            filtered += 1
                    else:
                        unreachable += 1
                except:
                    unreachable += 1
            
            if (start_idx + 1) % 50 == 0 or start_idx == num_spawns - 1:
                print(f"  进度: {start_idx + 1}/{num_spawns}, "
                      f"有效: {len(routes)}, 距离不符: {filtered}")
        
        print(f"  ✅ 穷举完成，共 {len(routes)} 条有效路线")
        
        # 按比例选择
        select_ratio = max(0.0, min(1.0, self.target_routes_ratio))
        if select_ratio < 1.0:
            random.shuffle(routes)
            target = max(1, int(len(routes) * select_ratio))
            routes = routes[:target]
            print(f"  📊 按比例选择 {select_ratio:.0%}，共 {len(routes)} 条")
        else:
            random.shuffle(routes)
        
        return routes
    
    def _calc_distance(self, loc1, loc2) -> float:
        """计算两点直线距离（仅用于调试）"""
        return np.sqrt((loc2.x - loc1.x)**2 + (loc2.y - loc1.y)**2)
    
    def validate_route(self, start_idx: int, end_idx: int) -> Tuple[bool, Any, float]:
        """验证路线可行性"""
        if self._route_planner is None:
            raise RuntimeError("❌ 路径规划器不可用")
        
        try:
            route = self._route_planner.trace_route(
                self.spawn_points[start_idx].location,
                self.spawn_points[end_idx].location
            )
            
            if not route:
                return False, None, 0.0
            
            distance = sum(
                route[i][0].transform.location.distance(
                    route[i-1][0].transform.location
                )
                for i in range(1, len(route))
            )
            return True, route, distance
        except Exception as e:
            print(f"⚠️ 路线验证失败: {e}")
            return False, None, 0.0
    
    def trace_route(self, start_location, end_location):
        """规划路线"""
        if self._route_planner is None:
            raise RuntimeError("❌ 路径规划器不可用")
        return self._route_planner.trace_route(start_location, end_location)
    
    def _load_from_cache(self, cache_path: str) -> Optional[List[Tuple[int, int, float]]]:
        """从缓存加载路线"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            
            # 验证核心配置是否匹配（这些参数影响路线生成结果）
            mismatches = []
            
            if config.get('town') != self.town:
                mismatches.append(f"town: 缓存={config.get('town')}, 当前={self.town}")
            if config.get('min_distance') != self.min_distance:
                mismatches.append(f"min_distance: 缓存={config.get('min_distance')}, 当前={self.min_distance}")
            if config.get('max_distance') != self.max_distance:
                mismatches.append(f"max_distance: 缓存={config.get('max_distance')}, 当前={self.max_distance}")
            
            # 验证影响路线选择的参数
            if config.get('overlap_threshold') != self.overlap_threshold:
                mismatches.append(f"overlap_threshold: 缓存={config.get('overlap_threshold')}, 当前={self.overlap_threshold}")
            if config.get('turn_priority_ratio') != self.turn_priority_ratio:
                mismatches.append(f"turn_priority_ratio: 缓存={config.get('turn_priority_ratio')}, 当前={self.turn_priority_ratio}")
            
            # 注意：target_routes_ratio 不需要验证，因为它只影响最终选择的数量，
            # 用户可能想用不同的比例从同一个候选集中选择
            
            if mismatches:
                print(f"⚠️ 缓存配置不匹配，重新生成:")
                for m in mismatches:
                    print(f"   - {m}")
                return None
            
            # 验证 spawn_points 数量是否一致
            cached_spawn_count = config.get('num_spawn_points', 0)
            if cached_spawn_count > 0 and cached_spawn_count != len(self.spawn_points):
                print(f"⚠️ spawn_points 数量不匹配 (缓存: {cached_spawn_count}, 当前: {len(self.spawn_points)})，重新生成")
                return None
            
            routes = data.get('routes', [])
            return [(r['start'], r['end'], r['distance']) for r in routes]
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
            return None
    
    def _save_to_cache(self, routes: List[Tuple[int, int, float]], 
                       cache_path: str, strategy: str):
        """保存路线到缓存"""
        try:
            data = {
                'config': {
                    'town': self.town,
                    'num_spawn_points': len(self.spawn_points),  # 添加 spawn_points 数量
                    'min_distance': self.min_distance,
                    'max_distance': self.max_distance,
                    'strategy': strategy,
                    'overlap_threshold': self.overlap_threshold,
                    'turn_priority_ratio': self.turn_priority_ratio,
                    'target_routes_ratio': self.target_routes_ratio
                },
                'routes': [
                    {'start': s, 'end': e, 'distance': d}
                    for s, e, d in routes
                ],
                'generated_at': datetime.now().isoformat(),
                'total_routes': len(routes)
            }
            
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 路线已缓存到: {cache_path}")
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
    
    def _print_statistics(self, routes: List[Tuple[int, int, float]]):
        """打印路线统计"""
        distances = [d for _, _, d in routes]
        print(f"\n📊 路线统计:")
        print(f"  • 总路线数: {len(routes)}")
        print(f"  • 平均距离: {np.mean(distances):.1f}m")
        print(f"  • 预计耗时: {len(routes) * 2:.0f}分钟")
