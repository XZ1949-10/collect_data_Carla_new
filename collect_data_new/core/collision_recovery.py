#!/usr/bin/env python
# coding=utf-8
"""
碰撞恢复模块

负责碰撞后的恢复点查找和车辆重生逻辑。
"""

from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


@dataclass
class RecoveryConfig:
    """碰撞恢复配置"""
    enabled: bool = True
    max_collisions_per_route: int = 99
    min_distance_to_destination: float = 30.0
    recovery_skip_distance: float = 25.0


class CollisionRecoveryManager:
    """碰撞恢复管理器"""
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        """
        初始化碰撞恢复管理器
        
        参数:
            config: 恢复配置
        """
        self.config = config or RecoveryConfig()
        
        # 当前路线信息
        self._route_waypoints: List = []
        self._destination = None
        self._destination_index: Optional[int] = None
        
        # 碰撞计数
        self._collision_count = 0
    
    def configure(self, enabled: bool = True, max_collisions: int = 99,
                  min_distance: float = 30.0, skip_distance: float = 25.0):
        """配置恢复参数"""
        self.config.enabled = enabled
        self.config.max_collisions_per_route = max_collisions
        self.config.min_distance_to_destination = min_distance
        self.config.recovery_skip_distance = skip_distance
    
    def set_route(self, waypoints: List, destination, destination_index: int):
        """
        设置当前路线信息
        
        参数:
            waypoints: 路线waypoints列表 [(waypoint, road_option), ...]
            destination: 终点位置
            destination_index: 终点spawn_point索引
        """
        self._route_waypoints = list(waypoints) if waypoints else []
        self._destination = destination
        self._destination_index = destination_index
        self._collision_count = 0
    
    def reset(self):
        """重置状态"""
        self._route_waypoints = []
        self._destination = None
        self._destination_index = None
        self._collision_count = 0
    
    @property
    def collision_count(self) -> int:
        """当前碰撞次数"""
        return self._collision_count
    
    @property
    def can_recover(self) -> bool:
        """是否可以恢复"""
        if not self.config.enabled:
            return False
        return self._collision_count < self.config.max_collisions_per_route
    
    @property
    def destination_index(self) -> Optional[int]:
        """终点索引"""
        return self._destination_index
    
    def increment_collision(self):
        """增加碰撞计数"""
        self._collision_count += 1
    
    def get_recovery_transform(self, vehicle_location) -> Optional[Any]:
        """
        获取恢复点transform
        
        参数:
            vehicle_location: 当前车辆位置
            
        返回:
            carla.Transform 或 None
        """
        if not self.config.enabled:
            return None
        
        if self._destination is None:
            return None
        
        # 优先从路线waypoints查找
        if self._route_waypoints and len(self._route_waypoints) > 0:
            return self._find_recovery_from_waypoints(vehicle_location)
        
        return None
    
    def _find_recovery_from_waypoints(self, vehicle_location) -> Optional[Any]:
        """从路线waypoints中查找恢复点"""
        if not self._route_waypoints:
            return None
        
        # 计算到终点的距离
        dist_to_dest = vehicle_location.distance(self._destination)
        
        # 如果已经很接近终点，不需要恢复
        if dist_to_dest < self.config.min_distance_to_destination:
            print(f"  ⚠️ 距终点仅 {dist_to_dest:.1f}m，不需要恢复")
            return None
        
        # 找到当前位置最近的waypoint索引
        min_dist = float('inf')
        current_idx = 0
        for i, (wp, _) in enumerate(self._route_waypoints):
            dist = vehicle_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                current_idx = i
        
        # 沿路线向前累积距离，跳过碰撞区域
        recovery_idx = current_idx
        accumulated_dist = 0.0
        
        while recovery_idx < len(self._route_waypoints) - 1:
            wp1 = self._route_waypoints[recovery_idx][0]
            wp2 = self._route_waypoints[recovery_idx + 1][0]
            segment_dist = wp1.transform.location.distance(wp2.transform.location)
            accumulated_dist += segment_dist
            recovery_idx += 1
            
            if accumulated_dist >= self.config.recovery_skip_distance:
                break
        
        # 检查是否还有足够的路线剩余
        if recovery_idx >= len(self._route_waypoints) - 1:
            print(f"  ⚠️ 路线剩余不足，无法恢复")
            return None
        
        # 获取恢复点的transform
        recovery_wp = self._route_waypoints[recovery_idx][0]
        recovery_transform = recovery_wp.transform
        
        # 检查恢复点到终点的距离
        recovery_to_dest = recovery_transform.location.distance(self._destination)
        if recovery_to_dest < self.config.min_distance_to_destination:
            print(f"  ⚠️ 恢复点距终点仅 {recovery_to_dest:.1f}m，不需要恢复")
            return None
        
        print(f"  📍 恢复点: waypoint[{recovery_idx}], "
              f"跳过 {accumulated_dist:.1f}m, 距终点 {recovery_to_dest:.1f}m")
        
        # 更新waypoints列表，移除已经走过的部分
        self._route_waypoints = self._route_waypoints[recovery_idx:]
        
        return recovery_transform
    
    def update_waypoints_from_agent(self, agent) -> bool:
        """
        从agent更新waypoints
        
        参数:
            agent: BasicAgent实例
            
        返回:
            是否成功更新
        """
        if agent is None:
            return False
        
        try:
            if hasattr(agent, 'get_local_planner'):
                local_planner = agent.get_local_planner()
                plan = list(local_planner.get_plan())
                if plan and len(plan) > 0:
                    self._route_waypoints = plan
                    return True
        except Exception as e:
            print(f"  ⚠️ 从agent获取路线失败: {e}")
        
        return False


def adjust_spawn_transform(transform, height_offset: float = 0.5):
    """
    调整生成位置（抬高避免碰撞）
    
    参数:
        transform: 原始transform
        height_offset: 抬高高度
        
    返回:
        调整后的transform
    """
    if not CARLA_AVAILABLE:
        return transform
    
    return carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + height_offset
        ),
        transform.rotation
    )
