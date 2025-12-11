#!/usr/bin/env python
# coding=utf-8
"""
碰撞处理器

处理 CARLA 车辆碰撞事件。
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class CollisionEvent:
    """碰撞事件数据"""
    frame: int
    other_actor_type: str
    impulse: tuple  # (x, y, z)
    timestamp: float = 0.0


class CollisionHandler:
    """
    碰撞处理器
    
    特性：
    - 记录碰撞历史
    - 支持碰撞回调
    - 可配置碰撞过滤
    """
    
    def __init__(self, on_collision: Optional[Callable[[CollisionEvent], None]] = None):
        """
        初始化碰撞处理器
        
        参数:
            on_collision: 碰撞回调函数
        """
        self._collision_detected = False
        self._collision_history: List[CollisionEvent] = []
        self._on_collision_callback = on_collision
        
        # 忽略的碰撞类型（如静态物体）
        self._ignored_types: List[str] = []
    
    @property
    def collision_detected(self) -> bool:
        """是否检测到碰撞"""
        return self._collision_detected
    
    @property
    def collision_history(self) -> List[CollisionEvent]:
        """碰撞历史"""
        return self._collision_history.copy()
    
    @property
    def last_collision(self) -> Optional[CollisionEvent]:
        """最后一次碰撞"""
        return self._collision_history[-1] if self._collision_history else None
    
    def handle_collision(self, event) -> None:
        """
        处理碰撞事件（作为 CARLA 传感器回调）
        
        参数:
            event: CARLA 碰撞事件
        """
        other_actor = event.other_actor
        actor_type = other_actor.type_id if other_actor else "unknown"
        
        # 检查是否应该忽略
        if any(ignored in actor_type for ignored in self._ignored_types):
            return
        
        # 创建碰撞事件
        # 尝试从 CARLA event 获取 frame，否则使用 0
        frame_number = getattr(event, 'frame', 0)
        
        collision_event = CollisionEvent(
            frame=frame_number,
            other_actor_type=actor_type,
            impulse=(
                event.normal_impulse.x,
                event.normal_impulse.y,
                event.normal_impulse.z
            ),
            timestamp=time.time()
        )
        
        self._collision_detected = True
        self._collision_history.append(collision_event)
        
        print(f"💥 检测到碰撞！碰撞对象: {actor_type}")
        
        # 调用回调
        if self._on_collision_callback:
            self._on_collision_callback(collision_event)
    
    def reset(self) -> None:
        """重置碰撞状态"""
        self._collision_detected = False
    
    def clear_history(self) -> None:
        """清空碰撞历史"""
        self._collision_history.clear()
        self._collision_detected = False
    
    def add_ignored_type(self, actor_type: str) -> None:
        """添加忽略的碰撞类型"""
        if actor_type not in self._ignored_types:
            self._ignored_types.append(actor_type)
    
    def remove_ignored_type(self, actor_type: str) -> None:
        """移除忽略的碰撞类型"""
        if actor_type in self._ignored_types:
            self._ignored_types.remove(actor_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取碰撞统计"""
        if not self._collision_history:
            return {'total_collisions': 0, 'by_type': {}}
        
        by_type: Dict[str, int] = {}
        for event in self._collision_history:
            actor_type = event.other_actor_type
            by_type[actor_type] = by_type.get(actor_type, 0) + 1
        
        return {
            'total_collisions': len(self._collision_history),
            'by_type': by_type
        }
