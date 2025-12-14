#!/usr/bin/env python
# coding=utf-8
"""
CARLA Actor 工具模块

提供统一的 Actor 有效性检查和安全销毁功能。
解决重复销毁导致的 "unable to destroy actor: not found" 错误。
"""

import time
from typing import List, Optional, Set, Any

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class ActorRegistry:
    """
    Actor 注册表
    
    统一追踪所有创建的 actor，避免重复销毁。
    使用单例模式确保全局唯一。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._destroyed_ids: Set[int] = set()
            cls._instance._pending_destroy: Set[int] = set()
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ActorRegistry':
        """获取单例实例"""
        return cls()
    
    def mark_destroyed(self, actor_id: int):
        """标记 actor 已销毁"""
        self._destroyed_ids.add(actor_id)
        self._pending_destroy.discard(actor_id)
    
    def mark_pending_destroy(self, actor_id: int):
        """标记 actor 待销毁"""
        if actor_id not in self._destroyed_ids:
            self._pending_destroy.add(actor_id)
    
    def is_destroyed(self, actor_id: int) -> bool:
        """检查 actor 是否已被销毁"""
        return actor_id in self._destroyed_ids
    
    def clear(self):
        """清空注册表（新场景时调用）"""
        self._destroyed_ids.clear()
        self._pending_destroy.clear()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'destroyed_count': len(self._destroyed_ids),
            'pending_count': len(self._pending_destroy),
        }


def is_actor_alive(actor) -> bool:
    """
    检查 actor 是否仍然有效
    
    参数:
        actor: CARLA actor 对象
        
    返回:
        bool: True 表示有效，False 表示无效或已销毁
    """
    if actor is None:
        return False
    
    # 检查注册表
    registry = ActorRegistry.get_instance()
    if registry.is_destroyed(actor.id):
        return False
    
    # 检查 actor 的 is_alive 属性
    try:
        return actor.is_alive
    except Exception:
        # 如果访问 is_alive 失败，说明 actor 已无效
        return False


def safe_destroy_actor(actor, wait_time: float = 0.0, silent: bool = False) -> bool:
    """
    安全销毁单个 actor
    
    在销毁前检查 actor 是否有效，避免 "not found" 错误。
    
    参数:
        actor: 要销毁的 actor
        wait_time: 销毁后等待时间（秒）
        silent: 是否静默模式（不打印警告）
        
    返回:
        bool: 是否成功（包括 actor 已经不存在的情况）
    """
    if actor is None:
        return True
    
    registry = ActorRegistry.get_instance()
    actor_id = actor.id
    
    # 检查是否已销毁
    if registry.is_destroyed(actor_id):
        return True
    
    # 检查 actor 是否有效
    if not is_actor_alive(actor):
        registry.mark_destroyed(actor_id)
        return True
    
    # 尝试销毁
    try:
        actor.destroy()
        registry.mark_destroyed(actor_id)
        if wait_time > 0:
            time.sleep(wait_time)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if 'not found' in error_msg or 'does not exist' in error_msg:
            # actor 已经不存在，标记为已销毁
            registry.mark_destroyed(actor_id)
            return True
        else:
            if not silent:
                print(f"⚠️ 销毁 actor {actor_id} 失败: {e}")
            return False


def safe_stop_sensor(sensor, silent: bool = False) -> bool:
    """
    安全停止传感器
    
    参数:
        sensor: 传感器 actor
        silent: 是否静默模式
        
    返回:
        bool: 是否成功
    """
    if sensor is None:
        return True
    
    if not is_actor_alive(sensor):
        return True
    
    try:
        sensor.stop()
        return True
    except Exception as e:
        if not silent:
            error_msg = str(e).lower()
            if 'not found' not in error_msg:
                print(f"⚠️ 停止传感器失败: {e}")
        return False


def safe_destroy_sensor(sensor, wait_time: float = 0.0, silent: bool = False) -> bool:
    """
    安全销毁传感器（先停止再销毁）
    
    参数:
        sensor: 传感器 actor
        wait_time: 销毁后等待时间
        silent: 是否静默模式
        
    返回:
        bool: 是否成功
    """
    if sensor is None:
        return True
    
    # 先停止
    safe_stop_sensor(sensor, silent=True)
    
    # 再销毁
    return safe_destroy_actor(sensor, wait_time, silent)


def batch_destroy_actors(client, actors: List, silent: bool = False) -> int:
    """
    批量销毁 actors（使用 CARLA 批量命令）
    
    在销毁前过滤掉无效的 actor，避免错误。
    
    参数:
        client: CARLA client 对象
        actors: actor 列表
        silent: 是否静默模式
        
    返回:
        int: 成功销毁的数量
    """
    if not actors:
        return 0
    
    if not CARLA_AVAILABLE:
        return 0
    
    registry = ActorRegistry.get_instance()
    
    # 过滤有效的 actors
    valid_actors = []
    for actor in actors:
        if actor is None:
            continue
        if registry.is_destroyed(actor.id):
            continue
        if is_actor_alive(actor):
            valid_actors.append(actor)
        else:
            registry.mark_destroyed(actor.id)
    
    if not valid_actors:
        return 0
    
    # 批量销毁
    try:
        batch = [carla.command.DestroyActor(a) for a in valid_actors]
        results = client.apply_batch_sync(batch, False)
        
        destroyed_count = 0
        for i, result in enumerate(results):
            actor_id = valid_actors[i].id
            if result.error:
                error_msg = str(result.error).lower()
                if 'not found' in error_msg or 'does not exist' in error_msg:
                    # actor 已经不存在
                    registry.mark_destroyed(actor_id)
                    destroyed_count += 1
                elif not silent:
                    print(f"⚠️ 批量销毁 actor {actor_id} 失败: {result.error}")
            else:
                registry.mark_destroyed(actor_id)
                destroyed_count += 1
        
        return destroyed_count
        
    except Exception as e:
        if not silent:
            print(f"⚠️ 批量销毁失败: {e}")
        
        # 降级为逐个销毁
        destroyed_count = 0
        for actor in valid_actors:
            if safe_destroy_actor(actor, silent=silent):
                destroyed_count += 1
        
        return destroyed_count


def destroy_all_resources(
    client,
    sensors: List = None,
    vehicle = None,
    wait_time: float = 0.5,
    silent: bool = False
) -> dict:
    """
    统一销毁所有资源
    
    按正确顺序销毁：传感器 → 车辆
    
    参数:
        client: CARLA client 对象
        sensors: 传感器列表
        vehicle: 车辆
        wait_time: 销毁后等待时间
        silent: 是否静默模式
        
    返回:
        dict: 销毁统计 {'sensors': int, 'vehicle': bool}
    """
    result = {'sensors': 0, 'vehicle': False}
    
    # 1. 停止所有传感器
    if sensors:
        for sensor in sensors:
            safe_stop_sensor(sensor, silent=True)
    
    # 2. 销毁传感器
    if sensors:
        valid_sensors = [s for s in sensors if s is not None]
        if valid_sensors and client is not None:
            result['sensors'] = batch_destroy_actors(client, valid_sensors, silent)
        else:
            for sensor in valid_sensors:
                if safe_destroy_actor(sensor, silent=silent):
                    result['sensors'] += 1
    
    # 3. 销毁车辆
    if vehicle is not None:
        result['vehicle'] = safe_destroy_actor(vehicle, silent=silent)
    
    # 4. 等待清理完成
    if wait_time > 0:
        time.sleep(wait_time)
    
    return result


def reset_actor_registry():
    """
    重置 actor 注册表
    
    在加载新地图或重新连接时调用。
    """
    registry = ActorRegistry.get_instance()
    registry.clear()
