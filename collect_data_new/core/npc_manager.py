#!/usr/bin/env python
# coding=utf-8
"""
CARLA NPC 管理器

管理 CARLA 中的 NPC 车辆和行人。
"""

import time
import random
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from contextlib import contextmanager

from ..config import NPCConfig
from .actor_utils import (
    is_actor_alive, 
    safe_destroy_actor, 
    batch_destroy_actors,
    ActorRegistry
)

if TYPE_CHECKING:
    from .sync_mode_manager import SyncModeManager

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class NPCManager:
    """
    CARLA NPC 管理器
    
    特性：
    - 统一管理 NPC 车辆和行人
    - 可配置 NPC 行为（交通规则）
    - 支持 Context Manager 自动清理
    - 支持 SyncModeManager 统一管理 tick
    """
    
    def __init__(self, client, world, blueprint_library=None, 
                 sync_manager: 'SyncModeManager' = None):
        """
        初始化 NPC 管理器
        
        参数:
            client: CARLA client 对象
            world: CARLA world 对象
            blueprint_library: 蓝图库，None 则从 world 获取
            sync_manager: 同步模式管理器，用于安全的 tick 调用
        """
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.client = client
        self.world = world
        self.blueprint_library = blueprint_library or world.get_blueprint_library()
        
        self._vehicles: List = []
        self._walkers: List = []
        self._walker_controllers: List[int] = []
        
        self._traffic_manager = None
        self._sync_manager: Optional['SyncModeManager'] = sync_manager
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
        return False
    
    @property
    def vehicles(self) -> List:
        return self._vehicles.copy()
    
    @property
    def walkers(self) -> List:
        return self._walkers.copy()
    
    @property
    def num_vehicles(self) -> int:
        return len(self._vehicles)
    
    @property
    def num_walkers(self) -> int:
        return len(self._walkers)
    
    @property
    def traffic_manager(self):
        if self._traffic_manager is None:
            self._traffic_manager = self.client.get_trafficmanager()
        return self._traffic_manager
    
    def spawn_all(self, config: NPCConfig, excluded_spawn_indices: List[int] = None) -> Dict[str, int]:
        """根据配置生成所有 NPC
        
        参数:
            config: NPC 配置
            excluded_spawn_indices: 需要排除的生成点索引列表（避免与数据收集车辆冲突）
        """
        stats = {'vehicles_spawned': 0, 'walkers_spawned': 0}
        
        if config.num_vehicles > 0:
            # 使用 get_effective_* 方法获取实际配置（考虑总开关）
            stats['vehicles_spawned'] = self.spawn_vehicles(
                num=config.num_vehicles,
                ignore_lights=config.get_effective_ignore_lights(),
                ignore_signs=config.get_effective_ignore_signs(),
                ignore_walkers=config.get_effective_ignore_walkers(),
                vehicle_filter=config.vehicle_filter,
                four_wheels_only=config.four_wheels_only,
                use_back_spawn_points=config.use_back_spawn_points,
                vehicle_distance=config.vehicle_distance,
                vehicle_speed_difference=config.vehicle_speed_difference,
                excluded_spawn_indices=excluded_spawn_indices
            )
        
        if config.num_walkers > 0:
            stats['walkers_spawned'] = self.spawn_walkers(
                num=config.num_walkers,
                walker_filter=config.walker_filter,
                speed_range=config.walker_speed_range
            )
        
        return stats
    
    def spawn_vehicles(self, num: int, 
                       ignore_lights: bool = True,
                       ignore_signs: bool = True,
                       ignore_walkers: bool = False,
                       vehicle_filter: str = 'vehicle.*',
                       four_wheels_only: bool = True,
                       use_back_spawn_points: bool = True,
                       vehicle_distance: float = 3.0,
                       vehicle_speed_difference: float = 30.0,
                       excluded_spawn_indices: List[int] = None) -> int:
        """生成 NPC 车辆
        
        参数:
            num: 要生成的车辆数量
            ignore_lights: 是否忽略红绿灯
            ignore_signs: 是否忽略交通标志
            ignore_walkers: 是否忽略行人
            vehicle_filter: 车辆蓝图过滤器
            four_wheels_only: 是否只使用四轮车辆
            use_back_spawn_points: 是否使用后半部分生成点
            vehicle_distance: 跟车距离
            vehicle_speed_difference: 速度差异百分比
            excluded_spawn_indices: 需要排除的生成点索引列表
        """
        print(f"\n🚗 正在生成 {num} 辆 NPC 车辆...")
        
        blueprints = list(self.blueprint_library.filter(vehicle_filter))
        if four_wheels_only:
            blueprints = [bp for bp in blueprints 
                         if int(bp.get_attribute('number_of_wheels')) == 4]
        
        if not blueprints:
            print("❌ 没有可用的车辆蓝图")
            return 0
        
        all_spawn_points = self.world.get_map().get_spawn_points()
        
        # 过滤掉需要排除的生成点（数据收集车辆使用的生成点）
        if excluded_spawn_indices:
            excluded_set = set(excluded_spawn_indices)
            spawn_points = [(i, sp) for i, sp in enumerate(all_spawn_points) 
                           if i not in excluded_set]
            if len(spawn_points) < len(all_spawn_points):
                print(f"  📍 已排除 {len(excluded_set)} 个数据收集生成点")
        else:
            spawn_points = list(enumerate(all_spawn_points))
        
        # 使用后半部分生成点
        if use_back_spawn_points:
            spawn_points = spawn_points[len(spawn_points) // 2:]
        
        random.shuffle(spawn_points)
        
        tm = self.traffic_manager
        spawned = 0
        
        for idx, sp in spawn_points[:num]:
            bp = random.choice(blueprints)
            
            if bp.has_attribute('color'):
                colors = bp.get_attribute('color').recommended_values
                bp.set_attribute('color', random.choice(colors))
            
            vehicle = self.world.try_spawn_actor(bp, sp)
            
            if vehicle:
                vehicle.set_autopilot(True, tm.get_port())
                
                # 交通规则设置
                if ignore_lights:
                    tm.ignore_lights_percentage(vehicle, 100)
                if ignore_signs:
                    tm.ignore_signs_percentage(vehicle, 100)
                if ignore_walkers:
                    tm.ignore_walkers_percentage(vehicle, 100)
                
                # 行为参数设置
                tm.distance_to_leading_vehicle(vehicle, vehicle_distance)
                tm.vehicle_percentage_speed_difference(vehicle, vehicle_speed_difference)
                
                self._vehicles.append(vehicle)
                spawned += 1
        
        print(f"✅ 成功生成 {spawned} 辆 NPC 车辆")
        return spawned
    
    def spawn_walkers(self, num: int,
                      walker_filter: str = 'walker.pedestrian.*',
                      speed_range: tuple = (1.0, 2.0)) -> int:
        """生成 NPC 行人"""
        print(f"\n🚶 正在生成 {num} 个 NPC 行人...")
        
        walker_bps = list(self.blueprint_library.filter(walker_filter))
        if not walker_bps:
            print("❌ 没有可用的行人蓝图")
            return 0
        
        spawn_points = []
        for _ in range(num):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(location=loc))
        
        if not spawn_points:
            print("❌ 无法获取行人生成点")
            return 0
        
        batch = [
            carla.command.SpawnActor(random.choice(walker_bps), sp) 
            for sp in spawn_points
        ]
        results = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in results if not r.error]
        
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), wid) 
            for wid in walker_ids
        ]
        results = self.client.apply_batch_sync(batch, True)
        controller_ids = [r.actor_id for r in results if not r.error]
        
        # 等待行人控制器初始化
        # 使用 SyncModeManager 进行安全的 tick，或在异步模式下等待
        self._wait_for_initialization()
        
        min_speed, max_speed = speed_range
        for ctrl_id in controller_ids:
            ctrl = self.world.get_actor(ctrl_id)
            if ctrl:
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
                ctrl.set_max_speed(min_speed + random.random() * (max_speed - min_speed))
        
        self._walkers = list(self.world.get_actors(walker_ids))
        self._walker_controllers = controller_ids
        
        print(f"✅ 成功生成 {len(self._walkers)} 个 NPC 行人")
        return len(self._walkers)
    
    def _wait_for_initialization(self, wait_time: float = 0.5, tick_count: int = 5):
        """
        等待初始化完成
        
        如果有 SyncModeManager，使用 safe_tick() 推进模拟多次；
        否则使用 time.sleep() 等待（适用于异步模式）。
        
        注意：不直接调用 world.tick()，避免与 SyncModeManager 职责重叠。
        
        参数:
            wait_time: 异步模式下的等待时间（秒）
            tick_count: 同步模式下执行的 tick 次数
        """
        if self._sync_manager is not None:
            # 使用 SyncModeManager 安全地推进模拟多次，确保初始化完成
            success_count = 0
            for _ in range(tick_count):
                if self._sync_manager.safe_tick():
                    success_count += 1
            if success_count < tick_count // 2:
                print(f"  ⚠️ NPC 初始化 tick 不完整: {success_count}/{tick_count}")
        else:
            # 异步模式下等待一段时间让初始化完成
            # 不调用 world.tick()，因为：
            # 1. 异步模式下 tick() 无效
            # 2. 同步模式下应该由 SyncModeManager 统一管理
            time.sleep(wait_time)
    
    def cleanup_all(self) -> None:
        """清理所有 NPC
        
        注意：必须在异步模式下清理 NPC，否则可能导致死锁或崩溃。
        """
        print("🧹 正在清理 NPC...")
        
        # 确保在异步模式下清理（在同步模式下销毁 actor 可能导致崩溃）
        if self._sync_manager is not None:
            try:
                print("  🔄 切换到异步模式...")
                self._sync_manager.ensure_async_mode(wait=True)
                print("  ✅ 已切换到异步模式")
            except Exception as e:
                print(f"⚠️ 切换异步模式失败: {e}")
        
        # 等待一下确保模式切换生效
        time.sleep(0.5)
        
        print(f"  🚗 开始清理 {len(self._vehicles)} 辆 NPC 车辆...")
        vehicles_cleaned = self.cleanup_vehicles()
        print(f"  ✅ 车辆清理完成: {vehicles_cleaned}")
        
        print(f"  🚶 开始清理 {len(self._walkers)} 个 NPC 行人...")
        walkers_cleaned = self.cleanup_walkers()
        print(f"  ✅ 行人清理完成: {walkers_cleaned}")
        
        print(f"✅ NPC 清理完成（车辆: {vehicles_cleaned}, 行人: {walkers_cleaned}）")
    
    def cleanup_vehicles(self) -> int:
        """清理所有 NPC 车辆
        
        使用统一的 actor_utils 进行安全销毁，避免 "not found" 错误。
        """
        count = len(self._vehicles)
        if count == 0:
            return 0
        
        # 使用统一的批量销毁工具
        destroyed = batch_destroy_actors(self.client, self._vehicles, silent=True)
        
        if destroyed < count:
            print(f"    ℹ️ NPC 车辆: {destroyed}/{count} 辆已销毁（部分可能已不存在）")
        
        self._vehicles.clear()
        return destroyed
    
    def cleanup_walkers(self) -> int:
        """清理所有 NPC 行人和控制器
        
        使用统一的 actor_utils 进行安全销毁，避免 "not found" 错误。
        """
        # 先停止所有控制器
        registry = ActorRegistry.get_instance()
        for ctrl_id in self._walker_controllers:
            if registry.is_destroyed(ctrl_id):
                continue
            try:
                ctrl = self.world.get_actor(ctrl_id)
                if ctrl and is_actor_alive(ctrl):
                    ctrl.stop()
            except:
                pass
        
        count = len(self._walkers)
        
        # 收集所有需要销毁的 actors
        actors_to_destroy = []
        
        # 控制器
        for ctrl_id in self._walker_controllers:
            if registry.is_destroyed(ctrl_id):
                continue
            try:
                ctrl = self.world.get_actor(ctrl_id)
                if ctrl and is_actor_alive(ctrl):
                    actors_to_destroy.append(ctrl)
                else:
                    registry.mark_destroyed(ctrl_id)
            except:
                registry.mark_destroyed(ctrl_id)
        
        # 行人
        for walker in self._walkers:
            if walker is not None and is_actor_alive(walker):
                actors_to_destroy.append(walker)
        
        # 批量销毁
        destroyed = batch_destroy_actors(self.client, actors_to_destroy, silent=True)
        
        if destroyed < len(actors_to_destroy):
            print(f"    ℹ️ NPC 行人: {destroyed}/{len(actors_to_destroy)} 个已销毁（部分可能已不存在）")
        
        self._walkers.clear()
        self._walker_controllers.clear()
        return count


@contextmanager
def npc_context(client, world, config: NPCConfig, 
                sync_manager: 'SyncModeManager' = None):
    """
    NPC 管理上下文
    
    参数:
        client: CARLA client 对象
        world: CARLA world 对象
        config: NPC 配置
        sync_manager: 同步模式管理器（推荐传入）
    
    使用示例:
        with npc_context(client, world, npc_config, sync_manager) as manager:
            # NPC 已生成
            ...
        # 自动清理
    """
    manager = NPCManager(client, world, sync_manager=sync_manager)
    try:
        manager.spawn_all(config)
        yield manager
    finally:
        manager.cleanup_all()
