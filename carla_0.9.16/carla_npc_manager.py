#!/usr/bin/env python
# coding=utf-8
"""
CARLA NPC 管理器（推理模块版本）

管理 CARLA 中的 NPC 车辆和行人，用于推理时创建真实交通环境。
基于 collect_data_new/core/npc_manager.py 简化而来。
"""

import time
import random
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


@dataclass
class NPCConfig:
    """NPC配置"""
    # 车辆配置
    num_vehicles: int = 20
    vehicles_ignore_lights: bool = False  # 推理时NPC应遵守交通规则
    vehicles_ignore_signs: bool = False
    vehicles_ignore_walkers: bool = False
    vehicle_filter: str = 'vehicle.*'
    four_wheels_only: bool = True
    use_back_spawn_points: bool = True  # 使用后半部分生成点，避免与自车冲突
    # NPC车辆行为参数
    vehicle_distance: float = 3.0  # 跟车距离（米）
    vehicle_speed_difference: float = 30.0  # 速度差异百分比
    
    # 行人配置
    num_walkers: int = 10
    walker_filter: str = 'walker.pedestrian.*'
    walker_speed_range: Tuple[float, float] = (1.0, 2.0)


class NPCManager:
    """
    CARLA NPC 管理器（推理版本）
    
    特性：
    - 统一管理 NPC 车辆和行人
    - 可配置 NPC 行为（交通规则）
    - 支持 Context Manager 自动清理
    """
    
    def __init__(self, client, world, blueprint_library=None):
        """
        初始化 NPC 管理器
        
        参数:
            client: CARLA client 对象
            world: CARLA world 对象
            blueprint_library: 蓝图库，None 则从 world 获取
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
    
    def spawn_all(self, config: NPCConfig) -> Dict[str, int]:
        """根据配置生成所有 NPC"""
        stats = {'vehicles_spawned': 0, 'walkers_spawned': 0}
        
        if config.num_vehicles > 0:
            stats['vehicles_spawned'] = self.spawn_vehicles(
                num=config.num_vehicles,
                ignore_lights=config.vehicles_ignore_lights,
                ignore_signs=config.vehicles_ignore_signs,
                ignore_walkers=config.vehicles_ignore_walkers,
                vehicle_filter=config.vehicle_filter,
                four_wheels_only=config.four_wheels_only,
                use_back_spawn_points=config.use_back_spawn_points,
                vehicle_distance=config.vehicle_distance,
                vehicle_speed_difference=config.vehicle_speed_difference
            )
        
        if config.num_walkers > 0:
            stats['walkers_spawned'] = self.spawn_walkers(
                num=config.num_walkers,
                walker_filter=config.walker_filter,
                speed_range=config.walker_speed_range
            )
        
        return stats
    
    def spawn_vehicles(self, num: int, 
                       ignore_lights: bool = False,
                       ignore_signs: bool = False,
                       ignore_walkers: bool = False,
                       vehicle_filter: str = 'vehicle.*',
                       four_wheels_only: bool = True,
                       use_back_spawn_points: bool = True,
                       vehicle_distance: float = 3.0,
                       vehicle_speed_difference: float = 30.0) -> int:
        """生成 NPC 车辆"""
        print(f"\n🚗 正在生成 {num} 辆 NPC 车辆...")
        
        blueprints = list(self.blueprint_library.filter(vehicle_filter))
        if four_wheels_only:
            blueprints = [bp for bp in blueprints 
                         if int(bp.get_attribute('number_of_wheels')) == 4]
        
        if not blueprints:
            print("❌ 没有可用的车辆蓝图")
            return 0
        
        spawn_points = self.world.get_map().get_spawn_points()
        if use_back_spawn_points:
            spawn_points = spawn_points[len(spawn_points) // 2:]
        
        random.shuffle(spawn_points)
        
        tm = self.traffic_manager
        spawned = 0
        
        for i in range(min(num, len(spawn_points))):
            bp = random.choice(blueprints)
            
            if bp.has_attribute('color'):
                colors = bp.get_attribute('color').recommended_values
                bp.set_attribute('color', random.choice(colors))
            
            vehicle = self.world.try_spawn_actor(bp, spawn_points[i])
            
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
        time.sleep(0.5)
        
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
    
    def cleanup_all(self) -> None:
        """清理所有 NPC"""
        print("🧹 正在清理 NPC...")
        
        vehicles_cleaned = self.cleanup_vehicles()
        walkers_cleaned = self.cleanup_walkers()
        
        print(f"✅ NPC 清理完成（车辆: {vehicles_cleaned}, 行人: {walkers_cleaned}）")
    
    def cleanup_vehicles(self) -> int:
        """清理所有 NPC 车辆"""
        count = 0
        for vehicle in self._vehicles:
            try:
                vehicle.destroy()
                count += 1
            except:
                pass
        self._vehicles.clear()
        return count
    
    def cleanup_walkers(self) -> int:
        """清理所有 NPC 行人和控制器"""
        for ctrl_id in self._walker_controllers:
            try:
                ctrl = self.world.get_actor(ctrl_id)
                if ctrl:
                    ctrl.stop()
                    ctrl.destroy()
            except:
                pass
        
        count = 0
        for walker in self._walkers:
            try:
                walker.destroy()
                count += 1
            except:
                pass
        
        self._walkers.clear()
        self._walker_controllers.clear()
        return count
