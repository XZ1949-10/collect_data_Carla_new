#!/usr/bin/env python
# coding=utf-8
"""
CARLA NPC 管理器

独立模块，用于管理 CARLA 中的 NPC 车辆和行人：
1. NPC 车辆生成和配置（交通规则行为）
2. NPC 行人生成和 AI 控制
3. 统一的资源清理

使用示例:
    from carla_npc_manager import NPCManager, NPCConfig
    
    # 创建管理器
    manager = NPCManager(client, world)
    
    # 配置并生成 NPC
    config = NPCConfig(
        num_vehicles=20,
        num_walkers=50,
        vehicles_ignore_lights=True
    )
    manager.spawn_all(config)
    
    # 清理
    manager.cleanup_all()

或使用 Context Manager:
    with NPCManager(client, world) as manager:
        manager.spawn_all(config)
        # NPC 活动中...
    # 自动清理
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

# CARLA 导入（延迟导入以支持类型提示）
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


@dataclass
class NPCConfig:
    """NPC 配置"""
    # 车辆配置
    num_vehicles: int = 0                       # NPC 车辆数量
    vehicles_ignore_lights: bool = True         # 是否忽略红绿灯
    vehicles_ignore_signs: bool = True          # 是否忽略停车标志
    vehicles_ignore_walkers: bool = False       # 是否忽略行人（建议 False）
    vehicle_filter: str = 'vehicle.*'           # 车辆蓝图过滤器
    four_wheels_only: bool = True               # 仅生成四轮车辆
    use_back_spawn_points: bool = True          # 使用后半部分生成点（避免占用主要路线）
    
    # 行人配置
    num_walkers: int = 0                        # NPC 行人数量
    walker_filter: str = 'walker.pedestrian.*'  # 行人蓝图过滤器
    walker_speed_range: tuple = (1.0, 2.0)      # 行人速度范围 (min, max) m/s


@dataclass
class NPCStats:
    """NPC 统计信息"""
    vehicles_spawned: int = 0
    vehicles_failed: int = 0
    walkers_spawned: int = 0
    walkers_failed: int = 0
    
    @property
    def total_spawned(self) -> int:
        return self.vehicles_spawned + self.walkers_spawned
    
    @property
    def total_failed(self) -> int:
        return self.vehicles_failed + self.walkers_failed


class NPCManager:
    """
    CARLA NPC 管理器
    
    特性：
    - 统一管理 NPC 车辆和行人
    - 可配置 NPC 行为（交通规则）
    - 支持 Context Manager 自动清理
    - 批量生成和销毁
    """
    
    def __init__(self, client, world, blueprint_library=None):
        """
        初始化 NPC 管理器
        
        参数:
            client: CARLA Client 对象
            world: CARLA World 对象
            blueprint_library: 蓝图库（可选，默认从 world 获取）
        """
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.client = client
        self.world = world
        self.blueprint_library = blueprint_library or world.get_blueprint_library()
        
        # NPC 列表
        self._vehicles: List = []
        self._walkers: List = []
        self._walker_controllers: List[int] = []  # 存储 controller actor ID
        
        # 统计
        self._stats = NPCStats()
        
        # Traffic Manager 引用
        self._traffic_manager = None
    
    # ==================== Context Manager ====================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
        return False
    
    # ==================== 属性 ====================
    
    @property
    def vehicles(self) -> List:
        """获取 NPC 车辆列表"""
        return self._vehicles.copy()
    
    @property
    def walkers(self) -> List:
        """获取 NPC 行人列表"""
        return self._walkers.copy()
    
    @property
    def num_vehicles(self) -> int:
        """当前 NPC 车辆数量"""
        return len(self._vehicles)
    
    @property
    def num_walkers(self) -> int:
        """当前 NPC 行人数量"""
        return len(self._walkers)
    
    @property
    def stats(self) -> NPCStats:
        """获取统计信息"""
        return self._stats
    
    @property
    def traffic_manager(self):
        """获取 Traffic Manager（延迟初始化）"""
        if self._traffic_manager is None:
            self._traffic_manager = self.client.get_trafficmanager()
        return self._traffic_manager
    
    # ==================== 生成方法 ====================
    
    def spawn_all(self, config: NPCConfig) -> NPCStats:
        """
        根据配置生成所有 NPC
        
        参数:
            config: NPC 配置
            
        返回:
            NPCStats: 生成统计
        """
        self._stats = NPCStats()
        
        if config.num_vehicles > 0:
            self.spawn_vehicles(
                num=config.num_vehicles,
                ignore_lights=config.vehicles_ignore_lights,
                ignore_signs=config.vehicles_ignore_signs,
                ignore_walkers=config.vehicles_ignore_walkers,
                vehicle_filter=config.vehicle_filter,
                four_wheels_only=config.four_wheels_only,
                use_back_spawn_points=config.use_back_spawn_points
            )
        
        if config.num_walkers > 0:
            self.spawn_walkers(
                num=config.num_walkers,
                walker_filter=config.walker_filter,
                speed_range=config.walker_speed_range
            )
        
        return self._stats
    
    def spawn_vehicles(self, num: int, 
                       ignore_lights: bool = True,
                       ignore_signs: bool = True,
                       ignore_walkers: bool = False,
                       vehicle_filter: str = 'vehicle.*',
                       four_wheels_only: bool = True,
                       use_back_spawn_points: bool = True) -> int:
        """
        生成 NPC 车辆
        
        参数:
            num: 生成数量
            ignore_lights: 是否忽略红绿灯
            ignore_signs: 是否忽略停车标志
            ignore_walkers: 是否忽略行人
            vehicle_filter: 车辆蓝图过滤器
            four_wheels_only: 仅生成四轮车辆
            use_back_spawn_points: 使用后半部分生成点
            
        返回:
            int: 成功生成的数量
        """
        print(f"\n🚗 正在生成 {num} 辆 NPC 车辆...")
        
        # 获取车辆蓝图
        blueprints = list(self.blueprint_library.filter(vehicle_filter))
        if four_wheels_only:
            blueprints = [bp for bp in blueprints 
                         if int(bp.get_attribute('number_of_wheels')) == 4]
        
        if not blueprints:
            print("❌ 没有可用的车辆蓝图")
            return 0
        
        # 获取生成点
        spawn_points = self.world.get_map().get_spawn_points()
        if use_back_spawn_points:
            half_idx = len(spawn_points) // 2
            spawn_points = spawn_points[half_idx:]
        
        random.shuffle(spawn_points)
        
        # 获取 Traffic Manager
        tm = self.traffic_manager
        
        # 生成车辆
        spawned = 0
        failed = 0
        
        for i in range(min(num, len(spawn_points))):
            bp = random.choice(blueprints)
            
            # 随机颜色
            if bp.has_attribute('color'):
                colors = bp.get_attribute('color').recommended_values
                bp.set_attribute('color', random.choice(colors))
            
            # 生成车辆
            vehicle = self.world.try_spawn_actor(bp, spawn_points[i])
            
            if vehicle:
                # 启用自动驾驶
                vehicle.set_autopilot(True, tm.get_port())
                
                # 配置交通规则行为
                if ignore_lights:
                    tm.ignore_lights_percentage(vehicle, 100)
                if ignore_signs:
                    tm.ignore_signs_percentage(vehicle, 100)
                if ignore_walkers:
                    tm.ignore_walkers_percentage(vehicle, 100)
                
                self._vehicles.append(vehicle)
                spawned += 1
            else:
                failed += 1
        
        self._stats.vehicles_spawned = spawned
        self._stats.vehicles_failed = failed
        
        # 打印行为配置
        behavior = []
        if ignore_lights:
            behavior.append("忽略红绿灯")
        if ignore_signs:
            behavior.append("忽略停车标志")
        if ignore_walkers:
            behavior.append("忽略行人")
        behavior_str = ", ".join(behavior) if behavior else "遵守所有规则"
        
        print(f"✅ 成功生成 {spawned} 辆 NPC 车辆（{behavior_str}）")
        if failed > 0:
            print(f"⚠️  {failed} 辆车辆生成失败（生成点被占用）")
        
        return spawned
    
    def spawn_walkers(self, num: int,
                      walker_filter: str = 'walker.pedestrian.*',
                      speed_range: tuple = (1.0, 2.0)) -> int:
        """
        生成 NPC 行人
        
        参数:
            num: 生成数量
            walker_filter: 行人蓝图过滤器
            speed_range: 行人速度范围 (min, max) m/s
            
        返回:
            int: 成功生成的数量
        """
        print(f"\n🚶 正在生成 {num} 个 NPC 行人...")
        
        # 获取行人蓝图
        walker_bps = list(self.blueprint_library.filter(walker_filter))
        if not walker_bps:
            print("❌ 没有可用的行人蓝图")
            return 0
        
        # 获取随机生成点
        spawn_points = []
        for _ in range(num):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(location=loc))
        
        if not spawn_points:
            print("❌ 无法获取行人生成点")
            return 0
        
        # 批量生成行人
        batch = [
            carla.command.SpawnActor(random.choice(walker_bps), sp) 
            for sp in spawn_points
        ]
        results = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in results if not r.error]
        
        # 生成 AI 控制器
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), wid) 
            for wid in walker_ids
        ]
        results = self.client.apply_batch_sync(batch, True)
        controller_ids = [r.actor_id for r in results if not r.error]
        
        # 等待一帧让 actor 生效
        self.world.tick()
        
        # 启动控制器
        min_speed, max_speed = speed_range
        for ctrl_id in controller_ids:
            ctrl = self.world.get_actor(ctrl_id)
            if ctrl:
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
                ctrl.set_max_speed(min_speed + random.random() * (max_speed - min_speed))
        
        # 保存引用
        self._walkers = list(self.world.get_actors(walker_ids))
        self._walker_controllers = controller_ids
        
        spawned = len(self._walkers)
        failed = num - spawned
        
        self._stats.walkers_spawned = spawned
        self._stats.walkers_failed = failed
        
        print(f"✅ 成功生成 {spawned} 个 NPC 行人")
        if failed > 0:
            print(f"⚠️  {failed} 个行人生成失败")
        
        return spawned
    
    # ==================== 清理方法 ====================
    
    def cleanup_all(self) -> None:
        """清理所有 NPC"""
        print("🧹 正在清理 NPC...")
        
        vehicles_cleaned = self.cleanup_vehicles()
        walkers_cleaned = self.cleanup_walkers()
        
        print(f"✅ NPC 清理完成（车辆: {vehicles_cleaned}, 行人: {walkers_cleaned}）")
    
    def cleanup_vehicles(self) -> int:
        """
        清理所有 NPC 车辆
        
        返回:
            int: 清理的数量
        """
        count = 0
        for vehicle in self._vehicles:
            try:
                vehicle.destroy()
                count += 1
            except Exception:
                pass
        
        self._vehicles.clear()
        return count
    
    def cleanup_walkers(self) -> int:
        """
        清理所有 NPC 行人和控制器
        
        返回:
            int: 清理的数量
        """
        # 先停止并销毁控制器
        for ctrl_id in self._walker_controllers:
            try:
                ctrl = self.world.get_actor(ctrl_id)
                if ctrl:
                    ctrl.stop()
                    ctrl.destroy()
            except Exception:
                pass
        
        # 销毁行人
        count = 0
        for walker in self._walkers:
            try:
                walker.destroy()
                count += 1
            except Exception:
                pass
        
        self._walkers.clear()
        self._walker_controllers.clear()
        return count
    
    # ==================== 工具方法 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'num_vehicles': self.num_vehicles,
            'num_walkers': self.num_walkers,
            'num_controllers': len(self._walker_controllers),
            'stats': {
                'vehicles_spawned': self._stats.vehicles_spawned,
                'vehicles_failed': self._stats.vehicles_failed,
                'walkers_spawned': self._stats.walkers_spawned,
                'walkers_failed': self._stats.walkers_failed,
            }
        }


# ==================== 便捷函数 ====================

@contextmanager
def npc_context(client, world, config: NPCConfig):
    """
    NPC 管理上下文
    
    使用示例:
        config = NPCConfig(num_vehicles=20, num_walkers=50)
        with npc_context(client, world, config) as manager:
            # NPC 活动中
            pass
        # 自动清理
    """
    manager = NPCManager(client, world)
    try:
        manager.spawn_all(config)
        yield manager
    finally:
        manager.cleanup_all()


def create_manager_from_config(client, world, config_dict: Dict[str, Any]) -> NPCManager:
    """
    从配置字典创建 NPC 管理器并生成 NPC
    
    参数:
        client: CARLA Client
        world: CARLA World
        config_dict: 配置字典，支持以下键：
            - spawn_npc_vehicles: bool
            - num_npc_vehicles: int
            - spawn_npc_walkers: bool
            - num_npc_walkers: int
            - npc_behavior.ignore_traffic_lights: bool
            - npc_behavior.ignore_signs: bool
            - npc_behavior.ignore_walkers: bool
    
    返回:
        NPCManager: 已生成 NPC 的管理器
    """
    manager = NPCManager(client, world)
    
    # 解析配置
    spawn_vehicles = config_dict.get('spawn_npc_vehicles', False)
    num_vehicles = config_dict.get('num_npc_vehicles', 0)
    spawn_walkers = config_dict.get('spawn_npc_walkers', False)
    num_walkers = config_dict.get('num_npc_walkers', 0)
    
    npc_behavior = config_dict.get('npc_behavior', {})
    ignore_lights = npc_behavior.get('ignore_traffic_lights', True)
    ignore_signs = npc_behavior.get('ignore_signs', True)
    ignore_walkers = npc_behavior.get('ignore_walkers', False)
    
    # 创建配置
    config = NPCConfig(
        num_vehicles=num_vehicles if spawn_vehicles else 0,
        num_walkers=num_walkers if spawn_walkers else 0,
        vehicles_ignore_lights=ignore_lights,
        vehicles_ignore_signs=ignore_signs,
        vehicles_ignore_walkers=ignore_walkers
    )
    
    # 生成 NPC
    manager.spawn_all(config)
    
    return manager


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("="*60)
    print("CARLA NPC 管理器测试")
    print("="*60)
    print("\n此模块需要连接到 CARLA 服务器才能测试。")
    print("请确保 CARLA 服务器正在运行。")
    
    # 测试配置类
    print("\n--- 测试配置类 ---")
    config = NPCConfig(
        num_vehicles=10,
        num_walkers=20,
        vehicles_ignore_lights=True,
        vehicles_ignore_signs=False
    )
    print(f"配置: 车辆={config.num_vehicles}, 行人={config.num_walkers}")
    print(f"车辆行为: 忽略红绿灯={config.vehicles_ignore_lights}, 忽略停车标志={config.vehicles_ignore_signs}")
    
    # 测试统计类
    print("\n--- 测试统计类 ---")
    stats = NPCStats(vehicles_spawned=8, vehicles_failed=2, walkers_spawned=18, walkers_failed=2)
    print(f"统计: 总生成={stats.total_spawned}, 总失败={stats.total_failed}")
    
    print("\n✅ 配置类测试完成")
    print("\n要进行完整测试，请运行:")
    print("  python carla_npc_manager.py --test")
