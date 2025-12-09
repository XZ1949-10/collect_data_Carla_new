#!/usr/bin/env python
# coding=utf-8
'''
车辆生成模块
负责在Carla世界中生成车辆
'''

import time
import random
from carla_config import (
    MAX_SPAWN_ATTEMPTS, 
    SPAWN_STABILIZE_TICKS, 
    SPAWN_STABILIZE_DELAY
)


class VehicleSpawner:
    """车辆生成器"""
    
    def __init__(self, world):
        """
        初始化车辆生成器
        
        参数:
            world: carla.World 对象
        """
        self.world = world
        self.spawn_points = world.get_map().get_spawn_points()
    
    def spawn(self, vehicle_filter='vehicle.tesla.model3', 
              spawn_index=None, max_attempts=MAX_SPAWN_ATTEMPTS):
        """
        生成车辆
        
        参数:
            vehicle_filter (str): 车辆类型
            spawn_index (int): 起点索引，None表示随机
            max_attempts (int): 最大尝试次数
            
        返回:
            carla.Vehicle: 生成的车辆对象
        """
        print(f"正在生成车辆 ({vehicle_filter})...")
        
        if len(self.spawn_points) == 0:
            raise RuntimeError("地图上没有可用的生成点！")
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_filter)[0]
        
        vehicle = None
        
        # 选择生成点
        if spawn_index is not None and 0 <= spawn_index < len(self.spawn_points):
            # 使用指定的生成点
            spawn_point = self.spawn_points[spawn_index]
            print(f"使用指定起点索引: {spawn_index}")
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if vehicle is not None:
                print(f"车辆生成成功！位置: {spawn_point.location}")
            else:
                raise RuntimeError(f"无法在指定位置生成车辆（索引 {spawn_index}）")
        else:
            # 随机选择生成点并尝试多次
            print("随机选择生成点...")
            for attempt in range(max_attempts):
                spawn_point = random.choice(self.spawn_points)
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle is not None:
                    actual_index = self.spawn_points.index(spawn_point)
                    print(f"车辆生成成功！位置: {spawn_point.location} "
                          f"(索引: {actual_index}, 尝试 {attempt + 1}/{max_attempts})")
                    break
            
            if vehicle is None:
                raise RuntimeError(f"无法生成车辆！尝试了 {max_attempts} 次")
        
        # 关闭自动驾驶
        vehicle.set_autopilot(False)
        
        # 等待物理引擎同步（增加超时处理）
        try:
            self.world.tick()
            for _ in range(SPAWN_STABILIZE_TICKS):
                self.world.tick()
                time.sleep(SPAWN_STABILIZE_DELAY)
        except RuntimeError as e:
            print(f"⚠️  警告: 物理引擎同步超时 ({e})")
            print("提示: 请检查 Carla 服务器是否正常运行")
        
        # 验证位置
        actual_location = vehicle.get_location()
        print(f"车辆实际位置: ({actual_location.x:.1f}, {actual_location.y:.1f}, {actual_location.z:.1f})")
        
        return vehicle
