#!/usr/bin/env python
# coding=utf-8
"""
Agent 工厂模块

提供 BasicAgent 的统一创建和配置逻辑，避免代码重复。
"""

from typing import Optional, Dict, Any

try:
    from agents.navigation.basic_agent import BasicAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


def create_basic_agent(
    vehicle,
    world_map,
    destination,
    start_location=None,
    target_speed: float = 10.0,
    simulation_fps: int = 20,
    ignore_traffic_lights: bool = True,
    ignore_signs: bool = True,
    ignore_vehicles_percentage: int = 80,
    custom_opt_dict: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    创建并配置 BasicAgent
    
    参数:
        vehicle: CARLA 车辆对象
        world_map: CARLA 地图对象
        destination: 目的地位置 (carla.Location)
        start_location: 起始位置 (carla.Location)，可选
        target_speed: 目标速度 (km/h)
        simulation_fps: 模拟帧率
        ignore_traffic_lights: 是否忽略红绿灯
        ignore_signs: 是否忽略停车标志
        ignore_vehicles_percentage: 忽略其他车辆的百分比 (0-100)
        custom_opt_dict: 自定义配置字典，会覆盖默认配置
        
    返回:
        BasicAgent 实例，如果 agents 模块不可用则返回 None
    """
    if not AGENTS_AVAILABLE:
        print("⚠️ agents 模块不可用，无法创建 BasicAgent")
        return None
    
    if vehicle is None:
        print("⚠️ 车辆对象为空，无法创建 BasicAgent")
        return None
    
    # 根据百分比决定是否忽略车辆
    # BasicAgent 只支持布尔值，>50 视为忽略
    ignore_vehicles = ignore_vehicles_percentage > 50
    
    # 默认配置
    opt_dict = {
        'target_speed': target_speed,
        'ignore_traffic_lights': ignore_traffic_lights,
        'ignore_stop_signs': ignore_signs,
        'ignore_vehicles': ignore_vehicles,
        'sampling_resolution': 1.0,
        'base_tlight_threshold': 5.0,
        'lateral_control_dict': {
            'K_P': 1.5,
            'K_I': 0.0,
            'K_D': 0.05,
            'dt': 1.0 / simulation_fps
        },
        'longitudinal_control_dict': {
            'K_P': 1.0,
            'K_I': 0.05,
            'K_D': 0.0,
            'dt': 1.0 / simulation_fps
        },
        'max_steering': 0.8,
        'max_throttle': 0.75,
        'max_brake': 0.5,
        'base_min_distance': 2.0,
        'distance_ratio': 0.3
    }
    
    # 合并自定义配置
    if custom_opt_dict:
        opt_dict.update(custom_opt_dict)
    
    try:
        agent = BasicAgent(
            vehicle,
            target_speed=target_speed,
            opt_dict=opt_dict,
            map_inst=world_map
        )
        
        # 设置目的地
        agent.set_destination(destination, start_location=start_location)
        
        print(f"✅ BasicAgent 已配置 (忽略车辆: {'是' if ignore_vehicles else '否'})")
        return agent
        
    except Exception as e:
        print(f"❌ 创建 BasicAgent 失败: {e}")
        return None


def is_agents_available() -> bool:
    """检查 agents 模块是否可用"""
    return AGENTS_AVAILABLE
