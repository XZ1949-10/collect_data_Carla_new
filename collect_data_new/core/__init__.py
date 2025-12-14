#!/usr/bin/env python
# coding=utf-8
"""
核心模块

资源管理说明：
    推荐使用 ResourceLifecycleHelper（配合 SyncModeManager）管理 CARLA 资源。
    CarlaResourceManager 已废弃，保留仅为向后兼容。
    
    示例：
        from .core import SyncModeManager, ResourceLifecycleHelper
        
        sync_mgr = SyncModeManager(world)
        helper = ResourceLifecycleHelper(sync_mgr)
        
        vehicle = helper.spawn_vehicle_safe(bp, transform)
        camera = helper.create_sensor_safe(bp, transform, vehicle, callback)
        helper.destroy_all_safe([camera], vehicle)
"""

from .base_collector import BaseDataCollector
# CarlaResourceManager 已废弃，建议使用 ResourceLifecycleHelper
from .resource_manager import CarlaResourceManager, ResourceState
from .npc_manager import NPCManager
from .route_planner import RoutePlanner
from .collision_recovery import CollisionRecoveryManager, RecoveryConfig, adjust_spawn_transform
from .agent_factory import create_basic_agent, is_agents_available
from .weather_manager import (
    WeatherManager,
    WeatherSetupConfig as WeatherManagerConfig,  # 使用新名称，避免与 config/settings.py 中的 WeatherConfig 混淆
    CustomWeatherParams,
    get_weather_list,
    is_valid_weather_preset,
    WEATHER_PRESETS_MAP,
    WEATHER_COLLECTION_PRESETS,
)
from .sync_mode_manager import (
    SyncModeManager,
    SyncModeConfig,
    SyncMode,
    ResourceLifecycleHelper,
    CollectorLifecycleManager,
)
from .traffic_light_manager import (
    TrafficLightManager,
    TrafficLightTiming,
    TrafficLightState,
    TrafficLightInfo,
    TRAFFIC_LIGHT_PRESETS,
    get_traffic_light_presets,
    create_traffic_light_manager,
    configure_traffic_lights,
)
from .traffic_light_route_planner import (
    TrafficLightRoutePlanner,
    TrafficLightRouteConfig as TrafficLightRoutePlannerConfig,  # 重命名避免与 config/settings.py 冲突
    create_traffic_light_route_planner,
)
# Actor 工具（统一的资源销毁）
from .actor_utils import (
    ActorRegistry,
    is_actor_alive,
    safe_destroy_actor,
    safe_destroy_sensor,
    batch_destroy_actors,
    destroy_all_resources,
    reset_actor_registry,
)

__all__ = [
    'BaseDataCollector',
    'CarlaResourceManager',
    'ResourceState',
    'NPCManager',
    'RoutePlanner',
    'CollisionRecoveryManager',
    'RecoveryConfig',
    'adjust_spawn_transform',
    'create_basic_agent',
    'is_agents_available',
    # 天气管理
    'WeatherManager',
    'WeatherManagerConfig',
    'CustomWeatherParams',
    'get_weather_list',
    'is_valid_weather_preset',
    'WEATHER_PRESETS_MAP',
    'WEATHER_COLLECTION_PRESETS',
    # 同步模式管理
    'SyncModeManager',
    'SyncModeConfig',
    'SyncMode',
    'ResourceLifecycleHelper',
    'CollectorLifecycleManager',
    # 红绿灯管理
    'TrafficLightManager',
    'TrafficLightTiming',
    'TrafficLightState',
    'TrafficLightInfo',
    'TRAFFIC_LIGHT_PRESETS',
    'get_traffic_light_presets',
    'create_traffic_light_manager',
    'configure_traffic_lights',
    # 红绿灯路线规划
    'TrafficLightRoutePlanner',
    'TrafficLightRoutePlannerConfig',  # 重命名避免与 config/settings.py 冲突
    'create_traffic_light_route_planner',
    # Actor 工具
    'ActorRegistry',
    'is_actor_alive',
    'safe_destroy_actor',
    'safe_destroy_sensor',
    'batch_destroy_actors',
    'destroy_all_resources',
    'reset_actor_registry',
]
