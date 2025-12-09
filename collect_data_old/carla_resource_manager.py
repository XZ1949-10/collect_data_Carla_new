#!/usr/bin/env python
# coding=utf-8
"""
CARLA 资源管理器 (兼容层)

此文件已被 carla_resource_manager_v2.py 替代。
保留此文件用于向后兼容，所有导入将重定向到 V2 版本。

新代码请直接使用:
    from carla_resource_manager_v2 import CarlaResourceManagerV2

迁移说明：
-----------
旧代码：
    from carla_resource_manager import CarlaResourceManager
    mgr = CarlaResourceManager(world, bp_lib)
    try:
        mgr.create_all(...)
    finally:
        mgr.destroy_all()

新代码（推荐）：
    from carla_resource_manager_v2 import CarlaResourceManagerV2
    with CarlaResourceManagerV2(world, bp_lib) as mgr:
        mgr.create_all(...)
    # 自动清理

或者使用便捷函数：
    from carla_resource_manager_v2 import carla_resources
    with carla_resources(world, bp_lib, transform, cam_cb, col_cb) as mgr:
        # 使用资源
    # 自动清理
"""

# 从 V2 导入所有内容
from carla_resource_manager_v2 import (
    CarlaResourceManagerV2 as CarlaResourceManager,
    CarlaResourceManagerV2,
    ResourceState,
    carla_resources
)

# 向后兼容：保留旧类名
__all__ = ['CarlaResourceManager', 'CarlaResourceManagerV2', 'ResourceState', 'carla_resources']
