#!/usr/bin/env python
# coding=utf-8
"""
CARLA 数据收集模块

模块化重构版本，提供清晰的代码结构和低耦合设计。

主要模块:
- config: 配置管理
- core: 核心功能（基础收集器、资源管理、NPC管理）
- detection: 检测功能（异常检测、碰撞处理）
- noise: 噪声注入
- collectors: 各类收集器实现
- utils: 工具函数
"""

__version__ = '2.0.0'
__author__ = 'AI Assistant'

# 导出主要类
from .config import CollectorConfig, NoiseConfig, AnomalyConfig
from .core import BaseDataCollector, CarlaResourceManager, NPCManager
from .detection import AnomalyDetector, AnomalyType
from .noise import Noiser

__all__ = [
    'CollectorConfig',
    'NoiseConfig', 
    'AnomalyConfig',
    'BaseDataCollector',
    'CarlaResourceManager',
    'NPCManager',
    'AnomalyDetector',
    'AnomalyType',
    'Noiser',
]
