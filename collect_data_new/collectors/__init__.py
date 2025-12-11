#!/usr/bin/env python
# coding=utf-8
"""收集器实现模块"""

from .command_based import CommandBasedCollector
from .interactive import InteractiveCollector
from .auto_collector import (
    AutoFullTownCollector,
    MultiWeatherCollector,
    run_single_weather_collection,
    run_multi_weather_collection,
)

# 从 core.weather_manager 导入天气相关函数
from ..core.weather_manager import get_weather_list, WEATHER_COLLECTION_PRESETS

__all__ = [
    'CommandBasedCollector',
    'InteractiveCollector',
    'AutoFullTownCollector',
    'MultiWeatherCollector',
    'run_single_weather_collection',
    'run_multi_weather_collection',
    'get_weather_list',
    'WEATHER_COLLECTION_PRESETS',
]
