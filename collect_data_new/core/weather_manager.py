#!/usr/bin/env python
# coding=utf-8
"""
天气管理模块

负责 CARLA 天气设置，支持预设天气和自定义天气参数。
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


# ==================== 天气预设定义 ====================

# 所有可用的天气预设（21种 + DustStorm）
WEATHER_PRESETS_MAP = {
    # 正午天气 (7种)
    'ClearNoon': 'ClearNoon',
    'CloudyNoon': 'CloudyNoon',
    'WetNoon': 'WetNoon',
    'WetCloudyNoon': 'WetCloudyNoon',
    'SoftRainNoon': 'SoftRainNoon',
    'MidRainyNoon': 'MidRainyNoon',
    'HardRainNoon': 'HardRainNoon',
    # 日落天气 (7种)
    'ClearSunset': 'ClearSunset',
    'CloudySunset': 'CloudySunset',
    'WetSunset': 'WetSunset',
    'WetCloudySunset': 'WetCloudySunset',
    'SoftRainSunset': 'SoftRainSunset',
    'MidRainSunset': 'MidRainSunset',
    'HardRainSunset': 'HardRainSunset',
    # 夜晚天气 (7种)
    'ClearNight': 'ClearNight',
    'CloudyNight': 'CloudyNight',
    'WetNight': 'WetNight',
    'WetCloudyNight': 'WetCloudyNight',
    'SoftRainNight': 'SoftRainNight',
    'MidRainyNight': 'MidRainyNight',
    'HardRainNight': 'HardRainNight',
    # 特殊天气
    'DustStorm': 'DustStorm',
}

# 天气组合预设
WEATHER_COLLECTION_PRESETS = {
    'basic': ['ClearNoon', 'CloudyNoon', 'ClearSunset', 'ClearNight'],
    'all_noon': ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 
                 'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon'],
    'all_sunset': ['ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset',
                   'SoftRainSunset', 'MidRainSunset', 'HardRainSunset'],
    'all_night': ['ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight',
                  'SoftRainNight', 'MidRainyNight', 'HardRainNight'],
    'clear_all': ['ClearNoon', 'ClearSunset', 'ClearNight'],
    'rain_all': ['SoftRainNoon', 'MidRainyNoon', 'HardRainNoon',
                 'SoftRainSunset', 'MidRainSunset', 'HardRainSunset',
                 'SoftRainNight', 'MidRainyNight', 'HardRainNight'],
    'full': ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
             'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon',
             'ClearSunset', 'CloudySunset', 'WetSunset',
             'ClearNight', 'CloudyNight', 'WetNight'],
    'complete': list(WEATHER_PRESETS_MAP.keys()),  # 所有22种天气
}


@dataclass
class CustomWeatherParams:
    """自定义天气参数"""
    cloudiness: float = 0.0           # 云量 0-100
    precipitation: float = 0.0        # 降水量 0-100
    precipitation_deposits: float = 0.0  # 地面积水 0-100
    wind_intensity: float = 0.0       # 风力强度 0-100
    sun_azimuth_angle: float = 0.0    # 太阳方位角 0-360
    sun_altitude_angle: float = 75.0  # 太阳高度角 -90到90
    fog_density: float = 0.0          # 雾密度 0-100
    fog_distance: float = 0.0         # 雾起始距离（米）
    wetness: float = 0.0              # 地面湿度 0-100
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CustomWeatherParams':
        """从字典创建"""
        return cls(
            cloudiness=data.get('cloudiness', 0.0),
            precipitation=data.get('precipitation', 0.0),
            precipitation_deposits=data.get('precipitation_deposits', 0.0),
            wind_intensity=data.get('wind_intensity', 0.0),
            sun_azimuth_angle=data.get('sun_azimuth_angle', 0.0),
            sun_altitude_angle=data.get('sun_altitude_angle', 75.0),
            fog_density=data.get('fog_density', 0.0),
            fog_distance=data.get('fog_distance', 0.0),
            wetness=data.get('wetness', 0.0),
        )


@dataclass
class WeatherSetupConfig:
    """天气设置配置（用于 WeatherManager.set_weather 方法）
    
    注意：这个类与 config/settings.py 中的 WeatherConfig 不同：
    - WeatherConfig (settings.py): 用于 CollectorConfig，custom 是 Dict
    - WeatherSetupConfig (这里): 用于 WeatherManager，custom 是 CustomWeatherParams
    
    在 core/__init__.py 中，此类被导出为 WeatherManagerConfig 以避免混淆。
    """
    preset: Optional[str] = None      # 预设名称
    custom: Optional[CustomWeatherParams] = None  # 自定义参数
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeatherSetupConfig':
        """从字典创建"""
        preset = data.get('preset')
        custom_data = data.get('custom', {})
        custom = CustomWeatherParams.from_dict(custom_data) if custom_data else None
        return cls(preset=preset, custom=custom)


# 保持向后兼容的别名
WeatherConfig = WeatherSetupConfig


class WeatherManager:
    """
    天气管理器
    
    特性：
    - 支持 CARLA 内置的 22 种天气预设
    - 支持自定义天气参数
    - 提供天气组合预设用于多天气数据收集
    """
    
    def __init__(self, world):
        """
        初始化天气管理器
        
        参数:
            world: CARLA world 对象
        """
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.world = world
        self._current_weather: Optional[str] = None
    
    @property
    def current_weather(self) -> Optional[str]:
        """当前天气名称"""
        return self._current_weather
    
    @staticmethod
    def get_available_presets() -> List[str]:
        """获取所有可用的天气预设名称"""
        return list(WEATHER_PRESETS_MAP.keys())
    
    @staticmethod
    def get_collection_preset(preset_name: str) -> List[str]:
        """
        获取天气组合预设
        
        参数:
            preset_name: 预设名称 (basic/all_noon/all_sunset/all_night/clear_all/rain_all/full/complete)
            
        返回:
            天气名称列表
        """
        return WEATHER_COLLECTION_PRESETS.get(preset_name, ['ClearNoon'])
    
    @staticmethod
    def get_all_collection_presets() -> Dict[str, List[str]]:
        """获取所有天气组合预设"""
        return WEATHER_COLLECTION_PRESETS.copy()
    
    def set_weather(self, config: WeatherSetupConfig) -> bool:
        """
        设置天气
        
        参数:
            config: 天气配置 (WeatherSetupConfig)
            
        返回:
            是否成功设置
        """
        if config.preset:
            return self.set_weather_preset(config.preset)
        elif config.custom:
            return self.set_custom_weather(config.custom)
        return False
    
    def set_weather_preset(self, preset_name: str) -> bool:
        """
        设置预设天气
        
        参数:
            preset_name: 预设名称
            
        返回:
            是否成功设置
        """
        if preset_name not in WEATHER_PRESETS_MAP:
            print(f"⚠️ 未知天气预设: {preset_name}")
            return False
        
        try:
            weather_params = getattr(carla.WeatherParameters, preset_name, None)
            if weather_params is None:
                print(f"⚠️ CARLA 不支持天气预设: {preset_name}")
                return False
            
            self.world.set_weather(weather_params)
            self._current_weather = preset_name
            print(f"🌤️ 天气已设置: {preset_name}")
            return True
            
        except Exception as e:
            print(f"❌ 设置天气失败: {e}")
            return False
    
    def set_custom_weather(self, params: CustomWeatherParams) -> bool:
        """
        设置自定义天气
        
        参数:
            params: 自定义天气参数
            
        返回:
            是否成功设置
        """
        try:
            weather = carla.WeatherParameters(
                cloudiness=params.cloudiness,
                precipitation=params.precipitation,
                precipitation_deposits=params.precipitation_deposits,
                wind_intensity=params.wind_intensity,
                sun_azimuth_angle=params.sun_azimuth_angle,
                sun_altitude_angle=params.sun_altitude_angle,
                fog_density=params.fog_density,
                fog_distance=params.fog_distance,
                wetness=params.wetness,
            )
            
            self.world.set_weather(weather)
            self._current_weather = 'Custom'
            
            print(f"🌤️ 天气已设置: 自定义参数")
            print(f"   云量: {params.cloudiness}, 降水: {params.precipitation}, "
                  f"雾: {params.fog_density}")
            return True
            
        except Exception as e:
            print(f"❌ 设置自定义天气失败: {e}")
            return False
    
    def set_weather_from_dict(self, config_dict: Dict) -> bool:
        """
        从字典设置天气（兼容旧配置格式）
        
        参数:
            config_dict: 天气配置字典，格式：
                {'preset': 'ClearNoon'} 或
                {'preset': null, 'custom': {...}}
                
        返回:
            是否成功设置
        """
        if not config_dict:
            return False
        
        preset = config_dict.get('preset')
        
        # 如果有预设且不为空
        if preset:
            return self.set_weather_preset(preset)
        
        # 否则尝试使用自定义参数
        custom_dict = config_dict.get('custom', {})
        if custom_dict:
            params = CustomWeatherParams.from_dict(custom_dict)
            return self.set_custom_weather(params)
        
        return False


# ==================== 便捷函数 ====================

def get_weather_list(preset_name: str) -> List[str]:
    """
    根据预设名称获取天气列表
    
    参数:
        preset_name: 预设名称
        
    返回:
        天气名称列表
    """
    return WeatherManager.get_collection_preset(preset_name)


def is_valid_weather_preset(name: str) -> bool:
    """检查是否是有效的天气预设名称"""
    return name in WEATHER_PRESETS_MAP
