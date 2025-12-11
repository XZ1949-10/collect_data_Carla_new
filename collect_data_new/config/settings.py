#!/usr/bin/env python
# coding=utf-8
"""
配置类和常量定义

集中管理所有配置参数，便于维护和修改。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ==================== 常量定义 ====================

# 导航命令映射（与CARLA RoadOption对应）
COMMAND_NAMES = {
    2: 'Follow',      # RoadOption.LANEFOLLOW
    3: 'Left',        # RoadOption.LEFT
    4: 'Right',       # RoadOption.RIGHT
    5: 'Straight'     # RoadOption.STRAIGHT
}

# 命令颜色（BGR格式，用于可视化）
COMMAND_COLORS = {
    2: (100, 255, 100),   # 绿色 - Follow
    3: (100, 100, 255),   # 蓝色 - Left
    4: (255, 100, 100),   # 红色 - Right
    5: (255, 255, 100)    # 黄色 - Straight
}

# 默认噪声模式配置（帧数格式，与旧版兼容）
# 注意：这些值与 collect_data_old/noiser.py 保持一致
DEFAULT_NOISE_MODES = {
    'impulse': {
        'noise_frames': [6, 12],       # 旧版值，短脉冲
        'decay_frames': [4, 8],
        'idle_frames': [5, 15],
        'strength_percent': 100,
        'probability_percent': 25,
    },
    'smooth': {
        'noise_frames': [15, 25],      # 旧版值
        'decay_frames': [8, 15],
        'idle_frames': [5, 15],
        'strength_percent': 80,
        'probability_percent': 35,
    },
    'drift': {
        'noise_frames': [20, 35],      # 旧版值
        'decay_frames': [10, 20],
        'idle_frames': [5, 15],
        'strength_percent': 40,
        'probability_percent': 20,
    },
    'jitter': {
        'noise_frames': [10, 20],      # 旧版值
        'decay_frames': [5, 10],
        'idle_frames': [5, 15],
        'strength_percent': 50,
        'probability_percent': 20,
    }
}


# ==================== 配置类 ====================

@dataclass
class VisualizationInfo:
    """可视化信息（用于传递给FrameVisualizer）
    
    这是一个数据传输对象，用于解耦收集器和可视化器之间的依赖。
    收集器负责填充这些信息，可视化器负责显示。
    """
    # 噪声信息
    noise_enabled: bool = False
    lateral_enabled: bool = False
    longitudinal_enabled: bool = False
    lateral_active: bool = False
    longitudinal_active: bool = False
    
    # 专家控制（标签值，保存到数据集的值）
    expert_steer: float = 0.0
    expert_throttle: float = 0.0
    expert_brake: float = 0.0
    
    # 实际控制（车辆执行的值，可能带噪声）
    actual_steer: float = 0.0
    actual_throttle: float = 0.0
    actual_brake: float = 0.0
    
    def to_noise_info(self) -> dict:
        """转换为噪声信息字典（兼容FrameVisualizer接口）"""
        return {
            'enabled': self.noise_enabled,
            'lateral_enabled': self.lateral_enabled,
            'longitudinal_enabled': self.longitudinal_enabled,
            'lateral_active': self.lateral_active,
            'longitudinal_active': self.longitudinal_active,
        }
    
    def to_control_info(self) -> dict:
        """转换为实际控制信息字典"""
        return {
            'steer': self.actual_steer,
            'throttle': self.actual_throttle,
            'brake': self.actual_brake,
        }
    
    def to_expert_control(self) -> dict:
        """转换为专家控制信息字典"""
        return {
            'steer': self.expert_steer,
            'throttle': self.expert_throttle,
            'brake': self.expert_brake,
        }


@dataclass
class CameraConfig:
    """摄像头配置"""
    raw_width: int = 800
    raw_height: int = 600
    output_width: int = 200
    output_height: int = 88
    fov: int = 90
    # 相对车辆位置
    location: Tuple[float, float, float] = (2.0, 0.0, 1.4)
    rotation: Tuple[float, float, float] = (0.0, -15.0, 0.0)  # roll, pitch, yaw
    # 图像裁剪区域（去除天空和车头）
    crop_top: int = 90
    crop_bottom: int = 485


@dataclass
class NoiseConfig:
    """噪声配置"""
    enabled: bool = False
    lateral_enabled: bool = True      # 横向噪声（转向）
    longitudinal_enabled: bool = False  # 纵向噪声（油门/刹车）
    noise_ratio: float = 0.4          # 噪声时间占比
    max_steer_offset: float = 0.35    # 最大转向偏移
    max_throttle_offset: float = 0.2  # 最大油门偏移
    mode_config: Optional[Dict] = None  # 噪声模式配置


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    enabled: bool = True
    # 打转检测
    spin_enabled: bool = True
    spin_threshold_degrees: float = 270.0
    spin_time_window: float = 3.0
    # 翻车检测
    rollover_enabled: bool = True
    rollover_pitch_threshold: float = 45.0
    rollover_roll_threshold: float = 45.0
    # 卡住检测
    stuck_enabled: bool = True
    stuck_speed_threshold: float = 0.5
    stuck_time_threshold: float = 5.0


@dataclass
class NPCConfig:
    """NPC配置"""
    # 车辆配置
    num_vehicles: int = 0
    # NPC车辆交通规则总开关：True=遵守所有规则，False=使用下面的详细配置
    vehicles_obey_traffic_rules: bool = False
    vehicles_ignore_lights: bool = True
    vehicles_ignore_signs: bool = True
    vehicles_ignore_walkers: bool = False
    vehicle_filter: str = 'vehicle.*'
    four_wheels_only: bool = True
    use_back_spawn_points: bool = True
    # NPC车辆行为参数
    vehicle_distance: float = 3.0  # 跟车距离（米），与前车保持的最小距离
    vehicle_speed_difference: float = 30.0  # 速度差异百分比，相对于限速的随机偏差范围
    # 行人配置
    num_walkers: int = 0
    walker_filter: str = 'walker.pedestrian.*'
    walker_speed_range: Tuple[float, float] = (1.0, 2.0)
    
    def get_effective_ignore_lights(self) -> bool:
        """获取实际的忽略红绿灯设置（考虑总开关）"""
        if self.vehicles_obey_traffic_rules:
            return False  # 遵守规则 = 不忽略
        return self.vehicles_ignore_lights
    
    def get_effective_ignore_signs(self) -> bool:
        """获取实际的忽略交通标志设置（考虑总开关）"""
        if self.vehicles_obey_traffic_rules:
            return False
        return self.vehicles_ignore_signs
    
    def get_effective_ignore_walkers(self) -> bool:
        """获取实际的忽略行人设置（考虑总开关）"""
        if self.vehicles_obey_traffic_rules:
            return False
        return self.vehicles_ignore_walkers


@dataclass
class WeatherConfig:
    """天气配置"""
    preset: Optional[str] = 'ClearNoon'  # 预设名称
    custom: Optional[Dict[str, float]] = None  # 自定义参数
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeatherConfig':
        """从字典创建"""
        return cls(
            preset=data.get('preset', 'ClearNoon'),
            custom=data.get('custom'),
        )


@dataclass
class MultiWeatherConfig:
    """多天气收集配置"""
    enabled: bool = False
    weather_preset: str = 'basic'  # 天气组合预设名称
    custom_weather_list: List[str] = field(default_factory=list)  # 自定义天气列表
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MultiWeatherConfig':
        """从字典创建"""
        return cls(
            enabled=data.get('enabled', False),
            weather_preset=data.get('weather_preset', 'basic'),
            custom_weather_list=data.get('custom_weather_list', []),
        )
    
    def get_weather_list(self) -> List[str]:
        """获取天气列表"""
        # 优先使用自定义列表
        if self.custom_weather_list:
            return self.custom_weather_list
        
        # 否则使用预设
        from ..core.weather_manager import get_weather_list
        return get_weather_list(self.weather_preset)


@dataclass
class RouteConfig:
    """路线生成配置"""
    strategy: str = 'smart'
    min_distance: float = 50.0
    max_distance: float = 500.0
    target_routes_ratio: float = 1.0
    overlap_threshold: float = 0.5
    turn_priority_ratio: float = 0.7
    max_candidates_to_analyze: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RouteConfig':
        """从字典创建"""
        return cls(
            strategy=data.get('strategy', 'smart'),
            min_distance=data.get('min_distance', 50.0),
            max_distance=data.get('max_distance', 500.0),
            target_routes_ratio=data.get('target_routes_ratio', 1.0),
            overlap_threshold=data.get('overlap_threshold', 0.5),
            turn_priority_ratio=data.get('turn_priority_ratio', 0.7),
            max_candidates_to_analyze=data.get('max_candidates_to_analyze', 0),
        )


@dataclass
class CollisionRecoveryConfig:
    """碰撞恢复配置"""
    enabled: bool = True
    max_collisions_per_route: int = 99
    min_distance_to_destination: float = 30.0
    recovery_skip_distance: float = 25.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollisionRecoveryConfig':
        """从字典创建"""
        return cls(
            enabled=data.get('enabled', True),
            max_collisions_per_route=data.get('max_collisions_per_route', 99),
            min_distance_to_destination=data.get('min_distance_to_destination', 30.0),
            recovery_skip_distance=data.get('recovery_skip_distance', 25.0),
        )


@dataclass
class AdvancedConfig:
    """高级设置"""
    enable_route_validation: bool = True
    retry_failed_routes: bool = False
    max_retries: int = 3
    pause_between_routes: int = 2
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AdvancedConfig':
        """从字典创建"""
        return cls(
            enable_route_validation=data.get('enable_route_validation', True),
            retry_failed_routes=data.get('retry_failed_routes', False),
            max_retries=data.get('max_retries', 3),
            pause_between_routes=data.get('pause_between_routes', 2),
        )


@dataclass
class CollectorConfig:
    """数据收集器主配置"""
    # CARLA连接
    host: str = 'localhost'
    port: int = 2000
    town: str = 'Town01'
    
    # 自车交通规则总开关：True=遵守所有规则，False=使用下面的详细配置
    obey_traffic_rules: bool = False
    # 交通规则详细配置（仅当 obey_traffic_rules=False 时生效）
    ignore_traffic_lights: bool = True
    ignore_signs: bool = True
    ignore_vehicles_percentage: int = 80
    
    def get_effective_ignore_lights(self) -> bool:
        """获取实际的忽略红绿灯设置（考虑总开关）"""
        if self.obey_traffic_rules:
            return False  # 遵守规则 = 不忽略
        return self.ignore_traffic_lights
    
    def get_effective_ignore_signs(self) -> bool:
        """获取实际的忽略交通标志设置（考虑总开关）"""
        if self.obey_traffic_rules:
            return False
        return self.ignore_signs
    
    def get_effective_ignore_vehicles_percentage(self) -> int:
        """获取实际的忽略车辆百分比（考虑总开关）"""
        if self.obey_traffic_rules:
            return 0  # 遵守规则 = 不忽略任何车辆
        return self.ignore_vehicles_percentage
    
    # 速度和帧率
    target_speed: float = 10.0
    simulation_fps: int = 20
    
    # 摄像头配置
    camera: CameraConfig = field(default_factory=CameraConfig)
    
    # 噪声配置
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    
    # 异常检测配置
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    
    # NPC配置
    npc: NPCConfig = field(default_factory=NPCConfig)
    
    # 天气配置
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    
    # 多天气配置
    multi_weather: MultiWeatherConfig = field(default_factory=MultiWeatherConfig)
    
    # 路线配置
    route: RouteConfig = field(default_factory=RouteConfig)
    
    # 碰撞恢复配置
    collision_recovery: CollisionRecoveryConfig = field(default_factory=CollisionRecoveryConfig)
    
    # 高级设置
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    
    # 数据保存
    save_path: str = './carla_data'
    segment_size: int = 200  # 每段数据帧数
    frames_per_route: int = 1000  # 每条路线最大帧数
    auto_save_interval: int = 200  # 自动保存间隔
    
    # 可视化
    enable_visualization: bool = True  # 默认启用可视化
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CollectorConfig':
        """从字典创建配置"""
        # 复制以避免修改原字典
        data = config_dict.copy()
        
        # 提取子配置
        camera_dict = data.pop('camera', {})
        noise_dict = data.pop('noise', {})
        anomaly_dict = data.pop('anomaly', {})
        npc_dict = data.pop('npc', {})
        weather_dict = data.pop('weather', {})
        multi_weather_dict = data.pop('multi_weather', {})
        route_dict = data.pop('route', {})
        collision_recovery_dict = data.pop('collision_recovery', {})
        advanced_dict = data.pop('advanced', {})
        
        return cls(
            host=data.get('host', 'localhost'),
            port=data.get('port', 2000),
            town=data.get('town', 'Town01'),
            # 交通规则配置
            obey_traffic_rules=data.get('obey_traffic_rules', False),
            ignore_traffic_lights=data.get('ignore_traffic_lights', True),
            ignore_signs=data.get('ignore_signs', True),
            ignore_vehicles_percentage=data.get('ignore_vehicles_percentage', 80),
            target_speed=data.get('target_speed', 10.0),
            simulation_fps=data.get('simulation_fps', 20),
            save_path=data.get('save_path', './carla_data'),
            segment_size=data.get('segment_size', 200),
            frames_per_route=data.get('frames_per_route', 1000),
            auto_save_interval=data.get('auto_save_interval', 200),
            enable_visualization=data.get('enable_visualization', False),
            camera=CameraConfig(**camera_dict) if camera_dict else CameraConfig(),
            noise=NoiseConfig(**noise_dict) if noise_dict else NoiseConfig(),
            anomaly=AnomalyConfig(**anomaly_dict) if anomaly_dict else AnomalyConfig(),
            npc=NPCConfig(**npc_dict) if npc_dict else NPCConfig(),
            weather=WeatherConfig.from_dict(weather_dict) if weather_dict else WeatherConfig(),
            multi_weather=MultiWeatherConfig.from_dict(multi_weather_dict) if multi_weather_dict else MultiWeatherConfig(),
            route=RouteConfig.from_dict(route_dict) if route_dict else RouteConfig(),
            collision_recovery=CollisionRecoveryConfig.from_dict(collision_recovery_dict) if collision_recovery_dict else CollisionRecoveryConfig(),
            advanced=AdvancedConfig.from_dict(advanced_dict) if advanced_dict else AdvancedConfig(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)
