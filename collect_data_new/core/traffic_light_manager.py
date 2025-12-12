#!/usr/bin/env python
# coding=utf-8
"""
红绿灯管理模块

负责 CARLA 红绿灯时间设置和状态管理。
独立模块，可安全调用，不会造成卡顿。

使用示例:
    from collect_data_new.core import TrafficLightManager
    
    # 创建管理器
    tl_manager = TrafficLightManager(world)
    
    # 设置红绿灯时间
    tl_manager.set_timing(red=5.0, green=10.0, yellow=2.0)
    
    # 重置所有红绿灯
    tl_manager.reset_all()
    
    # 获取红绿灯信息
    info = tl_manager.get_traffic_lights_info()
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


# ==================== 配置类 ====================

class TrafficLightState(Enum):
    """红绿灯状态枚举"""
    RED = "Red"
    YELLOW = "Yellow"
    GREEN = "Green"
    OFF = "Off"
    UNKNOWN = "Unknown"


@dataclass
class TrafficLightTiming:
    """红绿灯时间配置"""
    red_time: float = 5.0      # 红灯时间（秒）
    green_time: float = 10.0   # 绿灯时间（秒）
    yellow_time: float = 2.0   # 黄灯时间（秒）
    
    @property
    def cycle_time(self) -> float:
        """完整周期时间"""
        return self.red_time + self.green_time + self.yellow_time
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrafficLightTiming':
        """从字典创建"""
        return cls(
            red_time=data.get('red_time', 5.0),
            green_time=data.get('green_time', 10.0),
            yellow_time=data.get('yellow_time', 2.0),
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'red_time': self.red_time,
            'green_time': self.green_time,
            'yellow_time': self.yellow_time,
            'cycle_time': self.cycle_time,
        }


@dataclass
class TrafficLightInfo:
    """单个红绿灯信息"""
    actor_id: int
    state: TrafficLightState
    location: Tuple[float, float, float]
    red_time: float
    green_time: float
    yellow_time: float
    elapsed_time: float  # 当前状态已经过的时间
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'actor_id': self.actor_id,
            'state': self.state.value,
            'location': self.location,
            'red_time': self.red_time,
            'green_time': self.green_time,
            'yellow_time': self.yellow_time,
            'elapsed_time': self.elapsed_time,
        }


# ==================== 预设配置 ====================

# 红绿灯时间预设
TRAFFIC_LIGHT_PRESETS = {
    'default': TrafficLightTiming(red_time=5.0, green_time=10.0, yellow_time=2.0),
    'fast': TrafficLightTiming(red_time=3.0, green_time=5.0, yellow_time=1.0),
    'slow': TrafficLightTiming(red_time=10.0, green_time=20.0, yellow_time=3.0),
    'balanced': TrafficLightTiming(red_time=8.0, green_time=15.0, yellow_time=2.0),
    'always_green': TrafficLightTiming(red_time=0.1, green_time=999.0, yellow_time=0.1),
    'always_red': TrafficLightTiming(red_time=999.0, green_time=0.1, yellow_time=0.1),
    'quick_cycle': TrafficLightTiming(red_time=2.0, green_time=3.0, yellow_time=1.0),
}


# ==================== 主管理器类 ====================

class TrafficLightManager:
    """
    红绿灯管理器
    
    特性：
    - 独立模块，不依赖其他收集器组件
    - 线程安全，不会造成卡顿
    - 支持预设和自定义时间配置
    - 支持批量和单个红绿灯操作
    - 提供红绿灯状态查询
    
    注意事项：
    - 所有操作都是非阻塞的
    - 设置时间后需要等待下一个周期才会生效
    - 在同步模式下使用时，建议在 tick 之间调用
    """
    
    def __init__(self, world, verbose: bool = True):
        """
        初始化红绿灯管理器
        
        参数:
            world: CARLA world 对象
            verbose: 是否打印详细信息
        """
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.world = world
        self.verbose = verbose
        self._current_timing: Optional[TrafficLightTiming] = None
        self._traffic_lights_cache: List = []
        self._cache_time: float = 0
        self._cache_ttl: float = 5.0  # 缓存有效期（秒）
    
    # ==================== 属性 ====================
    
    @property
    def current_timing(self) -> Optional[TrafficLightTiming]:
        """当前设置的时间配置"""
        return self._current_timing
    
    @property
    def traffic_light_count(self) -> int:
        """红绿灯数量"""
        return len(self._get_traffic_lights())
    
    # ==================== 核心方法 ====================
    
    def _get_traffic_lights(self, force_refresh: bool = False) -> List:
        """
        获取所有红绿灯（带缓存）
        
        参数:
            force_refresh: 是否强制刷新缓存
            
        返回:
            红绿灯 actor 列表
        """
        current_time = time.time()
        
        # 检查缓存是否有效
        if not force_refresh and self._traffic_lights_cache:
            if current_time - self._cache_time < self._cache_ttl:
                return self._traffic_lights_cache
        
        # 刷新缓存
        try:
            self._traffic_lights_cache = list(
                self.world.get_actors().filter('traffic.traffic_light')
            )
            self._cache_time = current_time
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 获取红绿灯列表失败: {e}")
            self._traffic_lights_cache = []
        
        return self._traffic_lights_cache
    
    def set_timing(self, red: float = None, green: float = None, 
                   yellow: float = None) -> bool:
        """
        设置红绿灯时间
        
        参数:
            red: 红灯时间（秒），None 则不修改
            green: 绿灯时间（秒），None 则不修改
            yellow: 黄灯时间（秒），None 则不修改
            
        返回:
            bool: 是否成功
        """
        traffic_lights = self._get_traffic_lights()
        
        if not traffic_lights:
            if self.verbose:
                print("⚠️ 未找到红绿灯")
            return False
        
        try:
            modified_count = 0
            for tl in traffic_lights:
                try:
                    if red is not None:
                        tl.set_red_time(red)
                    if green is not None:
                        tl.set_green_time(green)
                    if yellow is not None:
                        tl.set_yellow_time(yellow)
                    modified_count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️ 设置红绿灯 {tl.id} 失败: {e}")
            
            # 更新当前配置记录
            if self._current_timing is None:
                self._current_timing = TrafficLightTiming()
            if red is not None:
                self._current_timing.red_time = red
            if green is not None:
                self._current_timing.green_time = green
            if yellow is not None:
                self._current_timing.yellow_time = yellow
            
            if self.verbose:
                print(f"✅ 已更新 {modified_count}/{len(traffic_lights)} 个红绿灯时间")
                if red is not None:
                    print(f"   红灯: {red}秒")
                if green is not None:
                    print(f"   绿灯: {green}秒")
                if yellow is not None:
                    print(f"   黄灯: {yellow}秒")
            
            return modified_count > 0
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 设置红绿灯时间失败: {e}")
            return False

    
    def set_timing_from_config(self, timing: TrafficLightTiming) -> bool:
        """
        从配置对象设置红绿灯时间
        
        参数:
            timing: TrafficLightTiming 配置对象
            
        返回:
            bool: 是否成功
        """
        return self.set_timing(
            red=timing.red_time,
            green=timing.green_time,
            yellow=timing.yellow_time
        )
    
    def set_timing_preset(self, preset_name: str) -> bool:
        """
        使用预设配置设置红绿灯时间
        
        参数:
            preset_name: 预设名称 (default/fast/slow/balanced/always_green/always_red/quick_cycle)
            
        返回:
            bool: 是否成功
        """
        if preset_name not in TRAFFIC_LIGHT_PRESETS:
            if self.verbose:
                print(f"⚠️ 未知红绿灯预设: {preset_name}")
                print(f"   可用预设: {list(TRAFFIC_LIGHT_PRESETS.keys())}")
            return False
        
        timing = TRAFFIC_LIGHT_PRESETS[preset_name]
        if self.verbose:
            print(f"🚦 使用预设: {preset_name}")
        return self.set_timing_from_config(timing)
    
    def reset_all(self) -> bool:
        """
        重置所有红绿灯状态
        
        让所有红绿灯重新开始计时周期。
        
        返回:
            bool: 是否成功
        """
        try:
            self.world.reset_all_traffic_lights()
            if self.verbose:
                print("✅ 已重置所有红绿灯")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ 重置红绿灯失败: {e}")
            return False
    
    def freeze_all(self, state: TrafficLightState = TrafficLightState.GREEN) -> bool:
        """
        冻结所有红绿灯到指定状态
        
        参数:
            state: 目标状态
            
        返回:
            bool: 是否成功
        """
        traffic_lights = self._get_traffic_lights()
        
        if not traffic_lights:
            if self.verbose:
                print("⚠️ 未找到红绿灯")
            return False
        
        try:
            # 映射状态到 CARLA 枚举
            state_map = {
                TrafficLightState.RED: carla.TrafficLightState.Red,
                TrafficLightState.YELLOW: carla.TrafficLightState.Yellow,
                TrafficLightState.GREEN: carla.TrafficLightState.Green,
                TrafficLightState.OFF: carla.TrafficLightState.Off,
            }
            
            carla_state = state_map.get(state, carla.TrafficLightState.Green)
            
            for tl in traffic_lights:
                try:
                    tl.set_state(carla_state)
                    tl.freeze(True)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️ 冻结红绿灯 {tl.id} 失败: {e}")
            
            if self.verbose:
                print(f"✅ 已冻结 {len(traffic_lights)} 个红绿灯为 {state.value}")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 冻结红绿灯失败: {e}")
            return False
    
    def unfreeze_all(self) -> bool:
        """
        解冻所有红绿灯
        
        返回:
            bool: 是否成功
        """
        traffic_lights = self._get_traffic_lights()
        
        if not traffic_lights:
            return True
        
        try:
            for tl in traffic_lights:
                try:
                    tl.freeze(False)
                except:
                    pass
            
            if self.verbose:
                print(f"✅ 已解冻 {len(traffic_lights)} 个红绿灯")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 解冻红绿灯失败: {e}")
            return False
    
    # ==================== 查询方法 ====================
    
    def get_traffic_lights_info(self) -> List[TrafficLightInfo]:
        """
        获取所有红绿灯的详细信息
        
        返回:
            TrafficLightInfo 列表
        """
        traffic_lights = self._get_traffic_lights(force_refresh=True)
        result = []
        
        for tl in traffic_lights:
            try:
                # 获取状态
                carla_state = tl.get_state()
                state_map = {
                    carla.TrafficLightState.Red: TrafficLightState.RED,
                    carla.TrafficLightState.Yellow: TrafficLightState.YELLOW,
                    carla.TrafficLightState.Green: TrafficLightState.GREEN,
                    carla.TrafficLightState.Off: TrafficLightState.OFF,
                }
                state = state_map.get(carla_state, TrafficLightState.UNKNOWN)
                
                # 获取位置
                loc = tl.get_location()
                location = (loc.x, loc.y, loc.z)
                
                info = TrafficLightInfo(
                    actor_id=tl.id,
                    state=state,
                    location=location,
                    red_time=tl.get_red_time(),
                    green_time=tl.get_green_time(),
                    yellow_time=tl.get_yellow_time(),
                    elapsed_time=tl.get_elapsed_time(),
                )
                result.append(info)
                
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ 获取红绿灯 {tl.id} 信息失败: {e}")
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取红绿灯摘要信息
        
        返回:
            摘要字典
        """
        infos = self.get_traffic_lights_info()
        
        state_counts = {
            'red': 0,
            'yellow': 0,
            'green': 0,
            'off': 0,
            'unknown': 0,
        }
        
        for info in infos:
            if info.state == TrafficLightState.RED:
                state_counts['red'] += 1
            elif info.state == TrafficLightState.YELLOW:
                state_counts['yellow'] += 1
            elif info.state == TrafficLightState.GREEN:
                state_counts['green'] += 1
            elif info.state == TrafficLightState.OFF:
                state_counts['off'] += 1
            else:
                state_counts['unknown'] += 1
        
        return {
            'total_count': len(infos),
            'state_counts': state_counts,
            'current_timing': self._current_timing.to_dict() if self._current_timing else None,
        }
    
    def print_status(self):
        """打印红绿灯状态摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("🚦 红绿灯状态")
        print("="*50)
        print(f"总数: {summary['total_count']}")
        print(f"状态分布:")
        print(f"  🔴 红灯: {summary['state_counts']['red']}")
        print(f"  🟡 黄灯: {summary['state_counts']['yellow']}")
        print(f"  🟢 绿灯: {summary['state_counts']['green']}")
        
        if summary['current_timing']:
            timing = summary['current_timing']
            print(f"当前时间配置:")
            print(f"  红灯: {timing['red_time']}秒")
            print(f"  绿灯: {timing['green_time']}秒")
            print(f"  黄灯: {timing['yellow_time']}秒")
            print(f"  周期: {timing['cycle_time']}秒")
        print("="*50 + "\n")

    
    # ==================== 单个红绿灯操作 ====================
    
    def set_single_timing(self, actor_id: int, red: float = None, 
                          green: float = None, yellow: float = None) -> bool:
        """
        设置单个红绿灯的时间
        
        参数:
            actor_id: 红绿灯 actor ID
            red: 红灯时间
            green: 绿灯时间
            yellow: 黄灯时间
            
        返回:
            bool: 是否成功
        """
        traffic_lights = self._get_traffic_lights()
        
        for tl in traffic_lights:
            if tl.id == actor_id:
                try:
                    if red is not None:
                        tl.set_red_time(red)
                    if green is not None:
                        tl.set_green_time(green)
                    if yellow is not None:
                        tl.set_yellow_time(yellow)
                    return True
                except Exception as e:
                    if self.verbose:
                        print(f"❌ 设置红绿灯 {actor_id} 失败: {e}")
                    return False
        
        if self.verbose:
            print(f"⚠️ 未找到红绿灯 ID: {actor_id}")
        return False
    
    def set_single_state(self, actor_id: int, state: TrafficLightState, 
                         freeze: bool = False) -> bool:
        """
        设置单个红绿灯的状态
        
        参数:
            actor_id: 红绿灯 actor ID
            state: 目标状态
            freeze: 是否冻结
            
        返回:
            bool: 是否成功
        """
        traffic_lights = self._get_traffic_lights()
        
        state_map = {
            TrafficLightState.RED: carla.TrafficLightState.Red,
            TrafficLightState.YELLOW: carla.TrafficLightState.Yellow,
            TrafficLightState.GREEN: carla.TrafficLightState.Green,
            TrafficLightState.OFF: carla.TrafficLightState.Off,
        }
        
        for tl in traffic_lights:
            if tl.id == actor_id:
                try:
                    carla_state = state_map.get(state, carla.TrafficLightState.Green)
                    tl.set_state(carla_state)
                    if freeze:
                        tl.freeze(True)
                    return True
                except Exception as e:
                    if self.verbose:
                        print(f"❌ 设置红绿灯 {actor_id} 状态失败: {e}")
                    return False
        
        if self.verbose:
            print(f"⚠️ 未找到红绿灯 ID: {actor_id}")
        return False
    
    # ==================== 区域操作 ====================
    
    def get_traffic_lights_in_radius(self, location: Tuple[float, float, float], 
                                      radius: float) -> List[TrafficLightInfo]:
        """
        获取指定位置半径内的红绿灯
        
        参数:
            location: 中心位置 (x, y, z)
            radius: 搜索半径（米）
            
        返回:
            TrafficLightInfo 列表
        """
        all_infos = self.get_traffic_lights_info()
        result = []
        
        for info in all_infos:
            # 计算距离
            dx = info.location[0] - location[0]
            dy = info.location[1] - location[1]
            dz = info.location[2] - location[2]
            distance = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            if distance <= radius:
                result.append(info)
        
        return result
    
    def set_timing_in_radius(self, location: Tuple[float, float, float], 
                              radius: float, red: float = None, 
                              green: float = None, yellow: float = None) -> int:
        """
        设置指定位置半径内红绿灯的时间
        
        参数:
            location: 中心位置 (x, y, z)
            radius: 搜索半径（米）
            red, green, yellow: 时间设置
            
        返回:
            int: 修改的红绿灯数量
        """
        infos = self.get_traffic_lights_in_radius(location, radius)
        modified = 0
        
        for info in infos:
            if self.set_single_timing(info.actor_id, red, green, yellow):
                modified += 1
        
        if self.verbose and modified > 0:
            print(f"✅ 已修改 {modified} 个红绿灯（半径 {radius}m 内）")
        
        return modified
    
    # ==================== 上下文管理器 ====================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 可选：退出时恢复默认设置
        return False


# ==================== 便捷函数 ====================

def get_traffic_light_presets() -> Dict[str, TrafficLightTiming]:
    """获取所有红绿灯时间预设"""
    return TRAFFIC_LIGHT_PRESETS.copy()


def create_traffic_light_manager(world, verbose: bool = True) -> TrafficLightManager:
    """
    创建红绿灯管理器的便捷函数
    
    参数:
        world: CARLA world 对象
        verbose: 是否打印详细信息
        
    返回:
        TrafficLightManager 实例
    """
    return TrafficLightManager(world, verbose=verbose)


def configure_traffic_lights(world, red: float = 5.0, green: float = 10.0, 
                              yellow: float = 2.0, verbose: bool = True) -> bool:
    """
    一次性配置红绿灯时间的便捷函数
    
    参数:
        world: CARLA world 对象
        red: 红灯时间
        green: 绿灯时间
        yellow: 黄灯时间
        verbose: 是否打印信息
        
    返回:
        bool: 是否成功
    """
    manager = TrafficLightManager(world, verbose=verbose)
    return manager.set_timing(red=red, green=green, yellow=yellow)
