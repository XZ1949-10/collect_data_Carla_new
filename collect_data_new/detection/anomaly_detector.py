#!/usr/bin/env python
# coding=utf-8
"""
车辆异常行为检测器（智能版 v2.0）

检测以下异常：
1. 打转 (Spin) - 短时间内累计旋转角度过大
2. 翻车 (Rollover) - 车辆倾斜角度过大
3. 卡住 (Stuck) - 智能检测，区分以下情况：
   - 正常等红灯（不算卡住）
   - 正常让行/拥堵（不算卡住）
   - 真正卡住（有油门但不动、被障碍物阻挡等）

智能卡住检测逻辑：
- 综合考虑：速度、位置变化、油门状态、红绿灯、前方障碍物
- 只有在"尝试移动但无法移动"时才判定为卡住
- 等红灯、让行等正常停车不会被误判
"""

import time
import math
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque

# 支持独立运行和包导入两种方式
try:
    from ..config import AnomalyConfig
except ImportError:
    # 独立运行时，尝试从 sys.modules 获取
    import sys
    if 'collect_data_new.config' in sys.modules:
        AnomalyConfig = sys.modules['collect_data_new.config'].AnomalyConfig
    else:
        # 最后的降级方案：定义一个简单的配置类
        from dataclasses import dataclass as _dataclass
        @_dataclass
        class AnomalyConfig:
            """异常检测配置（降级版）"""
            enabled: bool = True
            spin_enabled: bool = True
            spin_threshold_degrees: float = 270.0
            spin_time_window: float = 3.0
            rollover_enabled: bool = True
            rollover_pitch_threshold: float = 45.0
            rollover_roll_threshold: float = 45.0
            stuck_enabled: bool = True
            stuck_speed_threshold: float = 0.5
            stuck_time_threshold: float = 5.0
            stuck_position_threshold: float = 0.5
            stuck_throttle_threshold: float = 0.1
            stuck_check_traffic_light: bool = True
            stuck_check_blocking: bool = True
            stuck_blocking_distance: float = 5.0
            stuck_max_wait_at_light: float = 60.0
            stuck_consecutive_attempts: int = 3

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class AnomalyType(Enum):
    """异常类型枚举"""
    NONE = auto()
    SPIN = auto()
    ROLLOVER = auto()
    STUCK = auto()
    STUCK_AT_LIGHT_TOO_LONG = auto()  # 等红灯时间过长（可能红绿灯故障）


class StuckReason(Enum):
    """卡住原因枚举"""
    NONE = auto()
    THROTTLE_NO_MOVEMENT = auto()  # 有油门但不动
    BLOCKED_BY_OBSTACLE = auto()   # 被障碍物阻挡
    POSITION_NO_CHANGE = auto()    # 位置长时间无变化
    TRAFFIC_LIGHT_TIMEOUT = auto() # 等红灯超时


@dataclass
class VehicleState:
    """车辆状态数据（扩展版）"""
    # 基础状态
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    # 位置
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # 控制状态
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    # 时间戳
    timestamp: float = 0.0
    
    @classmethod
    def from_carla_vehicle(cls, vehicle, timestamp: float = None) -> 'VehicleState':
        """从 CARLA 车辆对象创建状态"""
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        control = vehicle.get_control()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        return cls(
            pitch=transform.rotation.pitch,
            roll=transform.rotation.roll,
            yaw=transform.rotation.yaw,
            speed=speed,
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
            throttle=control.throttle,
            brake=control.brake,
            steer=control.steer,
            timestamp=timestamp or time.time()
        )
    
    def distance_to(self, other: 'VehicleState') -> float:
        """计算与另一个状态的位置距离"""
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )


@dataclass
class StuckAnalysis:
    """卡住分析结果"""
    is_stuck: bool = False
    reason: StuckReason = StuckReason.NONE
    duration: float = 0.0  # 卡住持续时间
    details: str = ""
    # 诊断信息
    speed: float = 0.0
    throttle: float = 0.0
    position_change: float = 0.0
    at_traffic_light: bool = False
    blocked_by: Optional[str] = None


class AnomalyDetector:
    """
    车辆异常行为检测器（智能版 v2.0）
    
    改进点：
    1. 智能卡住检测：区分正常停车和真正卡住
    2. 红绿灯感知：等红灯不算卡住
    3. 障碍物检测：检测前方是否有障碍物
    4. 油门状态分析：有油门但不动才算卡住
    5. 位置历史追踪：通过位置变化判断是否真的在移动
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None, world=None):
        """
        初始化检测器
        
        参数:
            config: 异常检测配置
            world: CARLA world 对象（用于红绿灯和障碍物检测）
        """
        self.config = config or AnomalyConfig()
        self.world = world
        
        self._anomaly_detected = False
        self._anomaly_type = AnomalyType.NONE
        
        # 打转检测
        self._yaw_history: List[Tuple[float, float]] = []
        
        # 智能卡住检测
        self._stuck_start_time: Optional[float] = None
        self._state_history: deque = deque(maxlen=100)  # 保存最近100个状态
        self._throttle_attempt_count: int = 0  # 有油门但不动的次数
        self._traffic_light_wait_start: Optional[float] = None
        self._last_stuck_analysis: Optional[StuckAnalysis] = None
        
        # 缓存
        self._cached_traffic_lights = None
        self._cache_update_time: float = 0.0
    
    def set_world(self, world):
        """设置 CARLA world（用于红绿灯和障碍物检测）"""
        self.world = world
        self._cached_traffic_lights = None
    
    @property
    def anomaly_detected(self) -> bool:
        return self._anomaly_detected
    
    @property
    def anomaly_type(self) -> AnomalyType:
        return self._anomaly_type
    
    @property
    def anomaly_type_name(self) -> str:
        names = {
            AnomalyType.NONE: '无',
            AnomalyType.SPIN: '打转',
            AnomalyType.ROLLOVER: '翻车',
            AnomalyType.STUCK: '卡住',
            AnomalyType.STUCK_AT_LIGHT_TOO_LONG: '等红灯超时'
        }
        return names.get(self._anomaly_type, '未知')
    
    @property
    def last_stuck_analysis(self) -> Optional[StuckAnalysis]:
        """获取最近一次卡住分析结果"""
        return self._last_stuck_analysis

    def configure(self, **kwargs) -> None:
        """配置检测参数"""
        config_map = {
            'enabled': 'enabled',
            'spin_enabled': 'spin_enabled',
            'spin_threshold': 'spin_threshold_degrees',
            'spin_time_window': 'spin_time_window',
            'rollover_enabled': 'rollover_enabled',
            'rollover_pitch': 'rollover_pitch_threshold',
            'rollover_roll': 'rollover_roll_threshold',
            'stuck_enabled': 'stuck_enabled',
            'stuck_speed': 'stuck_speed_threshold',
            'stuck_time': 'stuck_time_threshold',
            'stuck_position_threshold': 'stuck_position_threshold',
            'stuck_throttle_threshold': 'stuck_throttle_threshold',
            'stuck_check_traffic_light': 'stuck_check_traffic_light',
            'stuck_check_blocking': 'stuck_check_blocking',
            'stuck_blocking_distance': 'stuck_blocking_distance',
            'stuck_max_wait_at_light': 'stuck_max_wait_at_light',
            'stuck_consecutive_attempts': 'stuck_consecutive_attempts',
        }
        for key, attr in config_map.items():
            if key in kwargs:
                setattr(self.config, attr, kwargs[key])
    
    def check(self, vehicle_or_state, vehicle=None) -> bool:
        """
        检测车辆异常
        
        参数:
            vehicle_or_state: VehicleState 对象或 CARLA vehicle 对象
            vehicle: 可选，CARLA vehicle 对象（用于障碍物检测）
            
        返回:
            bool: 是否检测到异常
        """
        if not self.config.enabled:
            return False
        
        if self._anomaly_detected:
            return True
        
        # 获取车辆状态
        current_time = time.time()
        if isinstance(vehicle_or_state, VehicleState):
            state = vehicle_or_state
            carla_vehicle = vehicle
        else:
            try:
                state = VehicleState.from_carla_vehicle(vehicle_or_state, current_time)
                carla_vehicle = vehicle_or_state
            except Exception as e:
                print(f"⚠️ 获取车辆状态失败: {e}")
                return False
        
        # 保存状态历史
        self._state_history.append(state)
        
        # 依次检测各种异常
        if self._check_rollover(state):
            return True
        if self._check_spin(state, current_time):
            return True
        if self._check_stuck_smart(state, current_time, carla_vehicle):
            return True
        
        return False
    
    def _check_rollover(self, state: VehicleState) -> bool:
        """检测翻车"""
        if not self.config.rollover_enabled:
            return False
        
        pitch = abs(state.pitch)
        roll = abs(state.roll)
        
        if pitch > self.config.rollover_pitch_threshold or roll > self.config.rollover_roll_threshold:
            self._anomaly_detected = True
            self._anomaly_type = AnomalyType.ROLLOVER
            print(f"🔄 检测到翻车！俯仰角: {pitch:.1f}°, 横滚角: {roll:.1f}°")
            return True
        return False
    
    def _check_spin(self, state: VehicleState, current_time: float) -> bool:
        """检测打转"""
        if not self.config.spin_enabled:
            return False
        
        self._yaw_history.append((current_time, state.yaw))
        
        cutoff_time = current_time - self.config.spin_time_window
        self._yaw_history = [(t, y) for t, y in self._yaw_history if t >= cutoff_time]
        
        if len(self._yaw_history) >= 2:
            total_rotation = 0.0
            for i in range(1, len(self._yaw_history)):
                prev_yaw = self._yaw_history[i-1][1]
                curr_yaw = self._yaw_history[i][1]
                delta = curr_yaw - prev_yaw
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360
                total_rotation += abs(delta)
            
            if total_rotation > self.config.spin_threshold_degrees:
                self._anomaly_detected = True
                self._anomaly_type = AnomalyType.SPIN
                print(f"🌀 检测到打转！{self.config.spin_time_window:.1f}秒内旋转 {total_rotation:.1f}°")
                return True
        return False
    
    def _check_stuck_smart(self, state: VehicleState, current_time: float, 
                           vehicle=None) -> bool:
        """
        智能卡住检测
        
        检测逻辑：
        1. 首先检查是否在等红灯（不算卡住，但有超时限制）
        2. 检查是否有油门但速度为0（可能被阻挡）
        3. 检查位置是否长时间无变化
        4. 综合判断是否真正卡住
        """
        if not self.config.stuck_enabled:
            return False
        
        analysis = StuckAnalysis(
            speed=state.speed,
            throttle=state.throttle
        )
        
        # 计算位置变化（与历史状态比较）
        position_change = self._calculate_position_change(state, current_time)
        analysis.position_change = position_change
        
        # 检查是否在红绿灯前
        at_traffic_light = False
        traffic_light_state = None
        if self.config.stuck_check_traffic_light and vehicle is not None:
            at_traffic_light, traffic_light_state = self._check_at_traffic_light(vehicle)
            analysis.at_traffic_light = at_traffic_light
        
        # 检查前方是否有障碍物
        blocked_by = None
        if self.config.stuck_check_blocking and vehicle is not None:
            blocked_by = self._check_blocking_obstacle(vehicle)
            analysis.blocked_by = blocked_by
        
        # ========== 智能判断逻辑 ==========
        
        is_low_speed = state.speed < self.config.stuck_speed_threshold
        is_trying_to_move = state.throttle > self.config.stuck_throttle_threshold
        is_position_stuck = position_change < self.config.stuck_position_threshold
        
        # 情况1：等红灯（正常，但有超时限制）
        if at_traffic_light and traffic_light_state == 'Red':
            if self._traffic_light_wait_start is None:
                self._traffic_light_wait_start = current_time
                print(f"🚦 检测到等红灯...")
            else:
                wait_time = current_time - self._traffic_light_wait_start
                if wait_time > self.config.stuck_max_wait_at_light:
                    analysis.is_stuck = True
                    analysis.reason = StuckReason.TRAFFIC_LIGHT_TIMEOUT
                    analysis.duration = wait_time
                    analysis.details = f"等红灯超时 {wait_time:.1f}秒（可能红绿灯故障）"
                    self._last_stuck_analysis = analysis
                    self._anomaly_detected = True
                    self._anomaly_type = AnomalyType.STUCK_AT_LIGHT_TOO_LONG
                    print(f"⏰ {analysis.details}")
                    return True
            # 正常等红灯，重置其他卡住计时
            self._stuck_start_time = None
            self._throttle_attempt_count = 0
            self._last_stuck_analysis = analysis
            return False
        else:
            # 不在红灯前，重置红灯等待计时
            self._traffic_light_wait_start = None
        
        # 情况2：有油门但不动（可能被阻挡或真正卡住）
        if is_low_speed and is_trying_to_move:
            self._throttle_attempt_count += 1
            
            if self._throttle_attempt_count >= self.config.stuck_consecutive_attempts:
                # 连续多次有油门但不动
                if blocked_by:
                    analysis.reason = StuckReason.BLOCKED_BY_OBSTACLE
                    analysis.details = f"被 {blocked_by} 阻挡，油门 {state.throttle:.2f} 但速度 {state.speed:.2f}"
                else:
                    analysis.reason = StuckReason.THROTTLE_NO_MOVEMENT
                    analysis.details = f"有油门 {state.throttle:.2f} 但速度 {state.speed:.2f}，可能卡住"
                
                # 开始计时
                if self._stuck_start_time is None:
                    self._stuck_start_time = current_time
                elif current_time - self._stuck_start_time > self.config.stuck_time_threshold:
                    analysis.is_stuck = True
                    analysis.duration = current_time - self._stuck_start_time
                    self._last_stuck_analysis = analysis
                    self._anomaly_detected = True
                    self._anomaly_type = AnomalyType.STUCK
                    print(f"⏸️ 检测到卡住！{analysis.details}，持续 {analysis.duration:.1f}秒")
                    return True
        else:
            # 速度正常或没有油门，重置计数
            self._throttle_attempt_count = 0
        
        # 情况3：位置长时间无变化（即使没有油门）
        if is_position_stuck and is_low_speed:
            # 检查是否是正常停车（没有油门且前方有障碍物或红灯）
            is_normal_stop = (
                not is_trying_to_move and 
                (blocked_by is not None or at_traffic_light)
            )
            
            if not is_normal_stop:
                if self._stuck_start_time is None:
                    self._stuck_start_time = current_time
                elif current_time - self._stuck_start_time > self.config.stuck_time_threshold * 1.5:
                    # 位置无变化的阈值稍微宽松一些
                    analysis.is_stuck = True
                    analysis.reason = StuckReason.POSITION_NO_CHANGE
                    analysis.duration = current_time - self._stuck_start_time
                    analysis.details = f"位置 {self.config.stuck_time_threshold * 1.5:.1f}秒 无变化"
                    self._last_stuck_analysis = analysis
                    self._anomaly_detected = True
                    self._anomaly_type = AnomalyType.STUCK
                    print(f"⏸️ 检测到卡住！{analysis.details}")
                    return True
        else:
            # 位置有变化，重置计时
            if not is_trying_to_move or not is_low_speed:
                self._stuck_start_time = None
        
        self._last_stuck_analysis = analysis
        return False

    def _calculate_position_change(self, current_state: VehicleState, 
                                    current_time: float) -> float:
        """
        计算一段时间内的位置变化
        
        返回:
            float: 位置变化距离（米）
        """
        if len(self._state_history) < 2:
            return float('inf')  # 数据不足，假设在移动
        
        # 找到约 stuck_time_threshold 秒前的状态
        target_time = current_time - self.config.stuck_time_threshold
        old_state = None
        
        for state in self._state_history:
            if state.timestamp <= target_time:
                old_state = state
            else:
                break
        
        if old_state is None:
            # 历史数据不足
            old_state = self._state_history[0]
        
        return current_state.distance_to(old_state)
    
    def _check_at_traffic_light(self, vehicle) -> Tuple[bool, Optional[str]]:
        """
        检查车辆是否在红绿灯前
        
        返回:
            Tuple[bool, Optional[str]]: (是否在红绿灯前, 红绿灯状态)
        """
        if not CARLA_AVAILABLE or self.world is None:
            return False, None
        
        try:
            # 获取车辆前方的红绿灯
            vehicle_location = vehicle.get_location()
            
            # 更新红绿灯缓存（每秒更新一次）
            current_time = time.time()
            if self._cached_traffic_lights is None or \
               current_time - self._cache_update_time > 1.0:
                self._cached_traffic_lights = list(
                    self.world.get_actors().filter('traffic.traffic_light')
                )
                self._cache_update_time = current_time
            
            # 获取车辆朝向
            vehicle_transform = vehicle.get_transform()
            forward_vector = vehicle_transform.get_forward_vector()
            
            for tl in self._cached_traffic_lights:
                tl_location = tl.get_location()
                distance = vehicle_location.distance(tl_location)
                
                # 检查红绿灯是否在前方且距离较近
                if distance < 30.0:  # 30米范围内
                    # 计算红绿灯相对于车辆的方向
                    to_tl = carla.Vector3D(
                        tl_location.x - vehicle_location.x,
                        tl_location.y - vehicle_location.y,
                        0
                    )
                    # 归一化
                    to_tl_len = math.sqrt(to_tl.x**2 + to_tl.y**2)
                    if to_tl_len > 0:
                        to_tl.x /= to_tl_len
                        to_tl.y /= to_tl_len
                    
                    # 点积判断是否在前方
                    dot = forward_vector.x * to_tl.x + forward_vector.y * to_tl.y
                    if dot > 0.5:  # 在前方约60度范围内
                        state = tl.get_state()
                        state_str = {
                            carla.TrafficLightState.Red: 'Red',
                            carla.TrafficLightState.Yellow: 'Yellow',
                            carla.TrafficLightState.Green: 'Green',
                        }.get(state, 'Unknown')
                        return True, state_str
            
            return False, None
            
        except Exception as e:
            # 出错时假设不在红绿灯前
            return False, None
    
    def _check_blocking_obstacle(self, vehicle) -> Optional[str]:
        """
        检查前方是否有障碍物
        
        返回:
            Optional[str]: 障碍物类型，None 表示无障碍物
        """
        if not CARLA_AVAILABLE or self.world is None:
            return None
        
        try:
            vehicle_location = vehicle.get_location()
            vehicle_transform = vehicle.get_transform()
            forward_vector = vehicle_transform.get_forward_vector()
            
            # 检测前方的车辆
            vehicles = self.world.get_actors().filter('*vehicle*')
            for other in vehicles:
                if other.id == vehicle.id:
                    continue
                
                other_location = other.get_location()
                distance = vehicle_location.distance(other_location)
                
                if distance < self.config.stuck_blocking_distance:
                    # 检查是否在前方
                    to_other = carla.Vector3D(
                        other_location.x - vehicle_location.x,
                        other_location.y - vehicle_location.y,
                        0
                    )
                    to_other_len = math.sqrt(to_other.x**2 + to_other.y**2)
                    if to_other_len > 0:
                        to_other.x /= to_other_len
                        to_other.y /= to_other_len
                    
                    dot = forward_vector.x * to_other.x + forward_vector.y * to_other.y
                    if dot > 0.7:  # 在前方约45度范围内
                        return f"车辆({other.type_id})"
            
            # 检测前方的行人
            walkers = self.world.get_actors().filter('*walker*')
            for walker in walkers:
                walker_location = walker.get_location()
                distance = vehicle_location.distance(walker_location)
                
                if distance < self.config.stuck_blocking_distance:
                    to_walker = carla.Vector3D(
                        walker_location.x - vehicle_location.x,
                        walker_location.y - vehicle_location.y,
                        0
                    )
                    to_walker_len = math.sqrt(to_walker.x**2 + to_walker.y**2)
                    if to_walker_len > 0:
                        to_walker.x /= to_walker_len
                        to_walker.y /= to_walker_len
                    
                    dot = forward_vector.x * to_walker.x + forward_vector.y * to_walker.y
                    if dot > 0.7:
                        return "行人"
            
            # 检测静态障碍物（使用射线检测）
            # 注意：这需要更复杂的实现，这里简化处理
            
            return None
            
        except Exception as e:
            return None
    
    def reset(self) -> None:
        """重置检测状态"""
        self._anomaly_detected = False
        self._anomaly_type = AnomalyType.NONE
        self._yaw_history = []
        self._stuck_start_time = None
        self._state_history.clear()
        self._throttle_attempt_count = 0
        self._traffic_light_wait_start = None
        self._last_stuck_analysis = None
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前检测状态"""
        status = {
            'anomaly_detected': self._anomaly_detected,
            'anomaly_type': self._anomaly_type.name,
            'anomaly_type_name': self.anomaly_type_name,
            'yaw_history_length': len(self._yaw_history),
            'state_history_length': len(self._state_history),
            'throttle_attempt_count': self._throttle_attempt_count,
            'stuck_duration': (
                time.time() - self._stuck_start_time 
                if self._stuck_start_time else 0.0
            ),
            'traffic_light_wait_duration': (
                time.time() - self._traffic_light_wait_start
                if self._traffic_light_wait_start else 0.0
            ),
        }
        
        if self._last_stuck_analysis:
            status['last_analysis'] = {
                'is_stuck': self._last_stuck_analysis.is_stuck,
                'reason': self._last_stuck_analysis.reason.name,
                'speed': self._last_stuck_analysis.speed,
                'throttle': self._last_stuck_analysis.throttle,
                'position_change': self._last_stuck_analysis.position_change,
                'at_traffic_light': self._last_stuck_analysis.at_traffic_light,
                'blocked_by': self._last_stuck_analysis.blocked_by,
            }
        
        return status


# ==================== 兼容性：保留旧版简单检测器 ====================

class SimpleAnomalyDetector:
    """
    简单异常检测器（旧版兼容）
    
    仅基于速度和时间判断卡住，不考虑红绿灯等因素。
    适用于不需要智能检测的场景。
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        
        self._anomaly_detected = False
        self._anomaly_type = AnomalyType.NONE
        
        self._yaw_history: list = []
        self._stuck_start_time: Optional[float] = None
    
    @property
    def anomaly_detected(self) -> bool:
        return self._anomaly_detected
    
    @property
    def anomaly_type(self) -> AnomalyType:
        return self._anomaly_type
    
    def check(self, vehicle_or_state) -> bool:
        """检测车辆异常（简单版）"""
        if not self.config.enabled:
            return False
        
        if self._anomaly_detected:
            return True
        
        if isinstance(vehicle_or_state, VehicleState):
            state = vehicle_or_state
        else:
            try:
                state = VehicleState.from_carla_vehicle(vehicle_or_state)
            except Exception as e:
                print(f"⚠️ 获取车辆状态失败: {e}")
                return False
        
        current_time = time.time()
        
        # 简单的速度+时间判断
        if state.speed < self.config.stuck_speed_threshold:
            if self._stuck_start_time is None:
                self._stuck_start_time = current_time
            elif current_time - self._stuck_start_time > self.config.stuck_time_threshold:
                self._anomaly_detected = True
                self._anomaly_type = AnomalyType.STUCK
                print(f"⏸️ 检测到卡住！速度 {state.speed:.2f} m/s 持续 {self.config.stuck_time_threshold:.1f}秒")
                return True
        else:
            self._stuck_start_time = None
        
        return False
    
    def reset(self) -> None:
        """重置检测状态"""
        self._anomaly_detected = False
        self._anomaly_type = AnomalyType.NONE
        self._yaw_history = []
        self._stuck_start_time = None
