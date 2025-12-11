#!/usr/bin/env python
# coding=utf-8
"""
车辆异常行为检测器

检测以下异常：
1. 打转 (Spin) - 短时间内累计旋转角度过大
2. 翻车 (Rollover) - 车辆倾斜角度过大
3. 卡住 (Stuck) - 长时间速度接近0
"""

import time
from enum import Enum, auto
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..config import AnomalyConfig


class AnomalyType(Enum):
    """异常类型枚举"""
    NONE = auto()
    SPIN = auto()
    ROLLOVER = auto()
    STUCK = auto()


@dataclass
class VehicleState:
    """车辆状态数据"""
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    
    @classmethod
    def from_carla_vehicle(cls, vehicle) -> 'VehicleState':
        """从 CARLA 车辆对象创建状态"""
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        return cls(
            pitch=transform.rotation.pitch,
            roll=transform.rotation.roll,
            yaw=transform.rotation.yaw,
            speed=speed
        )


class AnomalyDetector:
    """车辆异常行为检测器"""
    
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
    
    @property
    def anomaly_type_name(self) -> str:
        names = {
            AnomalyType.NONE: '无',
            AnomalyType.SPIN: '打转',
            AnomalyType.ROLLOVER: '翻车',
            AnomalyType.STUCK: '卡住'
        }
        return names.get(self._anomaly_type, '未知')
    
    def configure(self, **kwargs) -> None:
        """配置检测参数"""
        if 'enabled' in kwargs:
            self.config.enabled = kwargs['enabled']
        if 'spin_enabled' in kwargs:
            self.config.spin_enabled = kwargs['spin_enabled']
        if 'spin_threshold' in kwargs:
            self.config.spin_threshold_degrees = kwargs['spin_threshold']
        if 'spin_time_window' in kwargs:
            self.config.spin_time_window = kwargs['spin_time_window']
        if 'rollover_enabled' in kwargs:
            self.config.rollover_enabled = kwargs['rollover_enabled']
        if 'rollover_pitch' in kwargs:
            self.config.rollover_pitch_threshold = kwargs['rollover_pitch']
        if 'rollover_roll' in kwargs:
            self.config.rollover_roll_threshold = kwargs['rollover_roll']
        if 'stuck_enabled' in kwargs:
            self.config.stuck_enabled = kwargs['stuck_enabled']
        if 'stuck_speed' in kwargs:
            self.config.stuck_speed_threshold = kwargs['stuck_speed']
        if 'stuck_time' in kwargs:
            self.config.stuck_time_threshold = kwargs['stuck_time']
    
    def check(self, vehicle_or_state) -> bool:
        """检测车辆异常"""
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
        
        if self._check_rollover(state):
            return True
        if self._check_spin(state, current_time):
            return True
        if self._check_stuck(state, current_time):
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
    
    def _check_stuck(self, state: VehicleState, current_time: float) -> bool:
        """检测卡住"""
        if not self.config.stuck_enabled:
            return False
        
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
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前检测状态"""
        return {
            'anomaly_detected': self._anomaly_detected,
            'anomaly_type': self._anomaly_type.name,
            'anomaly_type_name': self.anomaly_type_name,
            'yaw_history_length': len(self._yaw_history),
            'stuck_duration': (
                time.time() - self._stuck_start_time 
                if self._stuck_start_time else 0.0
            )
        }
