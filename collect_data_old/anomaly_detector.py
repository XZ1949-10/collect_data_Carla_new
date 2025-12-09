#!/usr/bin/env python
# coding=utf-8
"""
车辆异常行为检测器

独立模块，用于检测 CARLA 车辆的异常行为：
1. 打转 (Spin) - 短时间内累计旋转角度过大
2. 翻车 (Rollover) - 车辆倾斜角度过大
3. 卡住 (Stuck) - 长时间速度接近0

使用示例:
    from anomaly_detector import AnomalyDetector
    
    detector = AnomalyDetector()
    detector.configure(spin_threshold=270.0, stuck_time=5.0)
    
    # 在每帧调用
    if detector.check(vehicle):
        print(f"检测到异常: {detector.anomaly_type}")
        detector.reset()
"""

import time
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


class AnomalyType(Enum):
    """异常类型枚举"""
    NONE = auto()       # 无异常
    SPIN = auto()       # 打转
    ROLLOVER = auto()   # 翻车
    STUCK = auto()      # 卡住


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    # 总开关
    enabled: bool = True
    
    # 打转检测
    spin_enabled: bool = True
    spin_threshold_degrees: float = 270.0   # 累计旋转角度阈值（度）
    spin_time_window: float = 3.0           # 检测时间窗口（秒）
    
    # 翻车检测
    rollover_enabled: bool = True
    rollover_pitch_threshold: float = 45.0  # 俯仰角阈值（度）
    rollover_roll_threshold: float = 45.0   # 横滚角阈值（度）
    
    # 卡住检测
    stuck_enabled: bool = True
    stuck_speed_threshold: float = 0.5      # 速度阈值（m/s）
    stuck_time_threshold: float = 5.0       # 卡住时间阈值（秒）


@dataclass
class VehicleState:
    """车辆状态数据（用于解耦 CARLA 依赖）"""
    # 位置旋转
    pitch: float = 0.0      # 俯仰角（度）
    roll: float = 0.0       # 横滚角（度）
    yaw: float = 0.0        # 航向角（度）
    
    # 速度
    speed: float = 0.0      # 速度（m/s）
    
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
    """
    车辆异常行为检测器
    
    特性：
    - 支持三种异常检测：打转、翻车、卡住
    - 可配置各项阈值
    - 支持 CARLA 车辆对象或手动传入状态
    - 线程安全的状态管理
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """
        初始化检测器
        
        参数:
            config: 检测配置，None 则使用默认配置
        """
        self.config = config or AnomalyConfig()
        
        # 检测状态
        self._anomaly_detected = False
        self._anomaly_type = AnomalyType.NONE
        
        # 打转检测状态
        self._yaw_history: list = []  # [(timestamp, yaw), ...]
        
        # 卡住检测状态
        self._stuck_start_time: Optional[float] = None
        
    # ==================== 属性 ====================
    
    @property
    def anomaly_detected(self) -> bool:
        """是否检测到异常"""
        return self._anomaly_detected
    
    @property
    def anomaly_type(self) -> AnomalyType:
        """异常类型"""
        return self._anomaly_type
    
    @property
    def anomaly_type_name(self) -> str:
        """异常类型名称（中文）"""
        names = {
            AnomalyType.NONE: '无',
            AnomalyType.SPIN: '打转',
            AnomalyType.ROLLOVER: '翻车',
            AnomalyType.STUCK: '卡住'
        }
        return names.get(self._anomaly_type, '未知')
    
    # ==================== 配置 ====================
    
    def configure(self, **kwargs) -> None:
        """
        配置检测参数
        
        支持的参数:
            enabled: 总开关
            spin_enabled: 打转检测开关
            spin_threshold: 打转角度阈值（度）
            spin_time_window: 打转检测时间窗口（秒）
            rollover_enabled: 翻车检测开关
            rollover_pitch: 翻车俯仰角阈值（度）
            rollover_roll: 翻车横滚角阈值（度）
            stuck_enabled: 卡住检测开关
            stuck_speed: 卡住速度阈值（m/s）
            stuck_time: 卡住时间阈值（秒）
        """
        if 'enabled' in kwargs:
            self.config.enabled = kwargs['enabled']
        
        # 打转配置
        if 'spin_enabled' in kwargs:
            self.config.spin_enabled = kwargs['spin_enabled']
        if 'spin_threshold' in kwargs:
            self.config.spin_threshold_degrees = kwargs['spin_threshold']
        if 'spin_time_window' in kwargs:
            self.config.spin_time_window = kwargs['spin_time_window']
        
        # 翻车配置
        if 'rollover_enabled' in kwargs:
            self.config.rollover_enabled = kwargs['rollover_enabled']
        if 'rollover_pitch' in kwargs:
            self.config.rollover_pitch_threshold = kwargs['rollover_pitch']
        if 'rollover_roll' in kwargs:
            self.config.rollover_roll_threshold = kwargs['rollover_roll']
        
        # 卡住配置
        if 'stuck_enabled' in kwargs:
            self.config.stuck_enabled = kwargs['stuck_enabled']
        if 'stuck_speed' in kwargs:
            self.config.stuck_speed_threshold = kwargs['stuck_speed']
        if 'stuck_time' in kwargs:
            self.config.stuck_time_threshold = kwargs['stuck_time']
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            'enabled': self.config.enabled,
            'spin_enabled': self.config.spin_enabled,
            'spin_threshold_degrees': self.config.spin_threshold_degrees,
            'spin_time_window': self.config.spin_time_window,
            'rollover_enabled': self.config.rollover_enabled,
            'rollover_pitch_threshold': self.config.rollover_pitch_threshold,
            'rollover_roll_threshold': self.config.rollover_roll_threshold,
            'stuck_enabled': self.config.stuck_enabled,
            'stuck_speed_threshold': self.config.stuck_speed_threshold,
            'stuck_time_threshold': self.config.stuck_time_threshold,
        }
    
    # ==================== 检测 ====================
    
    def check(self, vehicle_or_state) -> bool:
        """
        检测车辆异常
        
        参数:
            vehicle_or_state: CARLA 车辆对象 或 VehicleState 对象
            
        返回:
            bool: 是否检测到异常
        """
        if not self.config.enabled:
            return False
        
        if self._anomaly_detected:
            return True
        
        # 获取车辆状态
        if isinstance(vehicle_or_state, VehicleState):
            state = vehicle_or_state
        else:
            # 假设是 CARLA 车辆对象
            try:
                state = VehicleState.from_carla_vehicle(vehicle_or_state)
            except Exception as e:
                print(f"⚠️ 获取车辆状态失败: {e}")
                return False
        
        current_time = time.time()
        
        # 1. 翻车检测（优先级最高）
        if self._check_rollover(state):
            return True
        
        # 2. 打转检测
        if self._check_spin(state, current_time):
            return True
        
        # 3. 卡住检测
        if self._check_stuck(state, current_time):
            return True
        
        return False
    
    def check_with_state(self, pitch: float, roll: float, yaw: float, speed: float) -> bool:
        """
        使用手动传入的状态检测异常
        
        参数:
            pitch: 俯仰角（度）
            roll: 横滚角（度）
            yaw: 航向角（度）
            speed: 速度（m/s）
            
        返回:
            bool: 是否检测到异常
        """
        state = VehicleState(pitch=pitch, roll=roll, yaw=yaw, speed=speed)
        return self.check(state)
    
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
        
        yaw = state.yaw
        self._yaw_history.append((current_time, yaw))
        
        # 清理过期数据
        cutoff_time = current_time - self.config.spin_time_window
        self._yaw_history = [(t, y) for t, y in self._yaw_history if t >= cutoff_time]
        
        # 计算累计旋转角度
        if len(self._yaw_history) >= 2:
            total_rotation = 0.0
            for i in range(1, len(self._yaw_history)):
                prev_yaw = self._yaw_history[i-1][1]
                curr_yaw = self._yaw_history[i][1]
                
                # 处理角度跨越 -180/180 的情况
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
    
    # ==================== 状态管理 ====================
    
    def reset(self) -> None:
        """重置检测状态（在新 segment 开始时调用）"""
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


# ==================== 便捷函数 ====================

def create_detector_from_config(config_dict: Dict[str, Any]) -> AnomalyDetector:
    """
    从配置字典创建检测器
    
    参数:
        config_dict: 配置字典，支持以下键：
            - enabled
            - spin_detection.enabled, spin_detection.threshold_degrees, spin_detection.time_window
            - rollover_detection.enabled, rollover_detection.pitch_threshold, rollover_detection.roll_threshold
            - stuck_detection.enabled, stuck_detection.speed_threshold, stuck_detection.time_threshold
    
    返回:
        AnomalyDetector: 配置好的检测器
    """
    config = AnomalyConfig()
    
    config.enabled = config_dict.get('enabled', True)
    
    # 打转配置
    spin_cfg = config_dict.get('spin_detection', {})
    config.spin_enabled = spin_cfg.get('enabled', True)
    config.spin_threshold_degrees = spin_cfg.get('threshold_degrees', 270.0)
    config.spin_time_window = spin_cfg.get('time_window', 3.0)
    
    # 翻车配置
    rollover_cfg = config_dict.get('rollover_detection', {})
    config.rollover_enabled = rollover_cfg.get('enabled', True)
    config.rollover_pitch_threshold = rollover_cfg.get('pitch_threshold', 45.0)
    config.rollover_roll_threshold = rollover_cfg.get('roll_threshold', 45.0)
    
    # 卡住配置
    stuck_cfg = config_dict.get('stuck_detection', {})
    config.stuck_enabled = stuck_cfg.get('enabled', True)
    config.stuck_speed_threshold = stuck_cfg.get('speed_threshold', 0.5)
    config.stuck_time_threshold = stuck_cfg.get('time_threshold', 5.0)
    
    return AnomalyDetector(config)


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("="*60)
    print("异常检测器测试")
    print("="*60)
    
    # 创建检测器
    detector = AnomalyDetector()
    print(f"\n默认配置: {detector.get_config()}")
    
    # 测试翻车检测
    print("\n--- 测试翻车检测 ---")
    detector.reset()
    result = detector.check_with_state(pitch=50.0, roll=10.0, yaw=0.0, speed=5.0)
    print(f"俯仰角50°: 检测结果={result}, 类型={detector.anomaly_type_name}")
    
    # 测试打转检测
    print("\n--- 测试打转检测 ---")
    detector.reset()
    for i in range(100):
        yaw = i * 10  # 每帧旋转10度
        result = detector.check_with_state(pitch=0.0, roll=0.0, yaw=yaw % 360 - 180, speed=5.0)
        if result:
            print(f"第{i}帧检测到打转")
            break
        time.sleep(0.05)
    
    # 测试卡住检测
    print("\n--- 测试卡住检测 ---")
    detector.reset()
    detector.configure(stuck_time=2.0)  # 缩短测试时间
    for i in range(50):
        result = detector.check_with_state(pitch=0.0, roll=0.0, yaw=0.0, speed=0.1)
        if result:
            print(f"第{i}帧检测到卡住")
            break
        time.sleep(0.1)
    
    print("\n✅ 测试完成")
