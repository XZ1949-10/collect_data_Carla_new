#!/usr/bin/env python
# coding=utf-8
"""
噪声生成模块

设计逻辑：
1. segment开始时，按 probability_percent 选择3个噪声模式，随机排序
2. 动态执行：随机选取起始帧，执行完整周期（噪声+衰减+空闲）
3. 一个周期完成后才能开始下一个
"""

import random
import math
from typing import Dict, Optional, Any

from ..config import DEFAULT_NOISE_MODES


class Noiser:
    """动态规划噪声生成器"""

    def __init__(self, noise_type: str, max_offset: float = 0.35, fps: int = 20,
                 mode_config: Optional[Dict] = None, noise_ratio: float = 0.4,
                 segment_frames: int = 200):
        """
        初始化噪声生成器
        
        参数:
            noise_type: 'Spike'(转向), 'Throttle'(油门), 'None'(无)
            max_offset: 最大偏移量 (0-1)
            fps: 帧率
            mode_config: 噪声模式配置
            noise_ratio: 噪声帧占比 (0-1)
            segment_frames: segment大小
        """
        self.noise_type = noise_type
        self.fps = fps
        self.max_offset = max(0.05, min(1.0, max_offset))
        self.noise_ratio = max(0.0, min(0.95, noise_ratio))
        self.segment_frames = segment_frames
        
        self.noise_modes = self._parse_mode_config(mode_config)
        self.target_noise_frames = int(segment_frames * noise_ratio)
        
        # 状态变量
        self.frame_counter = 0
        self.selected_modes = []
        self.current_mode_idx = 0
        
        self.current_mode = None
        self.mode_strength = 1.0
        self.noise_direction = 1
        
        self.noise_start_frame = -1
        self.noise_frames = 0
        self.decay_frames = 0
        self.idle_frames = 0
        self.cycle_end_frame = -1
        
        self.state = 'idle'
        
        self.jitter_values = []
        self.jitter_index = 0
        
        self.mode_stats = {mode: 0 for mode in self.noise_modes}
        self.noise_frames_count = 0
        self.completed_cycles = 0
        
        self._next_trigger_frame = 0
        self._init_segment()
    
    def _parse_mode_config(self, config: Optional[Dict]) -> Dict:
        """解析配置"""
        if not config:
            return DEFAULT_NOISE_MODES.copy()
        
        parsed = {}
        for name, cfg in config.items():
            if name.startswith('_') or not isinstance(cfg, dict):
                continue
            
            default = DEFAULT_NOISE_MODES.get(name, {})
            parsed[name] = {
                'noise_frames': cfg.get('noise_frames', default.get('noise_frames', [10, 20])),
                'decay_frames': cfg.get('decay_frames', default.get('decay_frames', [5, 10])),
                'idle_frames': cfg.get('idle_frames', default.get('idle_frames', [5, 15])),
                'strength_percent': cfg.get('strength_percent', default.get('strength_percent', 100)),
                'probability_percent': cfg.get('probability_percent', default.get('probability_percent', 25)),
            }
        
        return parsed if parsed else DEFAULT_NOISE_MODES.copy()

    def _select_modes(self, count: int = 3) -> list:
        """按probability_percent加权随机选择模式"""
        modes = list(self.noise_modes.keys())
        weights = [self.noise_modes[m].get('probability_percent', 25) for m in modes]
        
        selected = []
        for _ in range(count):
            mode = random.choices(modes, weights=weights, k=1)[0]
            selected.append(mode)
        
        random.shuffle(selected)
        return selected
    
    def _init_segment(self):
        """初始化segment"""
        self.selected_modes = self._select_modes(3)
        self.current_mode_idx = 0
        self.state = 'idle'
        self.noise_start_frame = -1
        self.cycle_end_frame = -1
        
        max_start = max(1, self.segment_frames // 4)
        self._next_trigger_frame = random.randint(1, max_start)
    
    def _calculate_cycle_params(self) -> tuple:
        """计算当前周期的参数"""
        remaining_frames = self.segment_frames - self.frame_counter
        remaining_noise_needed = self.target_noise_frames - self.noise_frames_count
        remaining_modes = len(self.selected_modes) - self.current_mode_idx
        
        if remaining_modes <= 0 or remaining_noise_needed <= 0:
            return 0, 0, 0
        
        mode = self.selected_modes[self.current_mode_idx]
        config = self.noise_modes.get(mode, DEFAULT_NOISE_MODES.get(mode, {}))
        
        noise_range = config.get('noise_frames', [10, 20])
        decay_range = config.get('decay_frames', [5, 10])
        idle_range = config.get('idle_frames', [5, 15])
        
        noise_frames = random.randint(noise_range[0], noise_range[1])
        decay_frames = random.randint(decay_range[0], decay_range[1])
        idle_frames = random.randint(idle_range[0], idle_range[1])
        
        total_noise_decay = noise_frames + decay_frames
        if total_noise_decay > remaining_noise_needed:
            scale = remaining_noise_needed / total_noise_decay
            noise_frames = max(noise_range[0], int(noise_frames * scale))
            decay_frames = max(decay_range[0], int(decay_frames * scale))
        
        total_cycle = noise_frames + decay_frames + idle_frames
        max_allowed = int(remaining_frames * 0.8)
        if total_cycle > max_allowed:
            scale = max_allowed / total_cycle
            noise_frames = max(noise_range[0], int(noise_frames * scale))
            decay_frames = max(decay_range[0], int(decay_frames * scale))
            idle_frames = max(idle_range[0], int(idle_frames * scale))
        
        return noise_frames, decay_frames, idle_frames
    
    def _start_noise_cycle(self) -> bool:
        """开始一个新的噪声周期"""
        if self.current_mode_idx >= len(self.selected_modes):
            return False
        
        remaining = self.segment_frames - self.frame_counter
        if remaining < 15:
            return False
        
        self.noise_frames, self.decay_frames, self.idle_frames = self._calculate_cycle_params()
        
        if self.noise_frames < 5:
            return False
        
        self.current_mode = self.selected_modes[self.current_mode_idx]
        config = self.noise_modes[self.current_mode]
        self.mode_strength = config.get('strength_percent', 100) / 100.0
        self.noise_direction = random.choice([1, -1])
        
        self.noise_start_frame = self.frame_counter
        self.cycle_end_frame = self.frame_counter + self.noise_frames + self.decay_frames + self.idle_frames
        
        if self.current_mode == 'jitter':
            self._generate_jitter_sequence()
        
        self.state = 'noise'
        self.mode_stats[self.current_mode] += 1
        
        return True
    
    def _generate_jitter_sequence(self):
        """生成jitter模式的随机抖动序列"""
        total = self.noise_frames + self.decay_frames
        self.jitter_values = []
        self.jitter_index = 0
        
        current_value = 0.0
        for i in range(total):
            change = random.gauss(0, 0.3)
            current_value = current_value * 0.7 + change
            current_value = max(-1.0, min(1.0, current_value))
            
            if i >= self.noise_frames:
                decay_progress = (i - self.noise_frames) / max(1, self.decay_frames)
                current_value *= (1.0 - decay_progress)
            
            self.jitter_values.append(current_value)
    
    def _update_state(self):
        """更新状态机"""
        if self.state == 'idle':
            if self.frame_counter >= self._next_trigger_frame:
                self._start_noise_cycle()
        
        elif self.state == 'noise':
            elapsed = self.frame_counter - self.noise_start_frame
            if elapsed >= self.noise_frames:
                self.state = 'decay'
        
        elif self.state == 'decay':
            elapsed = self.frame_counter - self.noise_start_frame - self.noise_frames
            if elapsed >= self.decay_frames:
                self.state = 'idle_after'
        
        elif self.state == 'idle_after':
            if self.frame_counter >= self.cycle_end_frame:
                self.completed_cycles += 1
                self.current_mode_idx += 1
                self.state = 'idle'
                
                if self.current_mode_idx >= len(self.selected_modes):
                    remaining_noise_needed = self.target_noise_frames - self.noise_frames_count
                    if remaining_noise_needed > 10:
                        self.selected_modes = self._select_modes(3)
                        self.current_mode_idx = 0
                
                remaining = self.segment_frames - self.frame_counter
                if remaining > 20 and self.current_mode_idx < len(self.selected_modes):
                    self._next_trigger_frame = self.frame_counter + 1
                else:
                    self._next_trigger_frame = self.segment_frames + 1
    
    def _calculate_noise_value(self) -> float:
        """计算当前帧的噪声值"""
        if self.state not in ['noise', 'decay']:
            return 0.0
        
        max_noise = self.max_offset * self.mode_strength
        elapsed_noise = self.frame_counter - self.noise_start_frame
        elapsed_decay = elapsed_noise - self.noise_frames
        
        if self.current_mode == 'impulse':
            return self._calc_impulse(max_noise, elapsed_noise, elapsed_decay)
        elif self.current_mode == 'smooth':
            return self._calc_smooth(max_noise, elapsed_noise, elapsed_decay)
        elif self.current_mode == 'drift':
            return self._calc_drift(max_noise, elapsed_noise, elapsed_decay)
        elif self.current_mode == 'jitter':
            return self._calc_jitter(max_noise)
        
        return 0.0

    def _calc_impulse(self, max_noise: float, elapsed_noise: int, elapsed_decay: int) -> float:
        if self.state == 'noise':
            progress = min(1.0, elapsed_noise / max(1, self.noise_frames))
            return max_noise * (1.0 - math.exp(-5.0 * progress)) * self.noise_direction
        elif self.state == 'decay':
            progress = min(1.0, elapsed_decay / max(1, self.decay_frames))
            return max_noise * math.exp(-5.0 * progress) * self.noise_direction
        return 0.0
    
    def _calc_smooth(self, max_noise: float, elapsed_noise: int, elapsed_decay: int) -> float:
        if self.state == 'noise':
            progress = min(1.0, elapsed_noise / max(1, self.noise_frames))
            smooth = progress * progress * (3.0 - 2.0 * progress)
            return max_noise * smooth * self.noise_direction
        elif self.state == 'decay':
            progress = min(1.0, elapsed_decay / max(1, self.decay_frames))
            smooth = 1.0 - progress * progress * (3.0 - 2.0 * progress)
            return max_noise * smooth * self.noise_direction
        return 0.0
    
    def _calc_drift(self, max_noise: float, elapsed_noise: int, elapsed_decay: int) -> float:
        if self.state == 'noise':
            progress = min(1.0, elapsed_noise / max(1, self.noise_frames))
            return max_noise * math.sin(progress * math.pi / 2) * self.noise_direction
        elif self.state == 'decay':
            progress = min(1.0, elapsed_decay / max(1, self.decay_frames))
            return max_noise * math.cos(progress * math.pi / 2) * self.noise_direction
        return 0.0
    
    def _calc_jitter(self, max_noise: float) -> float:
        if self.jitter_index < len(self.jitter_values):
            value = max_noise * self.jitter_values[self.jitter_index]
            self.jitter_index += 1
            return value
        return 0.0
    
    def _copy_action(self, action):
        """
        创建 action 对象的副本
        
        支持 carla.VehicleControl 和类似的对象。
        使用模块级缓存的 carla 引用，避免每帧重复导入。
        
        参数:
            action: 原始控制对象
            
        返回:
            控制对象的副本
        """
        # 使用模块级缓存的 carla 引用
        if not hasattr(self, '_carla_module'):
            try:
                import carla
                self._carla_module = carla
            except ImportError:
                self._carla_module = None
        
        if self._carla_module is not None:
            new_action = self._carla_module.VehicleControl()
            new_action.steer = action.steer
            new_action.throttle = action.throttle
            new_action.brake = action.brake
            new_action.hand_brake = getattr(action, 'hand_brake', False)
            new_action.reverse = getattr(action, 'reverse', False)
            new_action.manual_gear_shift = getattr(action, 'manual_gear_shift', False)
            new_action.gear = getattr(action, 'gear', 0)
            return new_action
        else:
            # carla 不可用时，手动创建类似对象
            class VehicleControlCopy:
                pass
            new_action = VehicleControlCopy()
            new_action.steer = action.steer
            new_action.throttle = action.throttle
            new_action.brake = action.brake
            new_action.hand_brake = getattr(action, 'hand_brake', False)
            new_action.reverse = getattr(action, 'reverse', False)
            new_action.manual_gear_shift = getattr(action, 'manual_gear_shift', False)
            new_action.gear = getattr(action, 'gear', 0)
            return new_action

    def compute_noise(self, action, speed: float) -> tuple:
        """
        计算并应用噪声到控制信号
        
        注意：此方法会创建 action 的副本，不会修改原始对象。
        
        返回:
            tuple: (noisy_action, is_recovering, is_noise_active)
        """
        self.frame_counter += 1
        
        if self.noise_type == 'None':
            # 即使不添加噪声，也返回副本以保持一致性
            return self._copy_action(action), False, False
        
        # 边界检查：超出 segment 范围时停止噪声
        if self.frame_counter > self.segment_frames:
            # 返回副本而非原始对象，避免调用者意外修改原始action
            return self._copy_action(action), False, False
        
        self._update_state()
        
        is_noisy = self.state in ['noise', 'decay']
        if is_noisy:
            self.noise_frames_count += 1
        
        noise_value = self._calculate_noise_value()
        
        is_recovering = (self.state == 'decay')
        is_noise_active = (self.state == 'noise')
        
        # 创建 action 的副本，避免修改原始对象
        # 这对于保持 _expert_control 不被污染很重要
        noisy_action = self._copy_action(action)
        
        if self.noise_type == 'Spike':
            if is_noisy:
                noisy_steer = noisy_action.steer + noise_value
                noisy_action.steer = max(-1.0, min(1.0, noisy_steer))
            return noisy_action, is_recovering, is_noise_active
        
        elif self.noise_type == 'Throttle':
            if is_noisy:
                if noise_value > 0:
                    if noisy_action.brake > 0:
                        brake_reduction = min(noisy_action.brake, noise_value)
                        noisy_action.brake = max(0.0, noisy_action.brake - brake_reduction)
                        remaining = noise_value - brake_reduction
                        if remaining > 0:
                            noisy_action.throttle = max(0.0, min(1.0, noisy_action.throttle + remaining))
                    else:
                        noisy_action.throttle = max(0.0, min(1.0, noisy_action.throttle + noise_value))
                else:
                    abs_noise = abs(noise_value)
                    if noisy_action.throttle > 0:
                        throttle_reduction = min(noisy_action.throttle, abs_noise)
                        noisy_action.throttle = max(0.0, noisy_action.throttle - throttle_reduction)
                        remaining = abs_noise - throttle_reduction
                        if remaining > 0:
                            noisy_action.brake = max(0.0, min(1.0, noisy_action.brake + remaining))
                    else:
                        noisy_action.brake = max(0.0, min(1.0, noisy_action.brake + abs_noise))
            return noisy_action, is_recovering, is_noise_active
        
        return noisy_action, False, False
    
    def get_current_mode(self) -> Optional[str]:
        """获取当前噪声模式"""
        if self.state in ['noise', 'decay']:
            return self.current_mode
        return None
    
    @property
    def noise_being_set(self) -> bool:
        return self.state == 'noise'
    
    @property
    def remove_noise(self) -> bool:
        return self.state == 'decay'
    
    def get_mode_stats(self) -> Dict[str, int]:
        return self.mode_stats.copy()
    
    def get_noise_ratio_actual(self) -> float:
        if self.frame_counter == 0:
            return 0.0
        return self.noise_frames_count / self.frame_counter
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'frame': self.frame_counter,
            'state': self.state,
            'current_mode': self.current_mode,
            'selected_modes': self.selected_modes,
            'completed_cycles': self.completed_cycles,
            'noise_frames': self.noise_frames_count,
            'target_noise_frames': self.target_noise_frames
        }
    
    def reset(self):
        """重置噪声器状态"""
        self.frame_counter = 0
        self.noise_frames_count = 0
        self.completed_cycles = 0
        self.mode_stats = {mode: 0 for mode in self.noise_modes}
        self.state = 'idle'
        self.jitter_values = []
        self.jitter_index = 0
        self._init_segment()
