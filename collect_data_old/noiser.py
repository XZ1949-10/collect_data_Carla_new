"""
噪声生成模块（动态规划版本）

设计逻辑：
1. segment开始时，按 probability_percent 选择3个噪声模式，随机排序
2. 动态执行：随机选取起始帧，执行完整周期（噪声+衰减+空闲）
3. 一个周期完成后才能开始下一个
4. 如果剩余帧数不足以完成下一个周期，则停止

配置参数说明：
- noise_ratio: 噪声帧占比 (0-1)
- max_steer_offset: 最大转向偏移 (0-1)
- max_throttle_offset: 最大油门偏移 (0-1)
- strength_percent: 强度百分比 (0-100)
- probability_percent: 模式被选中的概率权重
"""

import random
import math


# 默认噪声模式配置
DEFAULT_NOISE_MODES = {
    'impulse': {
        'noise_frames': [6, 12],      # 噪声阶段帧数范围
        'decay_frames': [4, 8],       # 衰减阶段帧数范围
        'idle_frames': [5, 15],       # 空闲阶段帧数范围
        'strength_percent': 100,
        'probability_percent': 25,
    },
    'smooth': {
        'noise_frames': [15, 25],
        'decay_frames': [8, 15],
        'idle_frames': [5, 15],
        'strength_percent': 80,
        'probability_percent': 35,
    },
    'drift': {
        'noise_frames': [20, 35],
        'decay_frames': [10, 20],
        'idle_frames': [5, 15],
        'strength_percent': 40,
        'probability_percent': 20,
    },
    'jitter': {
        'noise_frames': [10, 20],
        'decay_frames': [5, 10],
        'idle_frames': [5, 15],
        'strength_percent': 50,
        'probability_percent': 20,
    }
}


class Noiser(object):
    """
    动态规划噪声生成器
    
    工作原理：
    1. reset()时选择3个噪声模式并随机排序
    2. 运行时动态决定每个噪声事件的起始帧和持续时间
    3. 必须完成一个完整周期才能开始下一个
    """

    def __init__(self, noise_type, max_offset=0.35, fps=20, mode_config=None, 
                 noise_ratio=0.4, segment_frames=200):
        """
        初始化噪声生成器
        
        Args:
            noise_type (str): 'Spike'(转向), 'Throttle'(油门), 'None'(无)
            max_offset (float): 最大偏移量 (0-1)
            fps (int): 帧率
            mode_config (dict): 噪声模式配置
            noise_ratio (float): 噪声帧占比 (0-1)
            segment_frames (int): segment大小
        """
        self.noise_type = noise_type
        self.fps = fps
        self.max_offset = max(0.05, min(1.0, max_offset))
        self.noise_ratio = max(0.0, min(0.95, noise_ratio))
        self.segment_frames = segment_frames
        
        # 解析配置
        self.noise_modes = self._parse_mode_config(mode_config)
        
        # 计算目标噪声帧数
        self.target_noise_frames = int(segment_frames * noise_ratio)
        
        # 状态变量
        self.frame_counter = 0
        self.selected_modes = []      # 预选的3个模式（随机排序后）
        self.current_mode_idx = 0     # 当前执行到第几个模式
        
        # 当前噪声事件状态
        self.current_mode = None
        self.mode_strength = 1.0
        self.noise_direction = 1
        
        self.noise_start_frame = -1
        self.noise_frames = 0         # 噪声阶段帧数
        self.decay_frames = 0         # 衰减阶段帧数
        self.idle_frames = 0          # 空闲阶段帧数
        self.cycle_end_frame = -1     # 当前周期结束帧
        
        # 状态：idle(空闲等待) -> noise(噪声) -> decay(衰减) -> idle_after(周期内空闲)
        self.state = 'idle'
        
        # jitter专用
        self.jitter_values = []
        self.jitter_index = 0
        
        # 统计
        self.mode_stats = {mode: 0 for mode in self.noise_modes}
        self.noise_frames_count = 0
        self.completed_cycles = 0
        
        # 初始化
        self._init_segment()
    
    def _parse_mode_config(self, config):
        """解析配置"""
        if not config:
            return DEFAULT_NOISE_MODES.copy()
        
        parsed = {}
        for name, cfg in config.items():
            if name.startswith('_') or not isinstance(cfg, dict):
                continue
            
            # 获取默认值
            default = DEFAULT_NOISE_MODES.get(name, {})
            
            parsed[name] = {
                'noise_frames': cfg.get('noise_frames', default.get('noise_frames', [10, 20])),
                'decay_frames': cfg.get('decay_frames', default.get('decay_frames', [5, 10])),
                'idle_frames': cfg.get('idle_frames', default.get('idle_frames', [5, 15])),
                'strength_percent': cfg.get('strength_percent', default.get('strength_percent', 100)),
                'probability_percent': cfg.get('probability_percent', default.get('probability_percent', 25)),
            }
        
        return parsed if parsed else DEFAULT_NOISE_MODES.copy()

    def _select_modes(self, count=3):
        """按probability_percent加权随机选择模式，然后随机排序"""
        modes = list(self.noise_modes.keys())
        weights = [self.noise_modes[m].get('probability_percent', 25) for m in modes]
        
        selected = []
        for _ in range(count):
            mode = random.choices(modes, weights=weights, k=1)[0]
            selected.append(mode)
        
        # 随机打乱顺序
        random.shuffle(selected)
        return selected
    
    def _init_segment(self):
        """初始化segment，选择模式"""
        self.selected_modes = self._select_modes(3)
        self.current_mode_idx = 0
        self.state = 'idle'
        self.noise_start_frame = -1
        self.cycle_end_frame = -1
        
        # 随机选择第一个噪声的起始帧（在前1/4的范围内）
        max_start = max(1, self.segment_frames // 4)
        self._next_trigger_frame = random.randint(1, max_start)
    
    def _calculate_cycle_params(self):
        """
        计算当前周期的参数（噪声帧数、衰减帧数、空闲帧数）
        
        使用配置文件中的 noise_frames 和 decay_frames 范围
        """
        remaining_frames = self.segment_frames - self.frame_counter
        remaining_noise_needed = self.target_noise_frames - self.noise_frames_count
        remaining_modes = len(self.selected_modes) - self.current_mode_idx
        
        if remaining_modes <= 0 or remaining_noise_needed <= 0:
            return 0, 0, 0
        
        # 获取当前模式的配置
        mode = self.selected_modes[self.current_mode_idx]
        config = self.noise_modes.get(mode, DEFAULT_NOISE_MODES.get(mode, {}))
        
        # 从配置中获取帧数范围
        noise_range = config.get('noise_frames', [10, 20])
        decay_range = config.get('decay_frames', [5, 10])
        idle_range = config.get('idle_frames', [5, 15])
        
        # 在范围内随机选择
        noise_frames = random.randint(noise_range[0], noise_range[1])
        decay_frames = random.randint(decay_range[0], decay_range[1])
        idle_frames = random.randint(idle_range[0], idle_range[1])
        
        # 检查是否超过剩余需要的噪声帧数
        total_noise_decay = noise_frames + decay_frames
        if total_noise_decay > remaining_noise_needed:
            # 按比例缩减
            scale = remaining_noise_needed / total_noise_decay
            noise_frames = max(noise_range[0], int(noise_frames * scale))
            decay_frames = max(decay_range[0], int(decay_frames * scale))
        
        # 检查是否超过剩余帧数的限制（包括空闲）
        total_cycle = noise_frames + decay_frames + idle_frames
        max_allowed = int(remaining_frames * 0.8)  # 最多占用80%的剩余帧
        if total_cycle > max_allowed:
            scale = max_allowed / total_cycle
            noise_frames = max(noise_range[0], int(noise_frames * scale))
            decay_frames = max(decay_range[0], int(decay_frames * scale))
            idle_frames = max(idle_range[0], int(idle_frames * scale))
        
        return noise_frames, decay_frames, idle_frames
    
    def _start_noise_cycle(self):
        """开始一个新的噪声周期"""
        if self.current_mode_idx >= len(self.selected_modes):
            return False
        
        # 检查是否还有足够的帧数
        remaining = self.segment_frames - self.frame_counter
        if remaining < 15:  # 至少需要15帧才能完成一个有意义的周期
            return False
        
        # 计算周期参数
        self.noise_frames, self.decay_frames, self.idle_frames = self._calculate_cycle_params()
        
        if self.noise_frames < 5:  # 噪声帧太少，不值得执行
            return False
        
        # 设置当前模式
        self.current_mode = self.selected_modes[self.current_mode_idx]
        config = self.noise_modes[self.current_mode]
        self.mode_strength = config.get('strength_percent', 100) / 100.0
        self.noise_direction = random.choice([1, -1])
        
        # 记录起始帧
        self.noise_start_frame = self.frame_counter
        self.cycle_end_frame = self.frame_counter + self.noise_frames + self.decay_frames + self.idle_frames
        
        # 生成jitter序列
        if self.current_mode == 'jitter':
            self._generate_jitter_sequence()
        
        # 更新状态
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
            
            # 衰减阶段逐渐减小
            if i >= self.noise_frames:
                decay_progress = (i - self.noise_frames) / max(1, self.decay_frames)
                current_value *= (1.0 - decay_progress)
            
            self.jitter_values.append(current_value)
    
    def _update_state(self):
        """更新状态机"""
        if self.state == 'idle':
            # 检查是否到达触发帧
            if self.frame_counter >= self._next_trigger_frame:
                if self._start_noise_cycle():
                    pass  # 成功开始新周期
                else:
                    # 无法开始新周期，保持idle
                    pass
        
        elif self.state == 'noise':
            # 检查噪声阶段是否结束
            elapsed = self.frame_counter - self.noise_start_frame
            if elapsed >= self.noise_frames:
                self.state = 'decay'
        
        elif self.state == 'decay':
            # 检查衰减阶段是否结束
            elapsed = self.frame_counter - self.noise_start_frame - self.noise_frames
            if elapsed >= self.decay_frames:
                self.state = 'idle_after'
        
        elif self.state == 'idle_after':
            # 检查周期内空闲是否结束
            if self.frame_counter >= self.cycle_end_frame:
                # 当前周期完成
                self.completed_cycles += 1
                self.current_mode_idx += 1
                self.state = 'idle'
                
                # 检查是否需要重新选择模式（3个模式执行完了）
                if self.current_mode_idx >= len(self.selected_modes):
                    # 检查是否还有剩余噪声帧数需要执行
                    remaining_noise_needed = self.target_noise_frames - self.noise_frames_count
                    if remaining_noise_needed > 10:
                        # 重新选择3个模式
                        self.selected_modes = self._select_modes(3)
                        self.current_mode_idx = 0
                
                # 设置下一个触发帧
                remaining = self.segment_frames - self.frame_counter
                if remaining > 20 and self.current_mode_idx < len(self.selected_modes):
                    # 立即触发下一个（空闲已经在idle_frames中处理了）
                    self._next_trigger_frame = self.frame_counter + 1
                else:
                    self._next_trigger_frame = self.segment_frames + 1  # 不再触发
    
    def _calculate_noise_value(self):
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

    def _calc_impulse(self, max_noise, elapsed_noise, elapsed_decay):
        """脉冲噪声"""
        if self.state == 'noise':
            progress = min(1.0, elapsed_noise / max(1, self.noise_frames))
            return max_noise * (1.0 - math.exp(-5.0 * progress)) * self.noise_direction
        elif self.state == 'decay':
            progress = min(1.0, elapsed_decay / max(1, self.decay_frames))
            return max_noise * math.exp(-5.0 * progress) * self.noise_direction
        return 0.0
    
    def _calc_smooth(self, max_noise, elapsed_noise, elapsed_decay):
        """平滑噪声"""
        if self.state == 'noise':
            progress = min(1.0, elapsed_noise / max(1, self.noise_frames))
            smooth = progress * progress * (3.0 - 2.0 * progress)
            return max_noise * smooth * self.noise_direction
        elif self.state == 'decay':
            progress = min(1.0, elapsed_decay / max(1, self.decay_frames))
            smooth = 1.0 - progress * progress * (3.0 - 2.0 * progress)
            return max_noise * smooth * self.noise_direction
        return 0.0
    
    def _calc_drift(self, max_noise, elapsed_noise, elapsed_decay):
        """漂移噪声"""
        if self.state == 'noise':
            progress = min(1.0, elapsed_noise / max(1, self.noise_frames))
            return max_noise * math.sin(progress * math.pi / 2) * self.noise_direction
        elif self.state == 'decay':
            progress = min(1.0, elapsed_decay / max(1, self.decay_frames))
            return max_noise * math.cos(progress * math.pi / 2) * self.noise_direction
        return 0.0
    
    def _calc_jitter(self, max_noise):
        """抖动噪声"""
        if self.jitter_index < len(self.jitter_values):
            value = max_noise * self.jitter_values[self.jitter_index]
            self.jitter_index += 1
            return value
        return 0.0

    def compute_noise(self, action, speed):
        """
        计算并应用噪声到控制信号（每帧调用一次）
        
        Returns:
            tuple: (noisy_action, is_recovering, is_noise_active)
        """
        self.frame_counter += 1
        
        if self.noise_type == 'None':
            return action, False, False
        
        # 更新状态机
        self._update_state()
        
        # 统计噪声帧
        is_noisy = self.state in ['noise', 'decay']
        if is_noisy:
            self.noise_frames_count += 1
        
        # 计算噪声值
        noise_value = self._calculate_noise_value()
        
        is_recovering = (self.state == 'decay')
        is_noise_active = (self.state == 'noise')
        
        if self.noise_type == 'Spike':
            if is_noisy:
                noisy_steer = action.steer + noise_value
                action.steer = max(-1.0, min(1.0, noisy_steer))
            return action, is_recovering, is_noise_active
        
        elif self.noise_type == 'Throttle':
            if is_noisy:
                if noise_value > 0:
                    if action.brake > 0:
                        brake_reduction = min(action.brake, noise_value)
                        action.brake = max(0.0, action.brake - brake_reduction)
                        remaining = noise_value - brake_reduction
                        if remaining > 0:
                            action.throttle = max(0.0, min(1.0, action.throttle + remaining))
                    else:
                        action.throttle = max(0.0, min(1.0, action.throttle + noise_value))
                else:
                    abs_noise = abs(noise_value)
                    if action.throttle > 0:
                        throttle_reduction = min(action.throttle, abs_noise)
                        action.throttle = max(0.0, action.throttle - throttle_reduction)
                        remaining = abs_noise - throttle_reduction
                        if remaining > 0:
                            action.brake = max(0.0, min(1.0, action.brake + remaining))
                    else:
                        action.brake = max(0.0, min(1.0, action.brake + abs_noise))
            return action, is_recovering, is_noise_active
        
        return action, False, False
    
    def get_current_mode(self):
        """获取当前噪声模式"""
        if self.state in ['noise', 'decay']:
            return self.current_mode
        return None
    
    @property
    def noise_being_set(self):
        """兼容属性：是否正在噪声阶段"""
        return self.state == 'noise'
    
    @property
    def remove_noise(self):
        """兼容属性：是否正在衰减阶段"""
        return self.state == 'decay'
    
    def get_mode_stats(self):
        """获取各模式使用统计"""
        return self.mode_stats.copy()
    
    def get_noise_ratio_actual(self):
        """获取实际噪声占比"""
        if self.frame_counter == 0:
            return 0.0
        return self.noise_frames_count / self.frame_counter
    
    def get_status(self):
        """获取当前状态信息"""
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
        """重置噪声器状态（新segment开始时调用）"""
        self.frame_counter = 0
        self.noise_frames_count = 0
        self.completed_cycles = 0
        self.mode_stats = {mode: 0 for mode in self.noise_modes}
        self.state = 'idle'
        self.jitter_values = []
        self.jitter_index = 0
        self._init_segment()


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    class Control:
        def __init__(self):
            self.steer = 0.0
            self.throttle = 0.0
            self.brake = 0.0

    # 测试参数
    noise_ratio = 0.4
    max_steer_offset = 0.35
    segment_frames = 200
    fps = 20
    
    print(f"\n{'='*60}")
    print(f"动态规划噪声测试")
    print(f"{'='*60}")
    print(f"  noise_ratio = {noise_ratio} ({noise_ratio*100:.0f}%帧有噪声)")
    print(f"  max_steer_offset = {max_steer_offset}")
    print(f"  segment_frames = {segment_frames}")
    print(f"  目标噪声帧数 = {int(segment_frames * noise_ratio)}")
    print(f"{'='*60}\n")
    
    # 测试多个segment
    all_noise_input = []
    all_mode_labels = []
    all_states = []
    segment_results = []
    
    for seg in range(5):
        noiser = Noiser('Spike', max_offset=max_steer_offset, fps=fps, 
                       noise_ratio=noise_ratio, segment_frames=segment_frames)
        
        print(f"Segment {seg+1}: 预选模式 = {noiser.selected_modes}")
        
        noise_input = []
        mode_labels = []
        states = []
        
        for i in range(segment_frames):
            action = Control()
            action.steer = 0.0
            
            noisy_action, _, _ = noiser.compute_noise(action, speed=20)
            noise_input.append(noisy_action.steer)
            mode_labels.append(noiser.get_current_mode() or '')
            states.append(noiser.state)
        
        actual_ratio = noiser.get_noise_ratio_actual()
        segment_results.append({
            'segment': seg + 1,
            'selected_modes': noiser.selected_modes.copy(),
            'completed_cycles': noiser.completed_cycles,
            'actual_ratio': actual_ratio,
            'noise_frames': noiser.noise_frames_count,
            'mode_stats': noiser.get_mode_stats()
        })
        
        print(f"  完成周期: {noiser.completed_cycles}/3, "
              f"噪声帧: {noiser.noise_frames_count}/{segment_frames}, "
              f"占比: {actual_ratio*100:.1f}%")
        print(f"  模式统计: {noiser.get_mode_stats()}\n")
        
        all_noise_input.extend(noise_input)
        all_mode_labels.extend(mode_labels)
        all_states.extend(states)
    
    # 总体统计
    total_noise = sum(r['noise_frames'] for r in segment_results)
    total_frames = len(segment_results) * segment_frames
    overall_ratio = total_noise / total_frames
    
    print(f"{'='*60}")
    print(f"总体统计: 噪声帧={total_noise}/{total_frames}, 占比={overall_ratio*100:.1f}%")
    print(f"目标占比: {noise_ratio*100:.0f}%")
    print(f"{'='*60}")

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # 图1: 噪声值
    axes[0].plot(range(len(all_noise_input)), all_noise_input, 'r-', linewidth=1)
    axes[0].axhline(y=0, color='g', linestyle='-', alpha=0.5, label='专家输入')
    axes[0].set_ylabel('转向值')
    axes[0].set_title(f'动态规划噪声效果 (目标ratio={noise_ratio}, 实际={overall_ratio:.1%})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.6, 0.6)
    axes[0].axhline(y=max_steer_offset, color='orange', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-max_steer_offset, color='orange', linestyle='--', alpha=0.5)
    
    # 标记segment边界
    for i in range(1, 5):
        axes[0].axvline(x=i*segment_frames, color='gray', linestyle=':', alpha=0.5)
    
    # 图2: 噪声模式
    colors = {'impulse': 'red', 'smooth': 'blue', 'drift': 'green', 'jitter': 'orange'}
    current_mode = None
    start_idx = 0
    
    for i, mode in enumerate(all_mode_labels):
        if mode != current_mode:
            if current_mode and current_mode in colors:
                axes[1].axvspan(start_idx, i, alpha=0.3, color=colors[current_mode], label=current_mode)
            current_mode = mode
            start_idx = i
    if current_mode and current_mode in colors:
        axes[1].axvspan(start_idx, len(all_mode_labels), alpha=0.3, color=colors[current_mode])
    
    axes[1].set_ylabel('噪声模式')
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys())
    
    # 图3: 状态
    state_colors = {'idle': 'white', 'noise': 'red', 'decay': 'orange', 'idle_after': 'lightgray'}
    current_state = None
    start_idx = 0
    
    for i, state in enumerate(all_states):
        if state != current_state:
            if current_state and current_state in state_colors:
                axes[2].axvspan(start_idx, i, alpha=0.3, color=state_colors[current_state], label=current_state)
            current_state = state
            start_idx = i
    if current_state and current_state in state_colors:
        axes[2].axvspan(start_idx, len(all_states), alpha=0.3, color=state_colors[current_state])
    
    axes[2].set_xlabel('帧')
    axes[2].set_ylabel('状态')
    handles, labels = axes[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[2].legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()
