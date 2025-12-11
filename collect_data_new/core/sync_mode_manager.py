#!/usr/bin/env python
# coding=utf-8
"""
CARLA 同步模式管理器 v2.0

统一管理 CARLA 的同步/异步模式切换，避免状态不一致导致的卡死问题。

v2.0 改进：
- 增强状态验证：每次切换后强制验证服务器实际状态
- 主动预防：在关键操作前主动确保状态一致
- 统一资源管理：CollectorLifecycleManager 管理完整生命周期
- 移除被动检测：不再需要在收集循环中检测低速问题

设计原则：
1. 单一职责：只负责模式切换和状态追踪
2. 状态一致性：确保 Python 客户端和 CARLA 服务器状态同步
3. 安全切换：模式切换前后有足够的等待时间
4. 上下文管理：支持 with 语句自动恢复模式
5. 主动验证：关键操作前主动验证状态，而非被动检测

核心改进点：
┌─────────────────────────────────────────────────────────────────────┐
│  问题根源：缓存状态与服务器实际状态不一致                              │
│                                                                     │
│  旧方案（被动）：                                                     │
│    收集循环 → 检测低速 → 重置同步模式 → 继续收集                       │
│                                                                     │
│  新方案（主动）：                                                     │
│    切换模式 → 强制验证 → 预热tick → 验证tick成功 → 开始操作            │
└─────────────────────────────────────────────────────────────────────┘
"""

import time
import threading
from enum import Enum, auto
from contextlib import contextmanager
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass, field

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class SyncMode(Enum):
    """同步模式枚举"""
    ASYNC = auto()   # 异步模式
    SYNC = auto()    # 同步模式
    UNKNOWN = auto() # 未知状态


@dataclass
class SyncModeConfig:
    """同步模式配置"""
    simulation_fps: int = 20                    # 模拟帧率
    mode_switch_wait: float = 0.5               # 模式切换等待时间（秒）
    post_switch_stabilize: float = 0.3          # 切换后稳定时间（秒）
    max_switch_retries: int = 3                 # 最大重试次数
    verify_after_switch: bool = True            # 切换后是否验证 - 默认开启！
    tick_timeout: float = 5.0                   # tick 超时时间（秒）
    tick_retry_count: int = 3                   # tick 失败重试次数
    tick_retry_delay: float = 0.1               # tick 重试间隔（秒）
    warmup_ticks: int = 10                      # 预热 tick 次数
    stabilize_ticks: int = 15                   # 稳定 tick 次数
    # v2.0 新增配置
    force_verify: bool = True                   # 强制验证模式（每次切换都验证）
    auto_recover: bool = True                   # 自动恢复（验证失败时自动重试）
    max_recover_attempts: int = 3               # 最大恢复尝试次数
    verify_with_tick: bool = True               # 使用 tick 验证同步模式是否真正生效
    
    @property
    def fixed_delta_seconds(self) -> float:
        """固定时间步长"""
        return 1.0 / self.simulation_fps


class SyncModeManager:
    """
    CARLA 同步模式管理器 v2.0
    
    功能：
    - 统一管理同步/异步模式切换
    - 追踪当前模式状态
    - 提供安全的模式切换方法
    - 支持上下文管理器
    - 【v2.0】主动验证和自动恢复机制
    
    使用示例：
        # 基本使用（推荐使用 ensure_sync_mode）
        sync_mgr = SyncModeManager(world)
        sync_mgr.ensure_sync_mode()  # 确保同步模式，自动验证
        # ... 数据收集 ...
        sync_mgr.ensure_async_mode()  # 确保异步模式
        
        # 上下文管理器
        with sync_mgr.sync_context():
            # 在同步模式下执行
            pass
        # 自动恢复原模式
        
        # 安全销毁资源
        with sync_mgr.async_context():
            sensor.destroy()
            vehicle.destroy()
    """
    
    def __init__(self, world, config: Optional[SyncModeConfig] = None):
        """
        初始化同步模式管理器
        
        参数:
            world: CARLA world 对象
            config: 同步模式配置
        """
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.world = world
        self.config = config or SyncModeConfig()
        
        # 状态追踪
        self._current_mode = SyncMode.UNKNOWN
        self._original_mode: Optional[SyncMode] = None
        self._lock = threading.Lock()
        self._last_successful_tick_time: float = 0  # 上次成功 tick 的时间
        self._consecutive_tick_failures: int = 0    # 连续 tick 失败次数
        
        # 回调
        self._on_mode_change: Optional[Callable[[SyncMode, SyncMode], None]] = None
        
        # 初始化时读取当前模式
        self._refresh_mode_state()
    
    @property
    def current_mode(self) -> SyncMode:
        """获取当前模式"""
        with self._lock:
            return self._current_mode
    
    @property
    def is_sync(self) -> bool:
        """是否为同步模式"""
        return self.current_mode == SyncMode.SYNC
    
    @property
    def is_async(self) -> bool:
        """是否为异步模式"""
        return self.current_mode == SyncMode.ASYNC
    
    def set_mode_change_callback(self, callback: Callable[[SyncMode, SyncMode], None]):
        """设置模式变化回调"""
        self._on_mode_change = callback
    
    def _refresh_mode_state(self) -> SyncMode:
        """从服务器刷新模式状态"""
        try:
            settings = self.world.get_settings()
            with self._lock:
                self._current_mode = SyncMode.SYNC if settings.synchronous_mode else SyncMode.ASYNC
            return self._current_mode
        except Exception as e:
            print(f"⚠️ 无法获取同步模式状态: {e}")
            with self._lock:
                self._current_mode = SyncMode.UNKNOWN
            return SyncMode.UNKNOWN
    
    def _apply_settings(self, sync_mode: bool, wait_time: float = None) -> bool:
        """
        应用同步模式设置
        
        参数:
            sync_mode: True=同步模式, False=异步模式
            wait_time: 等待时间，None则使用配置默认值
        """
        wait_time = wait_time or self.config.mode_switch_wait
        
        try:
            settings = self.world.get_settings()
            
            # 检查是否需要切换
            if settings.synchronous_mode == sync_mode:
                with self._lock:
                    self._current_mode = SyncMode.SYNC if sync_mode else SyncMode.ASYNC
                return True
            
            # 记录旧模式
            old_mode = self._current_mode
            
            # 应用新设置
            settings.synchronous_mode = sync_mode
            if sync_mode:
                settings.fixed_delta_seconds = self.config.fixed_delta_seconds
            else:
                settings.fixed_delta_seconds = None
            
            self.world.apply_settings(settings)
            
            # 等待设置生效
            time.sleep(wait_time)
            
            # 稳定期
            if self.config.post_switch_stabilize > 0:
                time.sleep(self.config.post_switch_stabilize)
            
            # 更新状态
            new_mode = SyncMode.SYNC if sync_mode else SyncMode.ASYNC
            with self._lock:
                self._current_mode = new_mode
            
            # 验证切换
            if self.config.verify_after_switch:
                actual_mode = self._refresh_mode_state()
                if actual_mode != new_mode:
                    print(f"⚠️ 模式切换验证失败: 期望 {new_mode}, 实际 {actual_mode}")
                    return False
            
            # 触发回调
            if self._on_mode_change:
                try:
                    self._on_mode_change(old_mode, new_mode)
                except:
                    pass
            
            return True
            
        except Exception as e:
            print(f"❌ 模式切换失败: {e}")
            return False
    
    def enable_sync_mode(self, wait_time: float = None, verbose: bool = False, 
                         force_refresh: bool = False) -> bool:
        """
        启用同步模式
        
        适用场景：
        - 数据收集开始前
        - 需要精确控制每帧时
        - 传感器初始化后
        
        参数:
            wait_time: 等待时间（秒）
            verbose: 是否打印详细信息
            force_refresh: 是否强制从服务器刷新状态（解决缓存不一致问题）
            
        返回:
            bool: 是否成功
        """
        # 关键修复：可选强制刷新状态，解决缓存与服务器不一致的问题
        if force_refresh:
            self._refresh_mode_state()
        
        # 如果已经是同步模式，直接返回
        if self.is_sync:
            return True
        if verbose:
            print("🔄 切换到同步模式...")
        success = self._apply_settings(True, wait_time)
        if success and verbose:
            print(f"✅ 同步模式已启用 (FPS: {self.config.simulation_fps})")
        return success
    
    def enable_async_mode(self, wait_time: float = None, verbose: bool = False,
                          force_refresh: bool = False) -> bool:
        """
        启用异步模式
        
        适用场景：
        - 销毁传感器/车辆前（必须！）
        - 生成大量 NPC 时
        - 可视化调试时
        
        参数:
            wait_time: 等待时间（秒）
            verbose: 是否打印详细信息
            force_refresh: 是否强制从服务器刷新状态
            
        返回:
            bool: 是否成功
        """
        # 关键修复：可选强制刷新状态
        if force_refresh:
            self._refresh_mode_state()
        
        # 如果已经是异步模式，直接返回
        if self.is_async:
            return True
        if verbose:
            print("🔄 切换到异步模式...")
        success = self._apply_settings(False, wait_time)
        if success and verbose:
            print("✅ 异步模式已启用")
        return success
    
    def reset_sync_mode(self, verbose: bool = True) -> bool:
        """
        重置同步模式（先异步再同步）
        
        用于解决同步模式状态不一致的问题。
        当怀疑模式状态不一致时调用此方法。
        
        关键修复：强制刷新状态，确保缓存与服务器一致。
        
        返回:
            bool: 是否成功
        """
        if verbose:
            print("🔄 重置同步模式...")
        
        # 关键：先从服务器刷新实际状态
        self._refresh_mode_state()
        
        # 强制切换到异步（不依赖缓存状态）
        if not self._apply_settings(False):
            print("⚠️ 切换到异步模式失败")
            return False
        
        # 额外等待确保服务器处理完成
        time.sleep(0.3)
        
        # 再切换回同步
        if not self._apply_settings(True):
            print("⚠️ 切换到同步模式失败")
            return False
        
        # 额外等待确保稳定
        time.sleep(0.3)
        
        if verbose:
            print("✅ 同步模式重置完成")
        return True
    
    def ensure_sync_mode(self, warmup: bool = True, verify: bool = True) -> bool:
        """
        【v2.0 核心方法】确保同步模式已启用并验证生效
        
        这是推荐的启用同步模式方法，会：
        1. 强制从服务器刷新状态
        2. 如果不是同步模式，切换到同步模式
        3. 执行验证 tick 确保模式真正生效
        4. 如果验证失败，自动重试
        
        参数:
            warmup: 是否执行预热 tick
            verify: 是否验证模式生效
            
        返回:
            bool: 是否成功
        """
        # 1. 强制刷新状态
        self._refresh_mode_state()
        
        # 2. 切换到同步模式
        if not self.is_sync:
            if not self._apply_settings(True):
                print("⚠️ 切换到同步模式失败")
                return False
        
        # 3. 验证模式生效（通过执行 tick）
        if verify and self.config.verify_with_tick:
            if not self._verify_sync_mode_with_tick():
                # 验证失败，尝试恢复
                if self.config.auto_recover:
                    return self._auto_recover_sync_mode()
                return False
        
        # 4. 预热 tick
        if warmup:
            success_count = self.warmup_tick()
            if success_count < self.config.warmup_ticks // 2:
                print(f"⚠️ 预热不完整 ({success_count}/{self.config.warmup_ticks})")
                if self.config.auto_recover:
                    return self._auto_recover_sync_mode()
                return False
        
        return True
    
    def ensure_async_mode(self, wait: bool = True) -> bool:
        """
        【v2.0 核心方法】确保异步模式已启用
        
        这是推荐的启用异步模式方法，会：
        1. 强制从服务器刷新状态
        2. 如果不是异步模式，切换到异步模式
        3. 等待模式生效
        
        参数:
            wait: 是否等待模式生效
            
        返回:
            bool: 是否成功
        """
        # 1. 强制刷新状态
        self._refresh_mode_state()
        
        # 2. 切换到异步模式
        if not self.is_async:
            if not self._apply_settings(False):
                print("⚠️ 切换到异步模式失败")
                return False
        
        # 3. 等待模式生效
        if wait:
            time.sleep(self.config.post_switch_stabilize)
        
        return True
    
    def _verify_sync_mode_with_tick(self) -> bool:
        """
        通过执行 tick 验证同步模式是否真正生效
        
        返回:
            bool: 验证是否通过
        """
        try:
            # 尝试执行一次 tick，使用较短超时
            self.world.tick(2.0)
            self._last_successful_tick_time = time.time()
            self._consecutive_tick_failures = 0
            return True
        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'timeout' in error_msg or 'time-out' in error_msg:
                print(f"⚠️ 同步模式验证失败（tick 超时）: {e}")
            else:
                print(f"⚠️ 同步模式验证失败: {e}")
            self._consecutive_tick_failures += 1
            return False
        except Exception as e:
            print(f"⚠️ 同步模式验证异常: {e}")
            self._consecutive_tick_failures += 1
            return False
    
    def _auto_recover_sync_mode(self) -> bool:
        """
        自动恢复同步模式
        
        当检测到同步模式不工作时，尝试自动恢复。
        
        返回:
            bool: 是否恢复成功
        """
        print("🔧 尝试自动恢复同步模式...")
        
        for attempt in range(self.config.max_recover_attempts):
            print(f"  尝试 {attempt + 1}/{self.config.max_recover_attempts}...")
            
            # 完整重置
            if self.reset_sync_mode(verbose=False):
                # 验证重置是否成功
                if self._verify_sync_mode_with_tick():
                    print(f"✅ 同步模式恢复成功（尝试 {attempt + 1}）")
                    return True
            
            # 等待后重试
            time.sleep(0.5)
        
        print("❌ 同步模式恢复失败")
        return False
    
    def safe_tick(self, timeout: float = None, auto_recover: bool = True) -> bool:
        """
        安全的 tick 调用（带超时、重试和自动恢复）
        
        只在同步模式下调用 world.tick()，
        异步模式下等待一帧时间。
        
        【v2.0 改进】：连续失败时自动触发恢复机制
        
        参数:
            timeout: 超时时间（秒），None 则使用配置默认值
            auto_recover: 是否在连续失败时自动恢复
            
        返回:
            bool: 是否成功
        """
        timeout = timeout or self.config.tick_timeout
        
        # 重要：先从服务器刷新实际状态，避免状态不一致
        # 只在首次或状态未知时刷新，避免每次都查询
        if self._current_mode == SyncMode.UNKNOWN:
            self._refresh_mode_state()
        
        if not self.is_sync:
            # 异步模式下等待一帧时间
            time.sleep(self.config.fixed_delta_seconds)
            return True
        
        for attempt in range(self.config.tick_retry_count + 1):
            try:
                # 使用较短的超时，避免长时间阻塞
                self.world.tick(timeout)
                # 成功，重置失败计数
                self._last_successful_tick_time = time.time()
                self._consecutive_tick_failures = 0
                return True
            except RuntimeError as e:
                error_msg = str(e).lower()
                self._consecutive_tick_failures += 1
                
                # tick 超时会抛出 RuntimeError
                if attempt < self.config.tick_retry_count:
                    # 检查是否是超时错误
                    if 'timeout' in error_msg or 'time-out' in error_msg:
                        # 重试前刷新状态，可能服务器已经切换到异步模式
                        self._refresh_mode_state()
                        if not self.is_sync:
                            # 服务器已经是异步模式，不需要 tick
                            time.sleep(self.config.fixed_delta_seconds)
                            return True
                        time.sleep(self.config.tick_retry_delay)
                    else:
                        # 其他 RuntimeError，可能是更严重的问题
                        print(f"⚠️ tick 出错 (尝试 {attempt + 1}): {e}")
                        time.sleep(self.config.tick_retry_delay)
                else:
                    # 最后一次失败
                    print(f"⚠️ tick 超时 ({attempt + 1} 次尝试): {e}")
                    
                    # 【v2.0】连续失败超过阈值，触发自动恢复
                    if auto_recover and self._consecutive_tick_failures >= 3:
                        print(f"⚠️ 连续 {self._consecutive_tick_failures} 次 tick 失败，触发自动恢复...")
                        if self._auto_recover_sync_mode():
                            # 恢复成功，再试一次
                            try:
                                self.world.tick(timeout)
                                self._last_successful_tick_time = time.time()
                                self._consecutive_tick_failures = 0
                                return True
                            except:
                                pass
                    return False
            except Exception as e:
                print(f"❌ tick 失败: {e}")
                self._consecutive_tick_failures += 1
                return False
        
        return False
    
    def tick(self, timeout: float = None) -> bool:
        """
        推进一帧模拟（safe_tick 的别名，更简洁的调用方式）
        
        参数:
            timeout: 超时时间（秒）
            
        返回:
            bool: 是否成功
        """
        return self.safe_tick(timeout)
    
    def tick_multiple(self, count: int, timeout: float = None, 
                      delay: float = 0.0, silent: bool = False) -> int:
        """
        执行多次 tick
        
        用于等待物理稳定、传感器初始化等场景。
        
        参数:
            count: tick 次数
            timeout: 每次 tick 的超时时间
            delay: 每次 tick 之间的延迟（秒）
            silent: 是否静默模式（不打印警告）
            
        返回:
            int: 成功执行的 tick 次数
        """
        success_count = 0
        for i in range(count):
            if self.safe_tick(timeout):
                success_count += 1
            else:
                if not silent:
                    print(f"⚠️ tick_multiple: 第 {i + 1}/{count} 次 tick 失败")
                break
            if delay > 0 and i < count - 1:
                time.sleep(delay)
        return success_count
    
    def warmup_tick(self, count: int = None) -> int:
        """
        预热 tick（用于初始化后稳定）
        
        参数:
            count: tick 次数，None 则使用配置默认值
            
        返回:
            int: 成功执行的 tick 次数
        """
        count = count or self.config.warmup_ticks
        return self.tick_multiple(count, delay=0.02, silent=True)
    
    def stabilize_tick(self, count: int = None) -> int:
        """
        稳定 tick（用于车辆/传感器稳定）
        
        参数:
            count: tick 次数，None 则使用配置默认值
            
        返回:
            int: 成功执行的 tick 次数
        """
        count = count or self.config.stabilize_ticks
        return self.tick_multiple(count, delay=0.05, silent=True)
    
    @contextmanager
    def sync_context(self):
        """
        同步模式上下文管理器
        
        使用示例：
            with sync_mgr.sync_context():
                # 在同步模式下执行数据收集
                for _ in range(1000):
                    world.tick()
                    collect_data()
            # 自动恢复原模式
        """
        # 保存原模式
        original = self.current_mode
        
        try:
            # 切换到同步模式
            if not self.is_sync:
                self.enable_sync_mode()
            yield self
        finally:
            # 恢复原模式
            if original == SyncMode.ASYNC:
                self.enable_async_mode()
    
    @contextmanager
    def async_context(self):
        """
        异步模式上下文管理器
        
        使用示例：
            with sync_mgr.async_context():
                # 在异步模式下安全销毁资源
                sensor.stop()
                sensor.destroy()
                vehicle.destroy()
            # 自动恢复原模式
        """
        # 保存原模式
        original = self.current_mode
        
        try:
            # 切换到异步模式
            if not self.is_async:
                self.enable_async_mode()
            yield self
        finally:
            # 恢复原模式
            if original == SyncMode.SYNC:
                self.enable_sync_mode()
    
    def save_original_mode(self):
        """保存当前模式为原始模式（用于后续恢复）"""
        self._original_mode = self.current_mode
    
    def restore_original_mode(self) -> bool:
        """恢复到原始模式"""
        if self._original_mode is None:
            return True
        
        if self._original_mode == SyncMode.SYNC:
            return self.enable_sync_mode()
        elif self._original_mode == SyncMode.ASYNC:
            return self.enable_async_mode()
        return True




class ResourceLifecycleHelper:
    """
    资源生命周期辅助类
    
    封装了资源创建和销毁时的模式切换逻辑，
    确保在正确的模式下执行操作。
    
    使用场景分析：
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                     资源创建流程                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  1. [异步] 生成车辆 (try_spawn_actor)                           │
    │  2. [同步] 等待车辆物理稳定 (多次 tick)                          │
    │  3. [同步] 创建传感器并附加到车辆                                │
    │  4. [同步] 等待传感器初始化 (多次 tick)                          │
    │  5. [同步] 开始数据收集                                         │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                     资源销毁流程                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  1. [同步→异步] 切换到异步模式                                   │
    │  2. [异步] 停止传感器监听 (sensor.stop())                        │
    │  3. [异步] 销毁传感器 (sensor.destroy())                         │
    │  4. [异步] 销毁车辆 (vehicle.destroy())                          │
    │  5. [异步] 等待清理完成                                          │
    │  6. [异步→同步] 如需继续收集，切换回同步模式                       │
    └─────────────────────────────────────────────────────────────────┘
    
    ⚠️ 关键警告：
    - 在同步模式下销毁传感器可能导致死锁！
    - 传感器的回调函数可能正在等待 tick()，而 destroy() 在等待回调完成
    """
    
    def __init__(self, sync_manager: SyncModeManager):
        """
        初始化辅助类
        
        参数:
            sync_manager: 同步模式管理器
        """
        self.sync_mgr = sync_manager
        self.world = sync_manager.world
    
    def spawn_vehicle_safe(self, blueprint, transform, 
                           stabilize_ticks: int = 10) -> 'carla.Actor':
        """
        安全地生成车辆
        
        流程：
        1. 在当前模式下尝试生成
        2. 如果是同步模式，执行多次 tick 等待物理稳定
        
        参数:
            blueprint: 车辆蓝图
            transform: 生成位置
            stabilize_ticks: 稳定所需的 tick 次数
            
        返回:
            carla.Actor: 生成的车辆，失败返回 None
        """
        try:
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            
            if vehicle is None:
                return None
            
            # 等待物理稳定
            if self.sync_mgr.is_sync:
                for _ in range(stabilize_ticks):
                    self.sync_mgr.safe_tick()
                    time.sleep(0.05)
            else:
                time.sleep(1.0)
            
            return vehicle
            
        except Exception as e:
            print(f"❌ 生成车辆失败: {e}")
            return None
    
    def create_sensor_safe(self, blueprint, transform, 
                           attach_to, callback,
                           init_ticks: int = 10) -> 'carla.Actor':
        """
        安全地创建传感器
        
        流程：
        1. 创建传感器并附加到车辆
        2. 注册回调
        3. 在同步模式下执行多次 tick 等待初始化
        
        参数:
            blueprint: 传感器蓝图
            transform: 相对位置
            attach_to: 附加到的 actor
            callback: 数据回调函数
            init_ticks: 初始化所需的 tick 次数
            
        返回:
            carla.Actor: 创建的传感器，失败返回 None
        """
        try:
            sensor = self.world.spawn_actor(
                blueprint, transform, attach_to=attach_to
            )
            
            if sensor is None:
                return None
            
            # 注册回调
            sensor.listen(callback)
            
            # 等待初始化
            if self.sync_mgr.is_sync:
                for _ in range(init_ticks):
                    self.sync_mgr.safe_tick()
                    time.sleep(0.05)
            else:
                time.sleep(1.0)
            
            return sensor
            
        except Exception as e:
            print(f"❌ 创建传感器失败: {e}")
            return None
    
    def destroy_sensor_safe(self, sensor, wait_time: float = 0.3) -> bool:
        """
        安全地销毁传感器
        
        ⚠️ 必须在异步模式下执行！
        
        流程：
        1. 确保在异步模式
        2. 停止传感器监听
        3. 销毁传感器
        4. 等待清理完成
        
        参数:
            sensor: 要销毁的传感器
            wait_time: 销毁后等待时间
            
        返回:
            bool: 是否成功
        """
        if sensor is None:
            return True
        
        # 确保异步模式
        was_sync = self.sync_mgr.is_sync
        if was_sync:
            self.sync_mgr.enable_async_mode()
        
        try:
            try:
                sensor.stop()
            except:
                pass
            
            try:
                sensor.destroy()
            except:
                pass
            
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            print(f"⚠️ 销毁传感器异常: {e}")
            return False
        finally:
            # 恢复原模式
            if was_sync:
                self.sync_mgr.enable_sync_mode()
    
    def destroy_vehicle_safe(self, vehicle, wait_time: float = 0.3) -> bool:
        """
        安全地销毁车辆
        
        流程：
        1. 确保在异步模式
        2. 销毁车辆
        3. 等待清理完成
        
        参数:
            vehicle: 要销毁的车辆
            wait_time: 销毁后等待时间
            
        返回:
            bool: 是否成功
        """
        if vehicle is None:
            return True
        
        # 确保异步模式
        was_sync = self.sync_mgr.is_sync
        if was_sync:
            self.sync_mgr.enable_async_mode()
        
        try:
            vehicle.destroy()
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            print(f"⚠️ 销毁车辆异常: {e}")
            return False
        finally:
            # 恢复原模式
            if was_sync:
                self.sync_mgr.enable_sync_mode()
    
    def destroy_all_safe(self, sensors: list, vehicle,
                         restore_sync: bool = False) -> bool:
        """
        安全地销毁所有资源（优化版本，减少等待时间）
        
        流程：
        1. 切换到异步模式
        2. 批量销毁：传感器 → 车辆
        3. 可选：恢复同步模式
        
        参数:
            sensors: 传感器列表
            vehicle: 车辆
            restore_sync: 是否恢复同步模式
            
        返回:
            bool: 是否全部成功
        """
        # 切换到异步模式
        self.sync_mgr.enable_async_mode()
        
        success = True
        
        # 批量销毁传感器（不单独等待）
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                except:
                    pass
                try:
                    sensor.destroy()
                except:
                    success = False
        
        # 销毁车辆
        if vehicle is not None:
            try:
                vehicle.destroy()
            except:
                success = False
        
        # 只等待一次
        time.sleep(0.3)
        
        # 恢复同步模式
        if restore_sync:
            self.sync_mgr.enable_sync_mode()
        
        return success


class CollectorLifecycleManager:
    """
    【v2.0 新增】数据收集生命周期管理器
    
    统一管理数据收集的完整生命周期，包括：
    - 资源创建（车辆、传感器）
    - 同步模式管理
    - 资源销毁
    - 路线切换
    
    使用此类可以避免手动管理同步模式带来的问题。
    
    使用示例：
        lifecycle = CollectorLifecycleManager(world, blueprint_library)
        
        # 开始新路线
        with lifecycle.route_context() as ctx:
            vehicle = ctx.spawn_vehicle(spawn_point)
            camera = ctx.create_camera(vehicle, callback)
            
            # 数据收集循环
            for frame in range(max_frames):
                ctx.tick()  # 自动处理同步模式
                collect_data()
        # 自动清理资源
    """
    
    def __init__(self, world, blueprint_library, config: Optional[SyncModeConfig] = None):
        """
        初始化生命周期管理器
        
        参数:
            world: CARLA world 对象
            blueprint_library: 蓝图库
            config: 同步模式配置
        """
        self.world = world
        self.blueprint_library = blueprint_library
        self.sync_mgr = SyncModeManager(world, config)
        self.helper = ResourceLifecycleHelper(self.sync_mgr)
        
        # 当前路线的资源
        self._current_vehicle = None
        self._current_sensors: List = []
        self._route_active = False
    
    @contextmanager
    def route_context(self):
        """
        路线上下文管理器
        
        自动处理：
        1. 开始时确保同步模式
        2. 结束时安全清理资源
        3. 异常时也能正确清理
        """
        try:
            # 开始新路线：确保同步模式
            if not self.sync_mgr.ensure_sync_mode():
                raise RuntimeError("无法启用同步模式")
            
            self._route_active = True
            yield self
            
        finally:
            # 结束路线：清理资源
            self._cleanup_route()
            self._route_active = False
    
    def spawn_vehicle(self, transform, vehicle_filter: str = 'vehicle.tesla.model3',
                      stabilize_ticks: int = 15):
        """
        生成车辆
        
        参数:
            transform: 生成位置
            vehicle_filter: 车辆蓝图过滤器
            stabilize_ticks: 稳定所需的 tick 次数
            
        返回:
            carla.Actor: 生成的车辆
        """
        vehicle_bp = self.blueprint_library.filter(vehicle_filter)[0]
        vehicle = self.helper.spawn_vehicle_safe(vehicle_bp, transform, stabilize_ticks)
        
        if vehicle:
            self._current_vehicle = vehicle
        
        return vehicle
    
    def create_camera(self, attach_to, callback, 
                      width: int = 800, height: int = 600, fov: int = 90,
                      location: tuple = (2.0, 0.0, 1.4), rotation: tuple = (0.0, -15.0, 0.0),
                      init_ticks: int = 10):
        """
        创建摄像头
        
        参数:
            attach_to: 附加到的 actor
            callback: 数据回调函数
            width, height, fov: 摄像头参数
            location, rotation: 相对位置和旋转
            init_ticks: 初始化所需的 tick 次数
            
        返回:
            carla.Actor: 创建的摄像头
        """
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        
        camera_transform = carla.Transform(
            carla.Location(x=location[0], y=location[1], z=location[2]),
            carla.Rotation(pitch=rotation[1])
        )
        
        camera = self.helper.create_sensor_safe(
            camera_bp, camera_transform, attach_to, callback, init_ticks
        )
        
        if camera:
            self._current_sensors.append(camera)
        
        return camera
    
    def create_collision_sensor(self, attach_to, callback):
        """
        创建碰撞传感器
        
        参数:
            attach_to: 附加到的 actor
            callback: 碰撞回调函数
            
        返回:
            carla.Actor: 创建的碰撞传感器
        """
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        
        sensor = self.helper.create_sensor_safe(
            collision_bp, collision_transform, attach_to, callback, init_ticks=5
        )
        
        if sensor:
            self._current_sensors.append(sensor)
        
        return sensor
    
    def tick(self, timeout: float = None) -> bool:
        """
        推进一帧模拟
        
        参数:
            timeout: 超时时间
            
        返回:
            bool: 是否成功
        """
        return self.sync_mgr.safe_tick(timeout)
    
    def _cleanup_route(self):
        """清理当前路线的资源"""
        if self._current_sensors or self._current_vehicle:
            self.helper.destroy_all_safe(
                self._current_sensors, 
                self._current_vehicle,
                restore_sync=False
            )
        
        self._current_sensors = []
        self._current_vehicle = None
    
    def prepare_next_route(self) -> bool:
        """
        准备下一条路线
        
        清理当前资源并确保同步模式就绪。
        
        返回:
            bool: 是否准备成功
        """
        # 清理当前资源
        self._cleanup_route()
        
        # 确保同步模式
        return self.sync_mgr.ensure_sync_mode()


# ==================== 使用指南 ====================
"""
## 同步模式使用指南 v2.0

### 1. 推荐方式：使用 CollectorLifecycleManager

```python
lifecycle = CollectorLifecycleManager(world, blueprint_library)

for route in routes:
    with lifecycle.route_context() as ctx:
        # 创建资源
        vehicle = ctx.spawn_vehicle(spawn_point)
        camera = ctx.create_camera(vehicle, on_image)
        collision = ctx.create_collision_sensor(vehicle, on_collision)
        
        # 数据收集循环
        for frame in range(max_frames):
            ctx.tick()  # 自动处理同步模式
            collect_data()
    # 自动清理资源，自动处理模式切换
```

### 2. 使用 ensure_sync_mode（推荐）

```python
sync_mgr = SyncModeManager(world)

# 确保同步模式（自动验证和恢复）
if not sync_mgr.ensure_sync_mode():
    print("无法启用同步模式")
    return

# 数据收集
for frame in range(max_frames):
    sync_mgr.safe_tick()  # 自动处理失败和恢复
    collect_data()

# 清理前确保异步模式
sync_mgr.ensure_async_mode()
```

### 3. 旧方式（仍然支持）

```python
sync_mgr = SyncModeManager(world)
helper = ResourceLifecycleHelper(sync_mgr)

# 创建资源
vehicle = helper.spawn_vehicle_safe(vehicle_bp, spawn_point)
camera = helper.create_sensor_safe(camera_bp, cam_transform, vehicle, on_image)

# 数据收集（同步模式）
with sync_mgr.sync_context():
    for frame in range(max_frames):
        sync_mgr.safe_tick()
        collect_data()

# 清理资源
helper.destroy_all_safe([camera], vehicle)
```

### 4. 常见问题排查

问题：车辆速度一直是 0
原因：同步模式下没有调用 tick() 或模式状态不一致
解决：使用 ensure_sync_mode() 代替 enable_sync_mode()

问题：销毁传感器时卡住
原因：在同步模式下销毁传感器
解决：使用 ensure_async_mode() 或 CollectorLifecycleManager

问题：模式切换后行为异常
原因：模式状态不一致
解决：ensure_sync_mode() 会自动验证和恢复

### 5. v2.0 改进总结

- ensure_sync_mode(): 主动验证同步模式是否真正生效
- ensure_async_mode(): 主动验证异步模式是否真正生效
- safe_tick(): 连续失败时自动触发恢复
- CollectorLifecycleManager: 统一管理整个生命周期
- 移除被动检测：不再需要在收集循环中检测低速问题
"""
