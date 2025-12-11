# CARLA 同步模式使用指南 v2.0

## 1. 概述

CARLA 支持两种运行模式：**同步模式**和**异步模式**。正确使用这两种模式是避免服务器卡死的关键。

### v2.0 改进

- **主动验证**：`ensure_sync_mode()` 会验证模式是否真正生效
- **自动恢复**：`safe_tick()` 连续失败时自动触发恢复机制
- **统一管理**：`CollectorLifecycleManager` 管理完整生命周期
- **移除被动检测**：不再需要在收集循环中检测低速问题

## 2. 模式对比

| 特性 | 同步模式 (Synchronous) | 异步模式 (Asynchronous) |
|------|------------------------|-------------------------|
| 时间推进 | 客户端调用 `world.tick()` | 服务器自动推进 |
| 帧率控制 | `fixed_delta_seconds` 固定 | 服务器决定 |
| 数据完整性 | 每帧都能处理 | 可能丢帧 |
| 适用场景 | 数据收集、训练 | 调试、资源销毁 |

## 3. 推荐使用方式（v2.0）

### 3.1 使用 CollectorLifecycleManager（最推荐）

```python
from collect_data_new.core import CollectorLifecycleManager

lifecycle = CollectorLifecycleManager(world, blueprint_library)

for route in routes:
    with lifecycle.route_context() as ctx:
        # 创建资源（自动处理同步模式）
        vehicle = ctx.spawn_vehicle(spawn_point)
        camera = ctx.create_camera(vehicle, on_image)
        collision = ctx.create_collision_sensor(vehicle, on_collision)
        
        # 数据收集循环
        for frame in range(max_frames):
            ctx.tick()  # 自动处理同步模式和失败恢复
            collect_data()
    # 自动清理资源，自动处理模式切换
```

### 3.2 使用 ensure_sync_mode（推荐）

```python
from collect_data_new.core import SyncModeManager

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

### 3.3 旧方式（仍然支持）

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

## 4. API 参考

### 4.1 SyncModeManager

```python
class SyncModeManager:
    # 核心方法（v2.0 推荐）
    def ensure_sync_mode(warmup=True, verify=True) -> bool
        """确保同步模式已启用并验证生效"""
    
    def ensure_async_mode(wait=True) -> bool
        """确保异步模式已启用"""
    
    def safe_tick(timeout=None, auto_recover=True) -> bool
        """安全的 tick 调用，带自动恢复"""
    
    # 旧方法（仍然支持）
    def enable_sync_mode() -> bool
    def enable_async_mode() -> bool
    def reset_sync_mode() -> bool
    def tick() -> bool
    def warmup_tick(count=None) -> int
    def stabilize_tick(count=None) -> int
    
    # 上下文管理器
    def sync_context()
    def async_context()
```

### 4.2 CollectorLifecycleManager

```python
class CollectorLifecycleManager:
    def route_context()
        """路线上下文管理器，自动处理模式切换和资源清理"""
    
    def spawn_vehicle(transform, vehicle_filter='vehicle.tesla.model3')
        """生成车辆"""
    
    def create_camera(attach_to, callback, width=800, height=600, fov=90)
        """创建摄像头"""
    
    def create_collision_sensor(attach_to, callback)
        """创建碰撞传感器"""
    
    def tick(timeout=None) -> bool
        """推进一帧模拟"""
    
    def prepare_next_route() -> bool
        """准备下一条路线"""
```

## 5. 配置选项

```python
@dataclass
class SyncModeConfig:
    simulation_fps: int = 20           # 模拟帧率
    mode_switch_wait: float = 0.5      # 模式切换等待时间
    tick_timeout: float = 5.0          # tick 超时时间
    tick_retry_count: int = 3          # tick 失败重试次数
    warmup_ticks: int = 10             # 预热 tick 次数
    
    # v2.0 新增
    force_verify: bool = True          # 强制验证模式
    auto_recover: bool = True          # 自动恢复
    max_recover_attempts: int = 3      # 最大恢复尝试次数
    verify_with_tick: bool = True      # 使用 tick 验证同步模式
```

## 6. 常见问题排查

### 问题 1: 车辆速度一直是 0

**原因**: 同步模式未真正生效

**解决**: 使用 `ensure_sync_mode()` 代替 `enable_sync_mode()`

```python
# 旧方式（可能失败）
sync_mgr.enable_sync_mode()

# 新方式（自动验证）
sync_mgr.ensure_sync_mode()
```

### 问题 2: 销毁传感器时卡住

**原因**: 在同步模式下销毁传感器

**解决**: 使用 `ensure_async_mode()` 或 `CollectorLifecycleManager`

```python
# 方式 1
sync_mgr.ensure_async_mode()
sensor.destroy()

# 方式 2（推荐）
with lifecycle.route_context() as ctx:
    # ... 使用资源 ...
# 自动清理
```

### 问题 3: tick 连续失败

**原因**: 同步模式状态不一致

**解决**: `safe_tick()` 会自动触发恢复机制

```python
# safe_tick 连续失败 3 次后会自动调用 _auto_recover_sync_mode()
success = sync_mgr.safe_tick()
```

## 7. 最佳实践

1. **使用 CollectorLifecycleManager**：统一管理整个生命周期
2. **使用 ensure_* 方法**：主动验证而非被动检测
3. **使用 safe_tick**：自动处理失败和恢复
4. **避免直接调用 world.tick()**：使用 sync_mgr.tick() 代替
5. **不要手动管理模式切换**：让 CollectorLifecycleManager 处理

## 8. 从 v1.0 迁移

| v1.0 | v2.0 |
|------|------|
| `enable_sync_mode()` | `ensure_sync_mode()` |
| `enable_async_mode()` | `ensure_async_mode()` |
| 被动检测低速 | 主动验证 + 自动恢复 |
| 手动管理资源 | `CollectorLifecycleManager` |
| `world.tick()` | `sync_mgr.safe_tick()` |
