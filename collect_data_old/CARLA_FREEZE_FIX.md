# CARLA 服务器卡住问题修复指南

## 问题诊断

CARLA 服务器（CarlaUE4.exe）卡住通常是由于**同步模式状态不一致**造成的。

### 症状

- 车辆速度一直是 0.0，即使 throttle > 0
- CARLA 原生窗口（CarlaUE4.exe）场景不变化
- Python 脚本继续运行，但物理模拟没有推进

### 根本原因

1. **同步模式状态不一致**：Python 客户端认为是同步模式，但 CARLA 服务器实际上不是
2. **资源清理后同步模式没有正确恢复**：切换异步→同步时服务器没有正确响应
3. **CARLA 服务器内部状态异常**：需要重启服务器

### 其他可能原因

1. **资源清理时的同步模式问题**：销毁传感器时没有先切换到异步模式
2. **等待时间不足**：模式切换后等待时间太短，服务器还没处理完
3. **传感器回调阻塞**：图像处理时间过长阻塞了 CARLA 的传感器线程

---

## 已应用的修复（2025-12-09）

### 修复 1：`step_simulation()` 添加超时保护

**文件**: `base_collector.py`

- `agent.run_step()` 添加 5 秒超时保护
- `world.tick()` 添加 10 秒超时保护
- 超时时自动跳过本帧，避免死锁
- 连续 3 次超时会打印警告

### 修复 2：增加资源清理等待时间

**文件**: `base_collector.py`, `auto_full_town_collection.py`, `carla_resource_manager_v2.py`

- 同步模式切换等待时间从 0.3 秒增加到 1.0 秒
- 每个资源销毁后单独等待 0.2-0.3 秒
- 销毁完成后等待 1.0 秒让服务器处理

### 修复 3：改进 `_reset_sync_mode()`

**文件**: `auto_full_town_collection.py`

- 切换到异步模式后等待 5 秒
- 额外等待 2 秒让服务器稳定
- 恢复同步模式后等待 2 秒
- 添加详细的日志输出

### 修复 4：改进 `_cleanup_inner_collector()`

**文件**: `auto_full_town_collection.py`

- 添加详细的清理日志
- 每个资源销毁后单独等待
- 恢复同步模式前额外等待

---

## 如果问题仍然存在

### 方案 A：增加客户端超时时间

在 `base_collector.py` 的 `connect()` 方法中：

```python
def connect(self):
    self.client = carla.Client(self.host, self.port)
    self.client.set_timeout(120.0)  # 增加到 120 秒
```

### 方案 B：定期重置同步模式

在 `_auto_collect()` 中添加定期检查：

```python
_last_reset_time = time.time()

while (saved_frames + pending_frames) < self.frames_per_route:
    # 每 60 秒检查一次
    if time.time() - _last_reset_time > 60:
        print("⚠️ 定期检查：重置同步模式...")
        self._reset_sync_mode()
        _last_reset_time = time.time()
    
    # ... 现有代码 ...
```

### 方案 C：降低模拟帧率

在 `auto_collection_config.json` 中：

```json
{
    "collection_settings": {
        "simulation_fps": 10  // 从 20 降低到 10
    }
}
```

### 方案 D：减少 NPC 数量

NPC 过多会增加服务器负担：

```json
{
    "world_settings": {
        "spawn_npc_vehicles": false,
        "num_npc_vehicles": 0,
        "spawn_npc_walkers": false,
        "num_npc_walkers": 0
    }
}
```

---

## 调试建议

1. **观察控制台输出**：
   - `[DEBUG]` 信息显示当前执行到哪一步
   - `⚠️ agent.run_step() 超时` 表示 agent 死锁
   - `⚠️ world.tick() 超时` 表示服务器无响应

2. **检查 CARLA 服务器日志**：
   - 查看 CarlaUE4.exe 的控制台输出
   - 检查是否有错误或警告

3. **监控系统资源**：
   - CPU 使用率是否过高
   - 内存是否不足
   - GPU 是否过载

4. **尝试重启**：
   - 如果服务器完全卡住，需要重启 CarlaUE4.exe
   - 然后重新运行数据收集脚本

---

## 配置建议

在 `auto_collection_config.json` 中的推荐配置：

```json
{
    "collection_settings": {
        "simulation_fps": 20,
        "auto_save_interval": 200
    },
    "advanced_settings": {
        "pause_between_routes": 3,
        "max_retries": 3
    },
    "world_settings": {
        "spawn_npc_vehicles": false,
        "num_npc_vehicles": 0
    }
}
```
