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

## 已应用的修复

### 修复 1：资源管理器统一管理

**文件**: `core/resource_manager.py`

- 使用 `CarlaResourceManager` 统一管理车辆、传感器的创建和销毁
- 支持 Context Manager 模式，确保资源自动清理
- 同步模式切换等待时间充足

### 修复 2：碰撞恢复机制

**文件**: `core/collision_recovery.py`

- 碰撞后完全清理所有资源
- 从路线 waypoints 中找恢复点
- 在恢复点位置重新生成车辆

### 修复 3：异常检测

**文件**: `detection/anomaly_detector.py`

- 打转检测：车辆原地打转超过阈值
- 翻车检测：pitch/roll 角度超过阈值
- 卡住检测：速度过低持续时间过长

---

## 如果问题仍然存在

### 方案 A：增加客户端超时时间

在连接时设置更长的超时：

```python
client = carla.Client(host, port)
client.set_timeout(120.0)  # 增加到 120 秒
```

### 方案 B：降低模拟帧率

在配置文件 `config/auto_collection_config.json` 中：

```json
{
    "collection_settings": {
        "simulation_fps": 10
    }
}
```

### 方案 C：减少 NPC 数量

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

### 方案 D：手动重置同步模式

如果检测到服务器卡住，可以尝试：

```python
# 1. 切换到异步模式
settings = world.get_settings()
settings.synchronous_mode = False
world.apply_settings(settings)
time.sleep(5.0)

# 2. 等待服务器稳定
time.sleep(2.0)

# 3. 重新开启同步模式
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / fps
world.apply_settings(settings)
time.sleep(2.0)
```

---

## 调试建议

1. **观察控制台输出**：
   - 检查是否有超时警告
   - 检查 `world.tick()` 是否正常返回

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

在 `config/auto_collection_config.json` 中的推荐配置：

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

---

## 代码架构说明

重构后的代码采用模块化设计，各模块职责清晰：

| 模块 | 职责 |
|------|------|
| `core/resource_manager.py` | 资源创建和销毁管理 |
| `core/collision_recovery.py` | 碰撞恢复逻辑 |
| `detection/anomaly_detector.py` | 异常状态检测 |
| `detection/collision_handler.py` | 碰撞事件处理 |
| `noise/noiser.py` | 噪声注入 |

这种设计使得问题定位更容易，修复更精准。
