# 架构设计文档

## 模块依赖图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           scripts/                                   │
│  run_auto_collection.py, run_interactive.py, verify_data.py, etc.   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          collectors/                                 │
│  auto_collector.py, command_based.py, interactive.py                │
│  (业务逻辑层 - 组合各模块实现具体收集功能)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│      core/        │   │    detection/     │   │      noise/       │
│  (核心功能模块)    │   │   (检测模块)       │   │   (噪声模块)       │
│                   │   │                   │   │                   │
│ • base_collector  │   │ • anomaly_detector│   │ • noiser          │
│ • resource_manager│   │ • collision_handler│  └───────────────────┘
│ • npc_manager     │   └───────────────────┘
│ • route_planner   │
│ • collision_recovery│
│ • agent_factory ★ │
│ • sync_mode_manager★│
└───────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           config/                                    │
│                         settings.py                                  │
│  (纯数据配置层 - 无外部依赖，被所有模块引用)                           │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           utils/                                     │
│  data_utils.py, visualization.py, balance_selector.py               │
│  (工具层 - 提供通用功能)                                              │
└─────────────────────────────────────────────────────────────────────┘
```

## 设计原则

### 1. 单一职责原则 (SRP)

每个模块只负责一个功能领域：

| 模块 | 职责 |
|------|------|
| `config/settings.py` | 配置类定义和常量 |
| `core/agent_factory.py` | BasicAgent 创建和配置 |
| `core/base_collector.py` | 数据收集基础功能 |
| `core/resource_manager.py` | CARLA 资源生命周期管理 |
| `core/npc_manager.py` | NPC 车辆和行人管理 |
| `core/route_planner.py` | 路线规划和分析 |
| `core/collision_recovery.py` | 碰撞恢复逻辑 |
| `core/sync_mode_manager.py` | 同步/异步模式管理 |
| `detection/anomaly_detector.py` | 异常行为检测 |
| `detection/collision_handler.py` | 碰撞事件处理 |
| `noise/noiser.py` | 噪声生成 |

### 2. 依赖倒置原则 (DIP)

- 高层模块（collectors）通过 `core/__init__.py` 统一入口导入
- 避免直接导入具体实现，减少循环依赖风险
- 使用配置对象注入依赖

### 3. 开闭原则 (OCP)

- 新增收集器只需继承 `BaseDataCollector`
- 新增检测器只需实现相同接口
- 配置通过 dataclass 扩展

## 关键设计决策

### Agent 工厂模式

**问题**: `base_collector.py` 和 `auto_collector.py` 都需要创建 BasicAgent，导致代码重复。

**解决方案**: 创建 `agent_factory.py` 模块，提供统一的 `create_basic_agent()` 函数。

```python
# 使用工厂函数
from ..core import create_basic_agent

agent = create_basic_agent(
    vehicle=vehicle,
    world_map=world.get_map(),
    destination=destination,
    target_speed=10.0,
    ...
)
```

**优点**:
- 消除代码重复
- 统一配置逻辑
- 便于测试和维护

### 统一导入入口

**问题**: `auto_collector.py` 直接导入多个 core 模块，增加循环依赖风险。

**解决方案**: 通过 `core/__init__.py` 统一导出。

```python
# 推荐方式
from ..core import (
    NPCManager,
    RoutePlanner,
    create_basic_agent,
    is_agents_available
)

# 避免
from ..core.npc_manager import NPCManager
from ..core.route_planner import RoutePlanner
```

### 资源管理器模式 ★

**推荐**: 使用 `ResourceLifecycleHelper`（配合 `SyncModeManager`）管理资源：

```python
from ..core import SyncModeManager, ResourceLifecycleHelper

sync_mgr = SyncModeManager(world)
helper = ResourceLifecycleHelper(sync_mgr)

# 安全生成车辆（自动等待物理稳定）
vehicle = helper.spawn_vehicle_safe(vehicle_bp, spawn_transform)

# 安全创建传感器（自动等待初始化）
camera = helper.create_sensor_safe(camera_bp, transform, vehicle, callback)

# 安全销毁资源（自动切换异步模式）
helper.destroy_all_safe([camera], vehicle)
```

**⚠️ 废弃**: `CarlaResourceManager` 已废弃，因为：
1. 内部有独立的同步模式管理，可能与外部 `SyncModeManager` 冲突
2. 缺少 `safe_tick()` 的自动恢复机制

### 同步模式管理器模式 ★

**问题**: 同步/异步模式切换分散在多个文件中，容易导致状态不一致和死锁。

**解决方案**: 创建 `sync_mode_manager.py` 模块，统一管理模式切换。

```python
from ..core import SyncModeManager, ResourceLifecycleHelper

# 创建管理器
sync_mgr = SyncModeManager(world)
helper = ResourceLifecycleHelper(sync_mgr)

# 使用上下文管理器
with sync_mgr.sync_context():
    # 在同步模式下收集数据
    for _ in range(1000):
        world.tick()
        collect_data()

# 安全销毁资源（自动切换异步模式）
helper.destroy_all_safe([camera, sensor], vehicle)
```

**关键规则**:
- 数据收集循环：必须使用同步模式
- 销毁传感器/车辆：必须先切换到异步模式
- 模式切换后：等待 0.5 秒让设置生效

详细说明请参考 [docs/SYNC_MODE_GUIDE.md](./docs/SYNC_MODE_GUIDE.md)

## 模块详细说明

### config/settings.py

定义所有配置类：

- `CollectorConfig`: 主配置
- `CameraConfig`: 摄像头配置
- `NoiseConfig`: 噪声配置
- `AnomalyConfig`: 异常检测配置
- `NPCConfig`: NPC 配置

### core/agent_factory.py ★

提供 BasicAgent 的统一创建接口：

- `create_basic_agent()`: 创建并配置 BasicAgent
- `is_agents_available()`: 检查 agents 模块是否可用

### core/sync_mode_manager.py ★

统一管理 CARLA 同步/异步模式切换：

- `SyncModeManager`: 模式切换管理器
  - `enable_sync_mode()`: 启用同步模式
  - `enable_async_mode()`: 启用异步模式
  - `reset_sync_mode()`: 重置模式（先异步再同步）
  - `sync_context()`: 同步模式上下文管理器
  - `async_context()`: 异步模式上下文管理器
- `ResourceLifecycleHelper`: 资源生命周期辅助类
  - `spawn_vehicle_safe()`: 安全生成车辆
  - `create_sensor_safe()`: 安全创建传感器
  - `destroy_all_safe()`: 安全销毁所有资源

### core/base_collector.py

数据收集器基类，包含：

- CARLA 连接管理
- 车辆生成和控制
- 传感器设置
- 导航命令获取
- 数据构建

### collectors/auto_collector.py

全自动收集器，组合使用：

- `RoutePlanner`: 路线生成
- `NPCManager`: NPC 管理
- `CollisionRecoveryManager`: 碰撞恢复
- `CommandBasedCollector`: 实际数据收集

### utils/visualization.py

可视化工具：

- `FrameVisualizer`: 数据收集过程实时可视化
  - 显示专家控制值（标签）
  - 显示实际控制值
  - 显示噪声公式（专家值 + 噪声 = 实际值）
  - 显示统计信息（已保存帧数、段数）
- `H5DataVisualizer`: H5数据查看器
  - 支持自动播放模式
  - 支持快捷键操作（H/E跳转首尾帧）
  - 显示数值条（方向盘双向、油门/刹车单向）

### utils/carla_visualizer.py

CARLA 世界可视化：

- `SpawnPointVisualizer`: 生成点可视化（彩虹色柱体+索引）
- `RouteVisualizer`: 路径可视化（起点/终点标记、蓝色路径线）
- `CountdownTimer`: 倒计时器（带进度条）
- `CarlaWorldVisualizer`: 整合所有可视化功能

### utils/report_generator.py

报告生成器：

- `VerificationReport`: 验证报告（JSON格式）
  - 文件统计、帧统计、命令分布
  - 速度统计、控制信号统计
  - 数据质量评分（5项指标）
- `DeletionReport`: 删除报告（JSON+TXT格式）
  - 按原因分类统计
  - 详细文件列表
- `ChartGenerator`: 图表生成器（需要matplotlib）
  - 命令分布饼图
  - 速度统计条形图
  - 控制信号统计
  - 质量评分条形图

### docs/CARLA_FREEZE_FIX.md

CARLA 服务器问题修复指南：

- 问题诊断（症状、根本原因）
- 已应用的修复
- 备选解决方案
- 调试建议
- 配置建议

## 扩展指南

### 添加新的收集器

1. 在 `collectors/` 创建新文件
2. 继承 `BaseDataCollector` 或组合使用
3. 在 `collectors/__init__.py` 导出

### 添加新的检测器

1. 在 `detection/` 创建新文件
2. 实现 `check()` 和 `reset()` 方法
3. 在 `detection/__init__.py` 导出

### 添加新的配置

1. 在 `config/settings.py` 添加新的 dataclass
2. 在 `CollectorConfig` 中添加字段
3. 更新相关模块使用新配置
