# CARLA 数据收集模块 (重构版)

模块化重构版本，提供清晰的代码结构和低耦合设计。

## 目录结构

```
collect_data_new/
├── __init__.py                    # 包初始化
├── config/                        # 配置模块
│   ├── __init__.py
│   ├── settings.py                # 配置类和常量
│   └── auto_collection_config.json # 自动收集配置模板
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── base_collector.py          # 基础收集器
│   ├── resource_manager.py        # 资源管理器
│   ├── npc_manager.py             # NPC管理器
│   ├── route_planner.py           # 路线规划器
│   ├── collision_recovery.py      # 碰撞恢复管理器
│   ├── weather_manager.py         # 天气管理器 (新增)
│   └── agent_factory.py           # Agent工厂
├── detection/                     # 检测模块
│   ├── __init__.py
│   ├── anomaly_detector.py        # 异常检测器
│   └── collision_handler.py       # 碰撞处理
├── noise/                         # 噪声模块
│   ├── __init__.py
│   └── noiser.py                  # 噪声生成器
├── collectors/                    # 收集器实现
│   ├── __init__.py
│   ├── command_based.py           # 命令分段收集
│   ├── interactive.py             # 交互式收集
│   └── auto_collector.py          # 全自动收集 + 多天气收集
├── utils/                         # 工具模块
│   ├── __init__.py
│   ├── data_utils.py              # 数据处理工具
│   ├── visualization.py           # 可视化工具
│   └── balance_selector.py        # 数据平衡选择器
├── scripts/                       # 脚本入口
│   ├── run_interactive.py         # 交互式收集脚本
│   ├── run_auto_collection.py     # 全自动收集脚本
│   ├── verify_data.py             # 数据验证脚本
│   ├── visualize_data.py          # 数据可视化脚本
│   ├── run_auto_collection.bat    # Windows批处理 (新增)
│   ├── run_multi_weather.bat      # 多天气收集批处理 (新增)
│   └── run_visualizer.bat         # 可视化批处理 (新增)
└── README.md
```

## 模块说明

### config - 配置模块
- `CollectorConfig`: 主配置类
- `NoiseConfig`: 噪声配置
- `AnomalyConfig`: 异常检测配置
- `NPCConfig`: NPC配置
- `CameraConfig`: 摄像头配置
- `WeatherConfig`: 天气配置
- `MultiWeatherConfig`: 多天气收集配置
- `RouteConfig`: 路线生成配置
- `CollisionRecoveryConfig`: 碰撞恢复配置
- `AdvancedConfig`: 高级设置

### core - 核心模块
- `BaseDataCollector`: 数据收集器基类，包含CARLA连接、车辆管理、传感器设置等
- `CarlaResourceManager`: 资源管理器，使用Context Manager模式管理资源生命周期
- `NPCManager`: NPC管理器，管理NPC车辆和行人
- `RoutePlanner`: 路线规划器，支持智能路线生成、去重、缓存
- `CollisionRecoveryManager`: 碰撞恢复管理器，处理碰撞后的恢复逻辑
- `WeatherManager`: 天气管理器，支持22种预设天气和自定义天气参数
- `agent_factory`: Agent工厂模块，提供统一的BasicAgent创建接口

### detection - 检测模块
- `AnomalyDetector`: 异常检测器，检测打转、翻车、卡住等异常
- `CollisionHandler`: 碰撞处理器，处理碰撞事件

### noise - 噪声模块
- `Noiser`: 噪声生成器，支持多种噪声模式（impulse, smooth, drift, jitter）

### collectors - 收集器实现
- `CommandBasedCollector`: 基于命令分段的收集器
- `InteractiveCollector`: 交互式收集器，提供可视化界面
- `AutoFullTownCollector`: 全自动收集器，自动遍历路线、支持碰撞恢复
- `MultiWeatherCollector`: 多天气收集器，自动轮换多个天气进行数据收集

### utils - 工具模块
- `DataSaver`: 数据保存器
- `DataLoader`: 数据加载器
- `FrameVisualizer`: 帧可视化器（支持噪声公式显示、统计信息）
- `H5DataVisualizer`: H5数据可视化器（支持自动播放、数值条显示）
- `CarlaWorldVisualizer`: CARLA世界可视化器（生成点、路径可视化）
- `VerificationReport`: 验证报告生成器（含质量评分）
- `DeletionReport`: 删除报告生成器（JSON+TXT格式）
- `ChartGenerator`: 图表生成器（需要matplotlib）

### docs - 文档
- `CARLA_FREEZE_FIX.md`: CARLA服务器问题修复指南

## 使用方法

### 交互式数据收集

```bash
# 基本使用
python -m collect_data_new.scripts.run_interactive

# 指定参数
python -m collect_data_new.scripts.run_interactive \
    --host localhost \
    --port 2000 \
    --town Town01 \
    --max-frames 50000 \
    --save-path ./carla_data \
    --target-speed 10.0

# 启用噪声注入
python -m collect_data_new.scripts.run_interactive --noise --noise-ratio 0.4
```

### 数据验证

```bash
# 验证数据（生成报告和图表）
python -m collect_data_new.scripts.verify_data --data-path ./carla_data

# 删除不满足条件的文件
python -m collect_data_new.scripts.verify_data --data-path ./carla_data --delete-invalid

# 指定最小帧数
python -m collect_data_new.scripts.verify_data --data-path ./carla_data --min-frames 200

# 不生成图表（无matplotlib时）
python -m collect_data_new.scripts.verify_data --data-path ./carla_data --no-charts
```

验证报告包含：
- 文件统计（总数、有效、损坏、不完整）
- 帧统计（总帧数、命令分布）
- 速度统计（平均、范围）
- 控制信号统计（方向盘、油门、刹车）
- 数据质量评分（5项指标）
- 可视化图表（verification_report.png）
- 删除报告（JSON+TXT格式）

### 数据可视化

```bash
# 查看单个文件
python -m collect_data_new.scripts.visualize_data --file data.h5

# 浏览目录
python -m collect_data_new.scripts.visualize_data --dir ./carla_data

# 自动连续播放（需按空格开始）
python -m collect_data_new.scripts.visualize_data --dir ./carla_data --auto

# 自动连续播放（直接开始）
python -m collect_data_new.scripts.visualize_data --dir ./carla_data --auto --auto-start
```

可视化功能：
- 数据统计打印（速度范围、命令分布）
- 进度条显示
- 数值条可视化（方向盘双向、油门/刹车单向）
- 快捷键操作（H/E跳转首尾帧、W/S调速、N/P切换文件）
- 播放速度显示（FPS）

### 全自动数据收集

```bash
# 基本使用（使用默认配置）
python -m collect_data_new.scripts.run_auto_collection

# 指定配置文件
python -m collect_data_new.scripts.run_auto_collection --config my_config.json

# 命令行参数覆盖
python -m collect_data_new.scripts.run_auto_collection \
    --town Town01 \
    --save-path ./my_data \
    --strategy smart \
    --frames-per-route 500 \
    --target-speed 15.0

# 启用噪声
python -m collect_data_new.scripts.run_auto_collection --noise --noise-ratio 0.4

# 启用实时可视化
python -m collect_data_new.scripts.run_auto_collection --visualize

# 单天气收集
python -m collect_data_new.scripts.run_auto_collection --weather ClearNoon

# 多天气收集（使用预设）
python -m collect_data_new.scripts.run_auto_collection --multi-weather basic

# 多天气收集（自定义列表）
python -m collect_data_new.scripts.run_auto_collection \
    --weather-list ClearNoon CloudyNoon WetNoon HardRainNoon
```

### 天气预设说明

| 预设名称 | 包含天气数量 | 说明 |
|----------|-------------|------|
| basic | 4 | ClearNoon, CloudyNoon, ClearSunset, ClearNight |
| all_noon | 7 | 所有正午天气 |
| all_sunset | 7 | 所有日落天气 |
| all_night | 7 | 所有夜晚天气 |
| clear_all | 3 | 所有晴朗天气 |
| rain_all | 9 | 所有雨天 |
| full | 13 | 完整组合 |
| complete | 22 | 所有天气（包括DustStorm）|

### Windows 批处理脚本

```bash
# 运行自动收集
scripts/run_auto_collection.bat

# 运行多天气收集
scripts/run_multi_weather.bat

# 运行数据可视化
scripts/run_visualizer.bat
```

## 代码示例

### 使用配置类

```python
from collect_data_new.config import CollectorConfig, NoiseConfig

# 创建配置
noise_config = NoiseConfig(
    enabled=True,
    noise_ratio=0.4,
    max_steer_offset=0.35
)

config = CollectorConfig(
    host='localhost',
    port=2000,
    town='Town01',
    target_speed=10.0,
    noise=noise_config
)
```

### 使用资源管理器

```python
from collect_data_new.core import CarlaResourceManager

# 使用 Context Manager
with CarlaResourceManager(world, blueprint_library) as mgr:
    mgr.create_all(spawn_transform, camera_callback, collision_callback)
    # 使用资源...
# 自动清理
```

### 使用异常检测器

```python
from collect_data_new.detection import AnomalyDetector

detector = AnomalyDetector()
detector.configure(spin_threshold=270.0, stuck_time=5.0)

# 每帧检测
if detector.check(vehicle):
    print(f"检测到异常: {detector.anomaly_type_name}")
    detector.reset()
```

### 使用噪声生成器

```python
from collect_data_new.noise import Noiser

noiser = Noiser('Spike', max_offset=0.35, noise_ratio=0.4)

# 每帧应用噪声
noisy_control, is_recovering, is_active = noiser.compute_noise(control, speed)
```

### 使用全自动收集器

```python
from collect_data_new.config import CollectorConfig
from collect_data_new.collectors import AutoFullTownCollector

# 创建配置
config = CollectorConfig(
    host='localhost',
    port=2000,
    town='Town01',
    target_speed=10.0
)

# 创建收集器
collector = AutoFullTownCollector(config)

# 配置路线参数
collector.configure_routes(
    min_distance=50.0,
    max_distance=500.0,
    turn_priority_ratio=0.7
)

# 配置碰撞恢复
collector.configure_recovery(
    enabled=True,
    max_collisions=99,
    skip_distance=25.0
)

# 运行收集
collector.run(
    save_path='./auto_collected_data',
    strategy='smart'
)
```

### 使用路线规划器

```python
from collect_data_new.core import RoutePlanner

# 初始化
planner = RoutePlanner(world, spawn_points)

# 配置参数
planner.configure(
    min_distance=50.0,
    max_distance=500.0,
    overlap_threshold=0.5,
    turn_priority_ratio=0.7
)

# 生成路线（支持缓存）
routes = planner.generate_routes(
    strategy='smart',
    cache_path='./route_cache.json'
)

# 验证单条路线
valid, route, distance = planner.validate_route(start_idx, end_idx)
```

### 使用天气管理器

```python
from collect_data_new.core import WeatherManager, get_weather_list

# 初始化
weather_mgr = WeatherManager(world)

# 设置预设天气
weather_mgr.set_weather_preset('ClearNoon')

# 设置自定义天气
from collect_data_new.core import CustomWeatherParams
params = CustomWeatherParams(
    cloudiness=50.0,
    precipitation=30.0,
    fog_density=10.0
)
weather_mgr.set_custom_weather(params)

# 获取天气列表
weather_list = get_weather_list('basic')  # ['ClearNoon', 'CloudyNoon', 'ClearSunset', 'ClearNight']
```

### 使用多天气收集器

```python
from collect_data_new.config import CollectorConfig
from collect_data_new.collectors import MultiWeatherCollector, run_multi_weather_collection

# 方式1: 使用 MultiWeatherCollector 类
config = CollectorConfig(town='Town01', target_speed=10.0)
collector = MultiWeatherCollector(config)
collector.run(
    weather_list=['ClearNoon', 'CloudyNoon', 'WetNoon'],
    base_save_path='./multi_weather_data',
    strategy='smart'
)

# 方式2: 使用便捷函数
total_frames = run_multi_weather_collection(
    config=config,
    weather_list=['ClearNoon', 'CloudyNoon', 'WetNoon'],
    base_save_path='./multi_weather_data',
    strategy='smart'
)
print(f"总收集帧数: {total_frames}")
```

## 设计原则

1. **单一职责**: 每个模块只负责一个功能领域
2. **低耦合**: 模块之间通过接口通信，减少直接依赖
3. **高内聚**: 相关功能集中在同一模块
4. **可配置**: 使用配置类集中管理参数
5. **可扩展**: 易于添加新的收集器或检测器
6. **工厂模式**: 使用 `agent_factory` 统一创建 BasicAgent，避免代码重复
7. **统一入口**: 通过 `core/__init__.py` 导入，减少循环依赖风险

详细架构设计请参考 [ARCHITECTURE.md](./ARCHITECTURE.md)

## 与旧版本的对比

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 代码组织 | 单文件 | 模块化 |
| 配置管理 | 分散 | 集中配置类 |
| 资源管理 | 手动 | Context Manager |
| 异常检测 | 内嵌 | 独立模块 |
| 噪声注入 | 内嵌 | 独立模块 |
| 可测试性 | 低 | 高 |
