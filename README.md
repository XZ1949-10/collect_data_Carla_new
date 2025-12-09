# CARLA 自动驾驶数据收集与模型推理系统

基于 CARLA 0.9.16 模拟器的端到端自动驾驶数据收集、模型训练与实时推理系统。实现了条件模仿学习（Conditional Imitation Learning, CIL）方法，支持全自动数据收集、噪声注入、碰撞恢复等高级功能。

## 项目概述

本项目包含三个核心模块：

1. **数据收集模块** (`collect_data_old/`) - 全自动化的驾驶数据收集系统
2. **模型推理模块** (`carla_0.9.16/`) - 训练好的模型在 CARLA 中实时推理
3. **导航代理模块** (`agents/`) - CARLA 官方导航代理，提供路径规划和车辆控制

## 系统架构

```
├── agents/                      # CARLA 导航代理模块
│   └── navigation/              # 路径规划和车辆控制
│       ├── basic_agent.py       # 基础导航代理
│       ├── global_route_planner.py  # 全局路径规划器
│       └── local_planner.py     # 局部路径规划器
│
├── carla_0.9.16/               # 模型推理模块
│   ├── network/                # 神经网络定义
│   │   └── carla_net.py        # CIL 网络结构
│   ├── model/                  # 训练好的模型权重
│   ├── carla_inference.py      # 主推理脚本
│   ├── carla_model_loader.py   # 模型加载器
│   ├── carla_model_predictor.py # 模型预测器
│   ├── carla_image_processor.py # 图像预处理
│   ├── carla_vehicle_controller.py # 车辆控制器
│   ├── carla_sensors.py        # 传感器管理
│   └── carla_visualizer.py     # 可视化工具
│
├── collect_data_old/           # 数据收集模块
│   ├── auto_full_town_collection.py  # 全自动数据收集器
│   ├── auto_collection_config.json   # 收集配置文件
│   ├── base_collector.py       # 收集器基类
│   ├── command_based_data_collection.py # 命令分段收集器
│   ├── noiser.py               # 噪声注入模块
│   ├── verify_collected_data.py # 数据验证工具
│   ├── visualize_h5_data.py    # 数据可视化工具
│   └── balance_data_selector.py # 数据平衡选择器
│
└── carla_cil_pytorch_eval/     # CIL 评估基准测试
```

## 核心功能

### 1. 全自动数据收集

自动在 CARLA 地图中规划路线、收集驾驶数据，支持：

- **智能路线生成**：自动分析地图生成点，规划多样化路线
- **命令平衡**：按比例收集 Follow/Left/Right/Straight 四种导航命令
- **路径去重**：避免收集重复路径的数据
- **碰撞恢复**：碰撞后自动从路线 waypoints 中找恢复点继续收集
- **异常检测**：检测打转、翻车、卡住等异常情况
- **多天气支持**：支持多种天气条件轮换收集

### 2. DAgger 风格噪声注入

实现了多种噪声模式，用于增强模型鲁棒性：

| 噪声模式 | 特点 | 适用场景 |
|---------|------|---------|
| Impulse | 短促脉冲，快速上升下降 | 模拟突发干扰 |
| Smooth | 平滑偏移，缓入缓出 | 模拟渐进偏离 |
| Drift | 正弦波形，缓慢漂移 | 模拟持续偏移 |
| Jitter | 高频抖动，随机序列 | 模拟传感器噪声 |

### 3. 条件模仿学习网络

网络结构基于 CIL 论文实现：

```
输入: RGB图像 (200x88) + 速度 + 导航命令
      ↓
CNN特征提取 (8层卷积)
      ↓
图像特征 FC (512维)
      ↓
速度特征 FC (128维)
      ↓
特征融合 (640 → 512维)
      ↓
条件分支 (4个分支对应4种命令)
      ↓
输出: 转向、油门、刹车
```

## 快速开始

### 环境要求

- CARLA 0.9.16
- Python 3.8+
- PyTorch 1.x / 2.x
- NumPy < 2.0
- OpenCV
- h5py

### 安装依赖

```bash
pip install torch torchvision numpy<2.0 opencv-python h5py networkx shapely
```

### 数据收集

1. 启动 CARLA 服务器：
```bash
CarlaUE4.exe -quality-level=Low
```

2. 修改配置文件 `collect_data_old/auto_collection_config.json`

3. 运行数据收集：
```bash
cd collect_data_old
python auto_full_town_collection.py
```

### 模型推理

```bash
cd carla_0.9.16
python carla_inference.py --model model/your_model.pth --town Town01
```

## 配置说明

### 数据收集配置 (`auto_collection_config.json`)

```json
{
    "carla_settings": {
        "host": "localhost",
        "port": 2000,
        "town": "Town01"
    },
    "route_generation": {
        "strategy": "smart",
        "min_distance": 150.0,
        "max_distance": 400.0,
        "turn_priority_ratio": 0.7
    },
    "noise_settings": {
        "enabled": true,
        "noise_ratio": 0.7,
        "max_steer_offset": 0.5
    },
    "collision_recovery": {
        "enabled": true,
        "recovery_skip_distance": 25.0
    }
}
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `min_distance` | 最小路线距离 (米) | 100-200 |
| `max_distance` | 最大路线距离 (米) | 300-500 |
| `turn_priority_ratio` | 转弯路线占比 | 0.6-0.8 |
| `noise_ratio` | 噪声帧占比 | 0.4-0.7 |
| `target_speed_kmh` | 目标车速 (km/h) | 15-25 |
| `auto_save_interval` | 自动保存间隔 (帧) | 200 |

## 数据格式

收集的数据以 HDF5 格式存储：

```
data_cmd{command}_{timestamp}.h5
├── rgb: (N, 200, 88, 3) uint8    # RGB图像
└── targets: (N, 4) float32       # [steer, throttle, brake, speed]
```

- 图像尺寸：200×88（裁剪后）
- 命令类型：2=Follow, 3=Left, 4=Right, 5=Straight

## 工具脚本

### 数据验证
```bash
python verify_collected_data.py --path /path/to/data --min-frames 200
```

### 数据可视化
```bash
python visualize_h5_data.py --file data.h5
```

### 数据平衡选择
```bash
python balance_data_selector.py --source /path/to/data --output /path/to/balanced
```

## 碰撞恢复机制

当车辆发生碰撞时，系统会：

1. 丢弃当前 segment 的数据（最多 200 帧）
2. 从当前路线的 waypoints 中查找恢复点
3. 跳过碰撞区域（默认 25 米）
4. 在恢复点重新生成车辆，继续沿原路线行驶

这确保了恢复后车辆仍在原规划路线上，而不是随机位置。

## 参考文献

- [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
- [CARLA: An Open Urban Driving Simulator](https://arxiv.org/abs/1711.03938)
- [Learning by Cheating](https://arxiv.org/abs/1912.12294)

## License

MIT License
