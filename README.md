# CARLA-CIL: End-to-End Autonomous Driving via Conditional Imitation Learning

<p align="center">
  <img src="carla_0.9.16.png" alt="Visualization Interface" width="800"/>
</p>

<p align="center">
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CARLA-0.9.16-blue" alt="CARLA Version"/>
  <img src="https://img.shields.io/badge/Python-3.8+-green" alt="Python Version"/>
  <img src="https://img.shields.io/badge/PyTorch-1.x%20%7C%202.x-orange" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## 📖 Abstract

本项目实现了基于 **条件模仿学习（Conditional Imitation Learning, CIL）** 的端到端自动驾驶系统。系统包含完整的数据收集、模型训练和实时推理流程，支持在 CARLA 0.9.16 模拟器中进行全自动化的驾驶数据收集，并实现了 DAgger 风格的噪声注入策略以增强模型鲁棒性。

**主要贡献：**
- 🚗 全自动化数据收集系统，支持智能路线规划、碰撞恢复、命令平衡
- 🎯 DAgger 风格噪声注入，包含 Impulse/Smooth/Drift/Jitter 四种模式
- 🧠 条件模仿学习网络实现，支持四种导航命令的分支预测
- 📊 完整的数据验证、可视化和平衡工具链

---

## 🎬 Demo

### 模型推理演示

使用训练好的 CIL 模型在 CARLA 中进行实时自动驾驶推理：

https://github.com/user-attachments/assets/84c0a3c8-fcc1-49ba-aa3a-576ed7d35160

> 📹 **[点击下载/查看完整视频：模型推理演示](使用训练好的模型进行推理的过程.mp4)**

### 噪声注入数据收集

DAgger 风格噪声注入后的数据收集过程，展示车辆偏离-恢复行为：

https://github.com/user-attachments/assets/2b613e98-06e3-4367-8ff4-cc6aa3442a33

> 📹 **[点击下载/查看完整视频：噪声注入演示](加噪之后示例视频HD.mp4)**

### 收集可视化界面介绍

<p align="center">
  <img src="collect_data_old/可视化界面的显示示例.png" alt="Data Collection Visualization" width="800"/>
  <br>
  <em>数据收集可视化界面：显示 RGB 图像、导航命令、车辆状态和噪声信息</em>
</p>

---

## 🏗️ Architecture

### 系统架构

```
CARLA-CIL/
├── agents/                      # CARLA 导航代理模块
│   └── navigation/              # 路径规划和车辆控制
│       ├── basic_agent.py       # 基础导航代理
│       ├── global_route_planner.py  # 全局路径规划器
│       └── local_planner.py     # 局部路径规划器
│
├── carla_0.9.16/               # 🧠 模型推理模块
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
├── collect_data_old/           # 📦 数据收集模块
│   ├── auto_full_town_collection.py  # 全自动数据收集器
│   ├── auto_collection_config.json   # 收集配置文件
│   ├── base_collector.py       # 收集器基类
│   ├── noiser.py               # 噪声注入模块
│   ├── verify_collected_data.py # 数据验证工具
│   └── visualize_h5_data.py    # 数据可视化工具
│
└── _benchmarks_results/        # 📊 基准测试结果
```

### 网络结构

```
┌─────────────────────────────────────────────────────────────┐
│                    CIL Network Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input: RGB Image (200×88×3) + Speed + Navigation Command  │
│                          ↓                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │         CNN Feature Extractor (8 Conv Layers)        │   │
│   │   Conv1→Conv2→Conv3→Conv4→Conv5→Conv6→Conv7→Conv8   │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│   ┌──────────────────┐       ┌──────────────────┐          │
│   │ Image FC (512-d) │       │ Speed FC (128-d) │          │
│   └────────┬─────────┘       └────────┬─────────┘          │
│            └──────────┬───────────────┘                     │
│                       ↓                                      │
│   ┌─────────────────────────────────────────────────────┐   │
│   │            Feature Fusion (640 → 512-d)              │   │
│   └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│   ┌─────────┬─────────┬─────────┬─────────┐                │
│   │ Follow  │  Left   │  Right  │Straight │  ← Command     │
│   │ Branch  │ Branch  │ Branch  │ Branch  │    Selection   │
│   └────┬────┴────┬────┴────┬────┴────┬────┘                │
│        └─────────┴─────────┴─────────┘                      │
│                          ↓                                   │
│            Output: [Steer, Throttle, Brake]                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Requirements

| Dependency | Version |
|------------|---------|
| CARLA Simulator | 0.9.16 |
| Python | ≥ 3.8 |
| PyTorch | 1.x / 2.x |
| NumPy | < 2.0 |
| OpenCV | Latest |
| h5py | Latest |

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/carla-cil.git
cd carla-cil

# Install dependencies
pip install torch torchvision numpy<2.0 opencv-python h5py networkx shapely

# Install CARLA Python API (adjust path to your CARLA installation)
pip install /path/to/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-py3.x-linux-x86_64.whl
```

---

## 🚀 Quick Start

### 1. Data Collection

```bash
# Start CARLA server
CarlaUE4.exe -quality-level=Low

# Run automatic data collection
cd collect_data_old
python auto_full_town_collection.py
```

### 2. Model Inference

```bash
cd carla_0.9.16
python carla_inference.py --model model/your_model.pth --town Town01
```

---

## 📊 Data Collection

### 配置文件

编辑 `collect_data_old/auto_collection_config.json`：

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
    }
}
```

### 噪声注入模式

| Mode | Description | Use Case |
|------|-------------|----------|
| **Impulse** | 短促脉冲，快速上升下降 | 模拟突发干扰 |
| **Smooth** | 平滑偏移，缓入缓出 | 模拟渐进偏离 |
| **Drift** | 正弦波形，缓慢漂移 | 模拟持续偏移 |
| **Jitter** | 高频抖动，随机序列 | 模拟传感器噪声 |

### 数据格式

```
data_cmd{command}_{timestamp}.h5
├── rgb: (N, 200, 88, 3) uint8    # RGB images
└── targets: (N, 4) float32       # [steer, throttle, brake, speed]
```

**Command Types:** 2=Follow, 3=Left, 4=Right, 5=Straight

---

## 🔧 Tools

### Data Verification
```bash
python collect_data_old/verify_collected_data.py --path /path/to/data --min-frames 200
```

### Data Visualization
```bash
python collect_data_old/visualize_h5_data.py --file data.h5
```

### Data Balancing
```bash
python collect_data_old/balance_data_selector.py --source /path/to/data --output /path/to/balanced
```

---

## 📈 Results

基准测试结果存储在 `_benchmarks_results/` 目录下，包含在 Town01 上的多次评估结果。

---

## 📚 References

```bibtex
@inproceedings{codevilla2018end,
  title={End-to-end driving via conditional imitation learning},
  author={Codevilla, Felipe and M{\"u}ller, Matthias and L{\'o}pez, Antonio and Koltun, Vladlen and Dosovitskiy, Alexey},
  booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4693--4700},
  year={2018},
  organization={IEEE}
}

@inproceedings{dosovitskiy2017carla,
  title={CARLA: An open urban driving simulator},
  author={Dosovitskiy, Alexey and Ros, German and Codevilla, Felipe and Lopez, Antonio and Koltun, Vladlen},
  booktitle={Conference on robot learning},
  pages={1--16},
  year={2017},
  organization={PMLR}
}

@inproceedings{chen2020learning,
  title={Learning by cheating},
  author={Chen, Dian and Zhou, Brady and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={Conference on Robot Learning},
  pages={66--75},
  year={2020},
  organization={PMLR}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [CARLA Simulator](https://carla.org/) - Open-source autonomous driving simulator
- [CIL Paper](https://arxiv.org/abs/1710.02410) - End-to-end Driving via Conditional Imitation Learning
- [Learning by Cheating](https://arxiv.org/abs/1912.12294) - Inspiration for data collection strategies

---

<p align="center">
  Made with ❤️ for autonomous driving research
</p>