<div align="center">

# CARLA-CIL

基于条件模仿学习的端到端自动驾驶系统

<img src="logo.png" alt="系统界面" width="700"/>

[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-blue)](https://carla.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x%20|%202.x-orange)](https://pytorch.org/)

</div>

## 这是什么

一套完整的 CIL（Conditional Imitation Learning）自动驾驶流水线，从数据收集到模型训练再到实车推理都有。

主要包含五个部分：
- **数据收集** - 在 CARLA 里自动跑车收集训练数据，支持噪声注入（DAgger）
- **模型训练** - 支持多卡 DDP 训练，有早停和学习率调节
- **增量微调** - 支持在已训练模型上微调，防止灾难性遗忘
- **实时推理** - 加载训练好的模型在 CARLA 里跑
- **导航规划** - 基于 CARLA 的 GlobalRoutePlanner 做路径规划

## 效果演示

### 推理效果

<table>
<tr>
<td align="center" width="33%">

**左转场景**

https://github.com/user-attachments/assets/2b747f1f-049f-4c86-9b5d-d70f5220c136

</td>
<td align="center" width="33%">

**右转场景**

https://github.com/user-attachments/assets/4d78d485-f5af-4d87-bbc6-5c10774e6bc0

</td>
<td align="center" width="33%">

**直行场景**

https://github.com/user-attachments/assets/c659094c-47b0-4d47-a513-e5332857a732

</td>
</tr>
</table>

### DAgger 噪声注入

车辆会故意偏离车道，然后记录恢复过程，用来增强模型鲁棒性：

https://github.com/user-attachments/assets/2b613e98-06e3-4367-8ff4-cc6aa3442a33

## 项目结构

```
CARLA-CIL/
├── collect_data_new/          # 数据收集（重构版，推荐用这个）
│   ├── collectors/            # 各种收集器
│   ├── core/                  # 核心模块（同步管理、天气、路线规划等）
│   ├── detection/             # 异常检测、碰撞处理
│   ├── noise/                 # 噪声注入
│   ├── utils/                 # 工具类
│   └── scripts/               # 运行脚本
│
├── collect_data_old/          # 数据收集（旧版，功能一样但代码比较乱）
│
├── carla_train/               # 训练代码
│   ├── main_ddp.py            # DDP 训练入口
│   ├── carla_net_ori.py       # 网络定义
│   ├── carla_loader_ddp.py    # 数据加载（固定帧数）
│   └── carla_loader_dynamic.py # 数据加载（动态帧数）
│
├── carla_train_traffic/       # 红绿灯场景微调（防遗忘）
│   ├── finetune_anti_forget.py # 防遗忘微调脚本
│   ├── carla_loader_mixed.py  # 混合数据加载器
│   ├── run_finetune.sh        # 微调启动脚本
│   └── README.md              # 微调说明文档
│
├── carla_0.9.16/              # 推理代码
│   ├── carla_inference.py     # 推理入口
│   └── network/carla_net.py   # 网络结构
│
└── agents/navigation/         # CARLA 导航模块
```

## 网络结构

简单说就是：图像过 CNN 提特征，速度过 FC，两个拼一起，然后根据导航命令选不同的分支输出控制量。

```
RGB Image (200×88×3)
       ↓
   8层 CNN (32→64→128→256)
       ↓
   Image FC (512)
       ↓
       ├──────────────┐
       ↓              ↓
   Fusion FC ← Speed FC (128) ← Speed
       ↓
   ┌───┴───┬───────┬───────┐
   ↓       ↓       ↓       ↓
Follow   Left   Right  Straight  ← 根据 Command 选一个
   └───────┴───────┴───────┘
              ↓
    [Steer, Throttle, Brake]
```

导航命令：`2=Follow | 3=Left | 4=Right | 5=Straight`

## 环境配置

需要的东西：
- CARLA 0.9.16
- Python >= 3.8
- PyTorch 1.x 或 2.x
- NumPy < 2.0（2.0 有兼容问题）
- OpenCV, h5py, NetworkX

```bash
# 装依赖
pip install torch torchvision numpy<2.0 opencv-python h5py networkx shapely tensorboardX

# 装 CARLA Python API
pip install /path/to/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-py3.x-linux-x86_64.whl
```

## 快速开始

### 1. 启动 CARLA

```bash
# Windows
CarlaUE4.exe -quality-level=Low

# Linux
./CarlaUE4.sh -quality-level=Low
```

### 2. 收集数据

```bash
cd collect_data_new/scripts
python run_auto_collection.py
```

配置文件在 `collect_data_new/config/auto_collection_config.json`，可以改地图、噪声比例这些。

### 3. 训练

```bash
cd carla_train

# 单卡
python main_ddp.py --batch-size 32

# 多卡
bash run_ddp.sh

# 动态帧数（h5文件帧数不一致时使用）
bash run_ddp.sh --dynamic-loader
```

### 4. 增量微调（防遗忘）

如果模型在某些场景（如红绿灯）表现不好，可以收集专门数据进行微调：

```bash
cd carla_train_traffic

# 修改 run_finetune.sh 中的路径后运行
bash run_finetune.sh
```

支持三种防遗忘策略：
- **混合数据训练** - 新旧数据按比例混合
- **知识蒸馏** - 用旧模型输出作为软标签
- **EWC** - 弹性权重巩固，约束重要参数

详见 `carla_train_traffic/README.md`

### 5. 推理

```bash
cd carla_0.9.16
python carla_inference.py --model model/your_model.pth --town Town01
```

## 数据收集详细说明

### 配置示例

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

### 路线生成策略

- **smart** - 智能路线生成，优先选择转弯多的路线
- **exhaustive** - 穷举所有可能路线
- **traffic_light** - 专门生成经过红绿灯的路线

### 噪声模式

有四种噪声注入方式：
- **Impulse** - 短促脉冲，模拟突发干扰
- **Smooth** - 平滑偏移，缓入缓出
- **Drift** - 正弦漂移，持续偏移
- **Jitter** - 高频抖动，模拟传感器噪声

### 数据格式

存成 h5 文件，包含图像和控制标签：

```
data_{timestamp}.h5
├── rgb: (N, 88, 200, 3) uint8      # 图像数据
└── targets: (N, 25+) float32       # 控制标签
    ├── [0:3] steer, throttle, brake
    ├── [10] speed (km/h)
    └── [24] command (2/3/4/5)
```

### 数据工具

```bash
# 验证数据
python -m collect_data_new.scripts.verify_data --data-path /path/to/data --min-frames 100

# 可视化
python -m collect_data_new.scripts.visualize_data --dir /path/to/data

# 数据平衡（转向命令容易不平衡）
python -m collect_data_new.scripts.run_balance_selector --source /path/to/data --output /path/to/balanced
```

## 训练详细说明

### 基础训练

```bash
cd carla_train

# 使用 run_ddp.sh（推荐）
bash run_ddp.sh

# 或手动指定参数
torchrun --nproc_per_node=6 main_ddp.py \
    --batch-size 1536 \
    --lr 1e-4 \
    --epochs 90 \
    --early-stop \
    --patience 12
```

### 动态帧数支持

如果你的 h5 文件帧数不一致（比如有的 100 帧，有的 200 帧），使用动态加载器：

```bash
bash run_ddp.sh --dynamic-loader --min-frames 10
```

### 增量微调

在已训练模型基础上，使用新数据微调（防止遗忘）：

```bash
cd carla_train_traffic
bash run_finetune.sh
```

关键参数：
- `--mix-ratio 0.3` - 新数据占 30%，旧数据占 70%
- `--distill-alpha 0.3` - 30% 损失来自知识蒸馏
- `--ewc-lambda 5000` - EWC 正则化强度

## 新版 vs 旧版

`collect_data_new` 是重构过的版本，主要改进：
- 同步模式管理更稳定，不容易卡死
- 支持 22 种天气预设
- 支持红绿灯路线专门生成
- 自动生成数据质量报告
- 资源管理更规范，不会泄漏

旧版 `collect_data_old` 功能一样，但代码组织比较乱，不推荐用了。

## 常见问题

### Q: h5 文件帧数不一致怎么办？
A: 使用 `--dynamic-loader` 参数，会自动检测每个文件的帧数。

### Q: 模型在红绿灯场景不停车？
A: 收集红绿灯场景数据，使用 `carla_train_traffic` 进行防遗忘微调。

### Q: 训练时显存不够？
A: 减小 `--batch-size`，或使用 `--use-amp` 开启混合精度。

### Q: 数据收集时卡死？
A: 检查 CARLA 是否正常运行，尝试重启 CARLA 服务器。

## 参考

```bibtex
@inproceedings{codevilla2018end,
  title={End-to-end driving via conditional imitation learning},
  author={Codevilla, Felipe and others},
  booktitle={ICRA},
  year={2018}
}

@inproceedings{dosovitskiy2017carla,
  title={CARLA: An open urban driving simulator},
  author={Dosovitskiy, Alexey and others},
  booktitle={CoRL},
  year={2017}
}
```

## License

MIT
