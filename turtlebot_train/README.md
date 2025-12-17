# TurtleBot 端到端训练

基于条件模仿学习 (CIL) 的 TurtleBot 端到端自动驾驶训练代码。

## 特点

- **直接预测速度**: 输出 `linear_vel` 和 `angular_vel`，而不是 CARLA 的 `steer, throttle, brake`
- **分支网络**: 4 个分支对应 4 种导航命令 (Follow, Left, Right, Straight)
- **输出维度**: 4 分支 × 2 维 = 8 维
- **支持分布式训练**: 使用 PyTorch DDP 多卡训练

## 数据格式

与 `turtlebot_collect` 收集的数据格式兼容：

```python
# H5 文件结构
{
    'rgb': (N, 88, 200, 3),      # 图像数据
    'targets': (N, 25),          # 控制信号
}

# targets 向量索引
targets[10] = speed         # 速度 (km/h)
targets[20] = linear_vel    # 线速度 (m/s)
targets[21] = angular_vel   # 角速度 (rad/s)
targets[24] = command       # 导航命令 (2=Follow, 3=Left, 4=Right, 5=Straight)
```

## 网络结构

```
输入:
  - img: (B, 3, 88, 200) RGB 图像
  - speed: (B, 1) 归一化速度 (speed / 25)

输出:
  - pred_control: (B, 8) 控制信号
    - [0:2] Follow 分支: [linear_vel, angular_vel]
    - [2:4] Left 分支: [linear_vel, angular_vel]
    - [4:6] Right 分支: [linear_vel, angular_vel]
    - [6:8] Straight 分支: [linear_vel, angular_vel]
  - pred_speed: (B, 1) 预测速度
```

## 归一化

训练时对速度进行归一化：

```python
# 输入归一化
speed_input = speed_kmh / 25.0

# 输出归一化
linear_vel_norm = linear_vel / MAX_LINEAR_VEL  # MAX_LINEAR_VEL = 0.7 m/s
angular_vel_norm = angular_vel / MAX_ANGULAR_VEL  # MAX_ANGULAR_VEL = 1.0 rad/s
```

推理时需要反归一化：

```python
linear_vel = pred[branch_idx * 2] * MAX_LINEAR_VEL
angular_vel = pred[branch_idx * 2 + 1] * MAX_ANGULAR_VEL
```

## 使用方法

### 1. 准备数据

确保数据目录结构如下：

```
/path/to/data/
├── train/
│   ├── data_001.h5
│   ├── data_002.h5
│   └── ...
└── eval/
    ├── data_001.h5
    └── ...
```

### 2. 训练

```bash
# 动态帧数版本 (推荐，支持不同帧数的 h5 文件)
bash run_ddp_dynamic.sh

# 或手动指定参数
torchrun --nproc_per_node=6 main_ddp.py \
    --batch-size 1536 \
    --lr 1e-4 \
    --epochs 90 \
    --dynamic-loader \
    --min-frames 10 \
    --train-dir /path/to/train \
    --eval-dir /path/to/eval
```

### 3. 测试网络

```bash
python test_turtlebot.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `turtlebot_net_ori.py` | 网络定义，输出 2 维控制 (linear_vel, angular_vel) |
| `turtlebot_loader.py` | 数据加载器 (固定帧数，单 GPU) |
| `turtlebot_loader_ddp.py` | 数据加载器 (固定帧数，DDP) |
| `turtlebot_loader_dynamic.py` | 数据加载器 (动态帧数，DDP) |
| `main_ddp.py` | DDP 分布式训练脚本 |
| `run_ddp_dynamic.sh` | 训练启动脚本 |
| `test_turtlebot.py` | 网络测试脚本 |
| `helper.py` | 辅助工具 |

## 推理示例

```python
from turtlebot_net_ori import FinalNet
import torch

# 加载模型
model = FinalNet(structure=1)
model.load_state_dict(torch.load('model.pth')['state_dict'])
model.eval()

# 准备输入
img = preprocess_image(camera_image)  # (1, 3, 88, 200)
speed = torch.tensor([[current_speed / 25.0]])  # 归一化

# 推理
with torch.no_grad():
    pred_control, pred_speed = model(img, speed)

# 根据命令选择分支
# command: 2=Follow, 3=Left, 4=Right, 5=Straight
branch_idx = command - 2
output = pred_control[0, branch_idx*2:(branch_idx+1)*2]

# 反归一化
MAX_LINEAR_VEL = 0.7
MAX_ANGULAR_VEL = 1.0
linear_vel = output[0].item() * MAX_LINEAR_VEL
angular_vel = output[1].item() * MAX_ANGULAR_VEL

print(f"linear_vel: {linear_vel:.3f} m/s")
print(f"angular_vel: {angular_vel:.3f} rad/s")
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | 总批次大小 (所有 GPU) | 1536 |
| `--lr` | 学习率 | 1e-4 |
| `--epochs` | 训练轮数 | 90 |
| `--dynamic-loader` | 使用动态帧数加载器 | False |
| `--min-frames` | 最小帧数阈值 | 10 |
| `--max-linear-vel` | 最大线速度 (归一化用) | 0.7 |
| `--max-angular-vel` | 最大角速度 (归一化用) | 1.0 |
| `--early-stop` | 启用早停 | True |
| `--patience` | 早停耐心值 | 12 |

## 与 CARLA 版本的区别

| 项目 | CARLA 版本 | TurtleBot 版本 |
|------|-----------|---------------|
| 输出维度 | 3 (steer, throttle, brake) | 2 (linear_vel, angular_vel) |
| 总输出 | 4 × 3 = 12 | 4 × 2 = 8 |
| 数据索引 | targets[0:3] | targets[20:22] |
| 归一化 | steer: [-1, 1], throttle/brake: [0, 1] | linear: [-1, 1], angular: [-1, 1] |

## Requirements

- Python 3.6+
- PyTorch >= 1.8
- tensorboardX
- opencv-python
- imgaug
- h5py

## 参考

- [carla_cil_pytorch](https://github.com/onlytailei/carla_cil_pytorch)
- [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
