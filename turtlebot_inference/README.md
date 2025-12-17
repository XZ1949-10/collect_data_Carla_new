# TurtleBot 模型推理

在 TurtleBot 上运行端到端自动驾驶模型推理。

## 目录结构

```
turtlebot_inference/
├── config/                    # 配置模块
│   ├── __init__.py
│   ├── inference_config.py    # 推理参数配置
│   └── topics.py              # ROS 话题和手柄配置
│
├── model/                     # 模型模块
│   ├── __init__.py
│   ├── model_loader.py        # 模型加载
│   └── model_predictor.py     # 模型预测
│
├── processing/                # 数据处理模块
│   ├── __init__.py
│   ├── image_processor.py     # 图像预处理
│   └── control_converter.py   # 控制信号转换
│
├── ros_interface/             # ROS 接口模块
│   ├── __init__.py
│   ├── ros_sensor.py          # 传感器数据订阅
│   └── ros_controller.py      # 速度命令发布
│
├── control/                   # 控制模块
│   ├── __init__.py
│   └── command_controller.py  # 导航命令控制
│
├── inference.py               # 主推理类
├── run_inference.py           # 启动脚本
├── requirements.txt
└── README.md
```

## 使用方法

### 1. 启动 TurtleBot

```bash
# TurtleBot3
roslaunch turtlebot3_bringup turtlebot3_robot.launch

# 启动摄像头
roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch

# 启动手柄（可选，用于切换导航命令）
rosrun joy joy_node
```

### 2. 运行推理

```bash
# 基本用法
python run_inference.py --model /path/to/model.pth

# 指定参数
python run_inference.py \
    --model /path/to/model.pth \
    --turtlebot burger \
    --gpu 0 \
    --rate 10 \
    --joystick xbox

# 使用 CPU
python run_inference.py --model /path/to/model.pth --gpu -1
```

### 3. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model, -m` | 模型权重路径 | (必需) |
| `--turtlebot, -t` | TurtleBot 型号 | `burger` |
| `--gpu, -g` | GPU ID，-1 使用 CPU | `0` |
| `--rate, -r` | 控制频率 (Hz) | `10` |
| `--joystick, -j` | 手柄类型 | `xbox` |
| `--duration, -d` | 运行时长（秒） | 无限 |
| `--image-topic` | 图像话题 | `/camera/rgb/image_raw` |
| `--odom-topic` | 里程计话题 | `/odom` |
| `--cmd-vel-topic` | 速度命令话题 | `/cmd_vel` |

## 手柄控制

运行时可通过手柄切换导航命令：

### Xbox 手柄
| 按键 | 功能 |
|------|------|
| Y | Follow 命令 |
| X | Left 命令 |
| B | Right 命令 |
| A | Straight 命令 |
| Back | 退出 |

### PS4 手柄
| 按键 | 功能 |
|------|------|
| △ | Follow 命令 |
| □ | Left 命令 |
| O | Right 命令 |
| X | Straight 命令 |
| Share | 退出 |

## 配置修改

### 修改 ROS 话题

编辑 `config/topics.py`:

```python
class TopicConfig:
    IMAGE_RAW = '/your/camera/topic'
    CMD_VEL = '/cmd_vel'
    ODOM = '/odom'
```

### 修改 TurtleBot 参数

编辑 `config/inference_config.py`:

```python
TURTLEBOT_PARAMS = {
    'burger': {'max_linear': 0.22, 'max_angular': 2.84},
    # ...
}
```

## 控制信号转换

模型输出 CARLA 格式 (steer, throttle, brake)，转换为 TurtleBot 格式：

```
CARLA                    TurtleBot
─────────────────────────────────────
steer (-1~1)        →    angular_vel (rad/s)
throttle (0~1)      →    linear_vel (m/s)
brake (0~1)         →    linear_vel (负值)
```

转换公式：
- `angular_vel = -steer * max_angular`
- `linear_vel = throttle * max_linear` (前进)
- `linear_vel = -brake * max_linear * 0.5` (后退)
