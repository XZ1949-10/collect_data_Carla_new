# TurtleBot 数据收集

用于收集端到端自动驾驶训练数据，格式与 CARLA 训练数据兼容。

## 目录结构

```
turtlebot_collect/
├── config/                           # 1. 配置模块
│   ├── __init__.py
│   ├── topics.py                     # ROS话题配置 + 手柄/键盘按键映射
│   ├── robot_config.py               # TurtleBot 各型号运动参数
│   ├── image_config.py               # 图像处理参数 (尺寸、裁剪)
│   ├── command_config.py             # 导航命令定义 + targets向量索引
│   ├── storage_config.py             # 数据存储参数 (压缩、文件名、分割帧数)
│   ├── display_config.py             # 显示参数 (窗口、颜色、字体)
│   └── collector_config.py           # 收集参数 (帧率、传感器同步)
│
├── ros_data/                         # 2. 原生 ROS 数据模块
│   ├── __init__.py
│   ├── ros_data_collector.py         # 原生 ROS 数据结构收集
│   └── ros_image_handler.py          # 原生 ROS 图像处理
│
├── processing/                       # 3. 数据处理模块
│   ├── __init__.py
│   ├── image_processor.py            # 图像预处理 (裁剪、缩放)
│   └── control_converter.py          # 控制信号转换
│
├── control/                          # 4. 控制器模块
│   ├── __init__.py
│   ├── joystick_controller.py        # 手柄控制器
│   └── keyboard_controller.py        # 键盘控制器
│
├── storage/                          # 5. 数据存储模块
│   ├── __init__.py
│   └── data_saver.py                 # H5 格式数据保存 (支持自动分割)
│
├── visualization/                    # 6. 可视化模块
│   ├── __init__.py
│   └── collector_visualizer.py       # 收集过程可视化界面
│
├── collector.py                      # 7. 主收集器
├── __init__.py
├── requirements.txt
└── README.md
```

## 主要功能

### 1. 可配置收集帧率
通过 `--rate` 参数或 `CollectorConfig.DEFAULT_RATE_HZ` 设置收集频率。

### 2. 传感器时间戳对齐
启用后，只有当图像和里程计时间戳差值小于阈值时才收集数据，确保数据同步。
- 配置: `CollectorConfig.ENABLE_SENSOR_SYNC`
- 阈值: `CollectorConfig.MAX_SENSOR_TIME_DIFF` (默认 100ms)
- 命令行: `--no-sync` 禁用

### 3. 自动分割保存
达到指定帧数后自动保存为新文件，避免单个文件过大。
- 配置: `StorageConfig.FRAMES_PER_FILE` (默认 200 帧)
- 命令行: `--frames` 参数

### 4. 可配置保存路径
- 配置: `StorageConfig.DEFAULT_OUTPUT_DIR`
- 命令行: `--output` 参数

### 5. 独立可视化模块 (美化版 v3 - 双栏布局)
`visualization/collector_visualizer.py` 负责所有显示逻辑，采用现代深色主题设计，全中文界面。

**界面布局:**
```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────┬────────────────────────────┐  │
│  │                                           │   ┌──────────────────────┐ │  │
│  │                                           │   │   运行状态           │ │  │
│  │           [摄像头画面]                     │   │  ● 录制中            │ │  │
│  │          (200x88 等比放大 3x)              │   │  ● 同步正常          │ │  │
│  │                                           │   ├──────────────────────┤ │  │
│  │                                           │   │   导航命令           │ │  │
│  │                                           │   │   [ 跟随 ]           │ │  │
│  │                                           │   ├──────────────────────┤ │  │
│  │                                           │   │   数据统计           │ │  │
│  │                                           │   │  片段: 5  帧数: 123  │ │  │
│  └───────────────────────────────────────────┴────────────────────────────┘  │
│  ┌─────────────────────────────────────┬──────────────────────────────────┐  │
│  │  CARLA 控制信号                     │  TurtleBot 速度                  │  │
│  │  转向 [◀━━━━━━│━━━━━━▶]  -0.350    │  线速度 [▓▓▓▓░░░░]  +0.22 m/s   │  │
│  │  油门 [▓▓▓▓▓▓░░░░░░░░]   0.750    │  角速度 [◀━━│━━━▶]  -0.15 r/s   │  │
│  │  刹车 [░░░░░░░░░░░░░░]   0.000    │  速度   [▓▓▓░░░░░]   2.5 km/h   │  │
│  └─────────────────────────────────────┴──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**显示内容:**
- 摄像头画面 (保持 200x88 宽高比，放大 3 倍显示)
- 右侧面板:
  - 运行状态: 录制中/待机 (带脉冲动画)、同步状态
  - 导航命令: 跟随/左转/右转/直行 (带颜色区分)
  - 数据统计: 片段编号、帧数
- 底部双栏:
  - 左栏 CARLA 控制信号: 转向(中心对称)、油门、刹车
  - 右栏 TurtleBot 速度: 线速度、角速度(中心对称)、速度

**界面特点:**
- 现代深色主题，减少视觉疲劳
- 全中文标签，清晰易懂
- 底部双栏分离显示转换后的控制信号和原始速度
- 进度条带刻度线和高亮边缘
- 录制状态红色脉冲动画提醒

---

## 配置模块说明

| 文件 | 说明 |
|------|------|
| `topics.py` | ROS 话题名称、手柄/键盘按键映射 |
| `robot_config.py` | TurtleBot 各型号的 max_linear、max_angular 参数 |
| `image_config.py` | 输出图像尺寸 (200x88)、裁剪比例 |
| `command_config.py` | 导航命令值、targets 向量索引 |
| `storage_config.py` | 文件前缀、压缩算法、每文件帧数 |
| `display_config.py` | 窗口尺寸、状态颜色、字体配置 |
| `collector_config.py` | 收集帧率、传感器同步配置 |

---

## 数据格式

与 CARLA 训练数据保持一致：

```python
# H5 文件结构
{
    'rgb': (N, 88, 200, 3),      # 图像数据 (N <= 200)
    'targets': (N, 25),          # 控制信号
}

# targets 向量定义 (索引定义在 config/command_config.py)
targets[0]  = steer         # 方向 (-1.0 ~ 1.0)
targets[1]  = throttle      # 油门 (0.0 ~ 1.0)
targets[2]  = brake         # 刹车 (0.0 ~ 1.0)
targets[10] = speed         # 速度 (km/h)
targets[20] = linear_vel    # 原始线速度 (m/s)
targets[21] = angular_vel   # 原始角速度 (rad/s)
targets[24] = command       # 导航命令 (2=Follow, 3=Left, 4=Right, 5=Straight)
```

---

## 使用方法

### 1. 启动 TurtleBot

```bash
# TurtleBot 2 / Kobuki
roslaunch turtlebot_bringup minimal.launch

# 启动摄像头
roslaunch turtlebot_bringup 3dsensor.launch

# 启动手柄驱动
rosrun joy joy_node
```

### 2. 运行数据收集

```bash
# 默认配置
python collector.py

# 自定义配置
python collector.py \
    --output ./my_data \
    --rate 15 \
    --frames 100 \
    --model burger \
    --joystick ps4

# 禁用传感器同步
python collector.py --no-sync
```

### 3. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output, -o` | 输出目录 | `./turtlebot_data` |
| `--control, -c` | 控制类型 (joystick/keyboard) | `joystick` |
| `--joystick, -j` | 手柄类型 (xbox/ps4) | `xbox` |
| `--model, -m` | TurtleBot型号 | `turtlebot2` |
| `--rate, -r` | 收集频率 (Hz) | `10` |
| `--frames, -f` | 每个文件的帧数 | `200` |
| `--no-sync` | 禁用传感器同步 | - |

---

## 手柄按键映射

### Xbox 手柄
| 按键 | 功能 |
|------|------|
| 左摇杆 Y轴 | 前进/后退 |
| 右摇杆 X轴 | 左转/右转 |
| Start | 开始录制 |
| Back | 停止录制 |
| Y | Follow 命令 |
| X | Left 命令 |
| B | Right 命令 |
| A | Straight 命令 |

### PS4 手柄
| 按键 | 功能 |
|------|------|
| 左摇杆 | 移动控制 |
| X | 开始/停止录制 |
| L1 | Follow 命令 |
| □ | Left 命令 |
| O | Right 命令 |
| △ | Straight 命令 |
| R1 | 紧急停止 |

---

## 配置修改示例

### 修改收集帧率和传感器同步

编辑 `config/collector_config.py`:

```python
class CollectorConfig:
    DEFAULT_RATE_HZ = 15              # 改为 15Hz
    ENABLE_SENSOR_SYNC = True
    MAX_SENSOR_TIME_DIFF = 0.05       # 改为 50ms
```

### 修改每文件帧数

编辑 `config/storage_config.py`:

```python
class StorageConfig:
    FRAMES_PER_FILE = 500             # 改为 500 帧
```

### 修改图像处理参数

编辑 `config/image_config.py`:

```python
class ImageConfig:
    OUTPUT_WIDTH = 200
    OUTPUT_HEIGHT = 88
    CROP_TOP_RATIO = 0.3
```

---

## 控制信号转换

不同 TurtleBot 型号的参数 (定义在 `config/robot_config.py`)：

| 型号 | max_linear (m/s) | max_angular (rad/s) |
|------|------------------|---------------------|
| turtlebot1 | 0.5 | 1.0 |
| turtlebot2 / kobuki | 0.7 | 1.0 |
| burger | 0.22 | 2.84 |
| waffle | 0.26 | 1.82 |
| waffle_pi | 0.26 | 1.82 |


<!-- 现在存在的问题就是这个刹车就是减速有点映射问题 -->