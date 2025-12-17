#!/usr/bin/env python
# coding=utf-8
'''
ROS 话题和输入设备配置

本文件定义了:
1. ROS 话题名称 - 用于订阅传感器数据和发布控制命令
2. 手柄按键映射 - Xbox 和 PS4 手柄的按键定义
3. 键盘按键映射 - 键盘控制的按键定义

话题配置说明:
    不同的 TurtleBot 型号和驱动可能使用不同的话题名称。
    如果数据收集无法正常工作，请先用 `rostopic list` 检查实际话题名称，
    然后修改此文件中的配置。

使用方法:
    from config import TopicConfig, JoystickConfig, KeyboardConfig
    
    # 获取话题名称
    image_topic = TopicConfig.IMAGE_RAW   # '/camera/rgb/image_color'
    cmd_topic = TopicConfig.CMD_VEL       # '/mobile_base/commands/velocity'
    
    # 获取手柄按键
    btn_record = JoystickConfig.Xbox.BTN_RECORD  # 7 (Start 按钮)
'''


class TopicConfig:
    """
    ROS 话题配置
    
    定义了数据收集需要订阅/发布的所有话题名称。
    根据你的 TurtleBot 配置修改这些值。
    
    调试方法:
        1. 运行 `rostopic list` 查看所有可用话题
        2. 运行 `rostopic echo <topic>` 检查话题数据
        3. 运行 `rostopic hz <topic>` 检查话题频率
    """
    
    # ============ 图像话题 ============
    # 摄像头发布的原始图像话题
    
    # --- Kinect / Freenect 驱动 ---
    # Xbox 360 Kinect 使用 freenect 驱动时的话题
    # IMAGE_RAW = '/camera/rgb/image_raw'       # 原始图像
    IMAGE_RAW = '/camera/rgb/image_color'       # 彩色图像（推荐）
    # IMAGE_RAW = '/camera/rgb/image_mono'      # 灰度图像
    
    # --- OpenNI / Xtion 驱动 ---
    # IMAGE_RAW = '/camera/rgb/image_raw'
    
    # --- USB 摄像头 (usb_cam) ---
    # IMAGE_RAW = '/usb_cam/image_raw'
    
    # --- Astra 摄像头 ---
    # IMAGE_RAW = '/camera/rgb/image_raw'
    
    # ============ 控制话题 ============
    # 发送速度命令的话题
    
    # --- TurtleBot 2 / Kobuki ---
    # Kobuki 底盘的标准话题
    CMD_VEL = '/mobile_base/commands/velocity'
    
    # --- 通用 cmd_vel ---
    # 如果使用 cmd_vel_mux 或其他配置
    # CMD_VEL = '/cmd_vel'
    
    # --- TurtleBot 3 ---
    # CMD_VEL = '/cmd_vel'
    
    # ============ 传感器话题 ============
    
    # 里程计话题
    # 提供机器人的位置和速度信息
    ODOM = '/odom'
    
    # IMU 话题 (可选)
    # 提供加速度和角速度信息
    IMU = '/imu'
    
    # ============ 手柄话题 ============
    # joy_node 发布的手柄输入话题
    JOY = '/joy'
    
    # ============ 其他话题 (可选) ============
    
    # 激光雷达话题
    SCAN = '/scan'


class JoystickConfig:
    """
    手柄按键映射配置
    
    定义了 Xbox 和 PS4 手柄的按键索引和功能映射。
    
    按键索引说明:
        - axes: 摇杆轴，值范围 [-1.0, 1.0]
        - buttons: 按钮，值为 0 (未按) 或 1 (按下)
    
    调试方法:
        1. 运行 `rosrun joy joy_node`
        2. 运行 `rostopic echo /joy`
        3. 按下按钮，观察 buttons 数组中哪个索引变为 1
        4. 推动摇杆，观察 axes 数组中哪个索引变化
    """
    
    # ============ Xbox 手柄 ============
    class Xbox:
        """
        Xbox 手柄按键映射
        适用于: Xbox 360, Xbox One, Xbox Series 手柄
        
        摇杆布局:
            [左摇杆]  [右摇杆]
              ↑Y        ↑Y
            ←X →      ←X →
              ↓         ↓
        
        按钮布局:
                [Y]
            [X]    [B]
                [A]
        """
        
        # --- 摇杆轴索引 ---
        AXIS_LINEAR = 1      # 左摇杆 Y轴 - 控制前进/后退
                             # +1.0 = 向前推, -1.0 = 向后拉
        AXIS_ANGULAR = 3     # 右摇杆 X轴 - 控制转向
                             # +1.0 = 向左推, -1.0 = 向右推
        
        # --- 按钮索引 ---
        BTN_A = 0            # A 按钮 (绿色)
        BTN_B = 1            # B 按钮 (红色)
        BTN_X = 2            # X 按钮 (蓝色)
        BTN_Y = 3            # Y 按钮 (黄色)
        BTN_LB = 4           # LB 按钮 (左肩键)
        BTN_RB = 5           # RB 按钮 (右肩键)
        BTN_BACK = 6         # Back 按钮 (返回/选择)
        BTN_START = 7        # Start 按钮 (开始/菜单)
        
        # --- 功能映射 ---
        # 将按钮映射到数据收集功能
        BTN_RECORD = BTN_START   # Start 按钮 - 开始录制
        BTN_STOP = BTN_BACK      # Back 按钮 - 停止录制
        BTN_FOLLOW = BTN_Y       # Y 按钮 - Follow 命令 (跟随)
        BTN_LEFT = BTN_X         # X 按钮 - Left 命令 (左转)
        BTN_RIGHT = BTN_B        # B 按钮 - Right 命令 (右转)
        BTN_STRAIGHT = BTN_A     # A 按钮 - Straight 命令 (直行)
        BTN_EMERGENCY = BTN_RB   # RB 按钮 - 紧急停止
    
    # ============ PS4 手柄 ============
    class PS4:
        """
        PS4 手柄按键映射
        适用于: PlayStation 4, PlayStation 5 手柄
        
        按钮布局:
                [△]
            [□]    [○]
                [×]
        """
        
        # --- 摇杆轴索引 ---
        AXIS_LINEAR = 1      # 左摇杆 Y轴 - 控制前进/后退
        AXIS_ANGULAR = 0     # 左摇杆 X轴 - 控制转向
                             # 注意: PS4 使用左摇杆同时控制移动和转向
        
        # --- 按钮索引 ---
        BTN_X = 0            # × 按钮 (叉)
        BTN_O = 1            # ○ 按钮 (圈)
        BTN_SQUARE = 2       # □ 按钮 (方块)
        BTN_TRIANGLE = 3     # △ 按钮 (三角)
        BTN_L1 = 4           # L1 按钮 (左肩键)
        BTN_R1 = 5           # R1 按钮 (右肩键)
        BTN_SHARE = 6        # Share 按钮 (分享)
        BTN_OPTIONS = 7      # Options 按钮 (选项)
        
        # --- 功能映射 ---
        BTN_RECORD = BTN_X       # × 按钮 - 开始/停止录制 (切换)
        BTN_STOP = BTN_OPTIONS   # Options 按钮 - 停止录制 (与 Xbox Back 对应)
        BTN_FOLLOW = BTN_L1      # L1 按钮 - Follow 命令
        BTN_LEFT = BTN_SQUARE    # □ 按钮 - Left 命令
        BTN_RIGHT = BTN_O        # ○ 按钮 - Right 命令
        BTN_STRAIGHT = BTN_TRIANGLE  # △ 按钮 - Straight 命令
        BTN_EMERGENCY = BTN_R1   # R1 按钮 - 紧急停止


class KeyboardConfig:
    """
    键盘按键映射配置
    
    定义了键盘控制模式下的按键映射。
    适用于没有手柄时的数据收集。
    
    注意: 键盘控制精度较低，建议仅用于测试。
    """
    
    # ============ 移动控制 ============
    # 格式: 'key': (linear_ratio, angular_ratio)
    # linear_ratio: 线速度比例 (-1 到 1)，正值前进，负值后退
    # angular_ratio: 角速度比例 (-1 到 1)，正值左转，负值右转
    MOVE_BINDINGS = {
        'w': (1, 0),     # W 键 - 前进
        's': (-1, 0),    # S 键 - 后退
        'a': (0, 1),     # A 键 - 原地左转
        'd': (0, -1),    # D 键 - 原地右转
        'q': (1, 1),     # Q 键 - 左前方移动
        'e': (1, -1),    # E 键 - 右前方移动
        ' ': (0, 0),     # 空格键 - 停止
    }
    
    # ============ 导航命令 ============
    # 格式: 'key': command_value
    # 数字键 1-4 对应四种导航命令
    COMMAND_BINDINGS = {
        '1': 2.0,  # 1 键 - Follow (跟随)
        '2': 3.0,  # 2 键 - Left (左转)
        '3': 4.0,  # 3 键 - Right (右转)
        '4': 5.0,  # 4 键 - Straight (直行)
    }
    
    # ============ 录制控制 ============
    
    # 录制切换键
    # 按一次开始录制，再按一次停止录制
    KEY_RECORD = 'r'
    
    # 退出键
    # ESC 键的 ASCII 码
    KEY_QUIT = '\x1b'  # ESC
