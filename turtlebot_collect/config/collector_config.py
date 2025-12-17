#!/usr/bin/env python
# coding=utf-8
'''
数据收集配置

本文件定义了数据收集过程的参数，包括收集频率和传感器同步设置。

收集流程:
    1. 以固定频率 (rate_hz) 循环
    2. 检查传感器数据是否同步
    3. 如果同步，收集当前帧数据
    4. 达到帧数上限时自动保存

传感器同步说明:
    TurtleBot 的图像和里程计数据来自不同传感器，
    它们的时间戳可能不完全一致。
    启用同步后，只有当两者时间差小于阈值时才收集数据，
    确保图像和速度信息是匹配的。

使用方法:
    from config import CollectorConfig
    
    rate = CollectorConfig.DEFAULT_RATE_HZ        # 10 Hz
    sync = CollectorConfig.ENABLE_SENSOR_SYNC     # True
    max_diff = CollectorConfig.MAX_SENSOR_TIME_DIFF  # 0.1 秒
'''


class CollectorConfig:
    """
    数据收集参数配置
    
    定义了:
    1. 收集频率
    2. 传感器同步设置
    3. 控制器默认配置
    """
    
    # ============ 收集频率 ============
    
    # 默认收集帧率 (Hz)
    # 即每秒收集多少帧数据
    #
    # 选择依据:
    #   - 10 Hz: 标准选择，与 CARLA 训练数据一致
    #   - 15-20 Hz: 更高精度，但数据量更大
    #   - 5 Hz: 数据量小，但可能丢失快速变化
    #
    # 注意: 实际帧率受限于:
    #   - 摄像头帧率 (通常 30 Hz)
    #   - 图像处理速度
    #   - ROS 通信延迟
    DEFAULT_RATE_HZ = 20
    
    # ============ 传感器对齐配置 ============
    
    # 是否启用传感器时间戳对齐
    # 
    # True (推荐): 只收集时间戳对齐的数据
    #   - 优点: 数据质量高，图像和速度信息匹配
    #   - 缺点: 可能丢弃部分帧
    #
    # False: 不检查时间戳，收集所有帧
    #   - 优点: 不丢帧
    #   - 缺点: 图像和速度可能有轻微不匹配
    ENABLE_SENSOR_SYNC = True
    
    # 传感器数据最大时间差 (秒)
    # 当图像时间戳和里程计时间戳的差值超过此阈值时，
    # 认为数据不同步，跳过该帧
    #
    # 调整建议:
    #   - 0.05 (50ms): 严格同步，可能丢帧较多
    #   - 0.1 (100ms): 标准设置（推荐）
    #   - 0.2 (200ms): 宽松同步，基本不丢帧
    #
    # 如果发现界面频繁显示 "SYNC!" 警告，可适当增大此值
    MAX_SENSOR_TIME_DIFF = 0.2  # 100ms
    
    # ============ 控制类型默认配置 ============
    
    # 默认控制类型
    # 可选值:
    #   - 'joystick': 手柄控制（推荐，操作更精确）
    #   - 'keyboard': 键盘控制（无手柄时使用）
    DEFAULT_CONTROL_TYPE = 'joystick'
    
    # 默认手柄类型
    # 可选值:
    #   - 'xbox': Xbox 手柄（包括 Xbox 360, Xbox One）
    #   - 'ps4': PlayStation 4 手柄
    #
    # 不同手柄的按键映射不同，需要正确设置
    # 按键映射定义在 topics.py 的 JoystickConfig 中
    DEFAULT_JOYSTICK_TYPE = 'xbox'
