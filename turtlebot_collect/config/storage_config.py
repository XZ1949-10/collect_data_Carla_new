#!/usr/bin/env python
# coding=utf-8
'''
数据存储配置

本文件定义了数据保存相关的参数，包括文件格式、压缩设置和自动分割配置。
收集的数据以 HDF5 (.h5) 格式保存，与 CARLA 训练数据格式兼容。

H5 文件结构:
    {
        'rgb': (N, 88, 200, 3),    # N 帧 RGB 图像
        'targets': (N, 25),        # N 帧控制信号
        attrs: {                   # 元数据
            'platform': 'turtlebot',
            'frames': N,
            'timestamp': '...',
            'episode': 0
        }
    }

使用方法:
    from config import StorageConfig
    
    output_dir = StorageConfig.DEFAULT_OUTPUT_DIR  # './turtlebot_data'
    frames = StorageConfig.FRAMES_PER_FILE         # 200
'''


class StorageConfig:
    """
    数据存储参数配置
    
    定义了:
    1. 文件命名规则
    2. 自动分割设置
    3. HDF5 压缩参数
    4. 数据集名称
    """
    
    # ============ 文件配置 ============
    
    # 文件名前缀
    # 生成的文件名格式: {prefix}_{episode:04d}_{timestamp}.h5
    # 例如: turtlebot_0001_20231215_143052.h5
    FILE_PREFIX = 'turtlebot'
    
    # 文件扩展名
    # HDF5 格式，高效存储大量数值数据
    FILE_EXTENSION = '.h5'
    
    # 默认输出目录
    # 可通过命令行参数 --output 覆盖
    DEFAULT_OUTPUT_DIR = './turtlebot_data'
    
    # ============ 自动分割配置 ============
    
    # 每个 H5 文件的最大帧数
    # 达到此数量后自动保存当前文件，开始新文件
    # 
    # 设置原因:
    #   1. 避免单个文件过大，便于管理和传输
    #   2. 减少数据丢失风险（程序崩溃时最多丢失一个文件的数据）
    #   3. 便于并行训练时的数据加载
    #
    # 调整建议:
    #   - 200 帧 ≈ 20 秒数据 (10Hz)
    #   - 如需更长的连续序列，可增大到 500 或 1000
    #   - 如需更频繁保存，可减小到 100
    FRAMES_PER_FILE = 200
    
    # ============ 压缩配置 ============
    # HDF5 支持多种压缩算法，用于减小文件体积
    
    # 压缩算法
    # 可选值:
    #   - 'gzip': 通用压缩，兼容性最好（推荐）
    #   - 'lzf': 速度快，压缩率较低
    #   - 'szip': 科学数据专用，需要额外库
    #   - None: 不压缩
    COMPRESSION = 'gzip'
    
    # 压缩级别 (仅 gzip 有效)
    # 范围: 1-9
    #   - 1: 最快速度，最低压缩率
    #   - 9: 最慢速度，最高压缩率
    #   - 4: 平衡选择（推荐）
    #
    # 对于图像数据，压缩率通常在 2-4 倍
    COMPRESSION_LEVEL = 4
    
    # ============ 数据集名称 ============
    # H5 文件中的数据集名称，与 CARLA 训练代码保持一致
    
    # RGB 图像数据集名称
    # 数据格式: (N, H, W, 3), dtype=uint8
    DATASET_RGB = 'rgb'
    
    # 控制信号数据集名称
    # 数据格式: (N, 25), dtype=float32
    DATASET_TARGETS = 'targets'
    
    # ============ 元数据 ============
    
    # 平台标识，写入 H5 文件的 attrs 中
    # 用于区分数据来源 (turtlebot vs carla)
    PLATFORM_NAME = 'turtlebot'
