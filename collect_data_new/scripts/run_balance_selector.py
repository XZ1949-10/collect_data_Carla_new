#!/usr/bin/env python
# coding=utf-8
"""
数据平衡选择脚本

使用方法:
    # 交互式模式
    python -m collect_data_new.scripts.run_balance_selector
    
    # 命令行模式
    python -m collect_data_new.scripts.run_balance_selector \
        --source E:/carla_data1,E:/carla_data2 \
        --output E:/selected_data
    
    # 仅分析不复制
    python -m collect_data_new.scripts.run_balance_selector \
        --source E:/carla_data1 \
        --analyze-only
    
    # 自定义比例
    python -m collect_data_new.scripts.run_balance_selector \
        --source E:/carla_data1 \
        --output E:/selected \
        --follow 0.3 --left 0.25 --right 0.25 --straight 0.2
"""

import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collect_data_new.utils.balance_selector import main

if __name__ == '__main__':
    main()
