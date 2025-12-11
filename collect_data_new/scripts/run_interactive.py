#!/usr/bin/env python
# coding=utf-8
"""
交互式数据收集脚本

使用方法:
    python -m collect_data_new.scripts.run_interactive
    python -m collect_data_new.scripts.run_interactive --host localhost --port 2000 --town Town01
"""

import argparse
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collect_data_new.config import CollectorConfig, NoiseConfig
from collect_data_new.collectors import InteractiveCollector


def main():
    parser = argparse.ArgumentParser(description='交互式CARLA数据收集')
    
    # 连接参数
    parser.add_argument('--host', type=str, default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--town', type=str, default='Town01', help='地图名称')
    
    # 收集参数
    parser.add_argument('--max-frames', type=int, default=50000, help='最大帧数')
    parser.add_argument('--save-path', type=str, default='./carla_data', help='保存路径')
    parser.add_argument('--target-speed', type=float, default=10.0, help='目标速度 (km/h)')
    parser.add_argument('--fps', type=int, default=20, help='模拟帧率')
    
    # 交通规则
    parser.add_argument('--ignore-lights', action='store_true', default=True, help='忽略红绿灯')
    parser.add_argument('--ignore-signs', action='store_true', default=True, help='忽略停车标志')
    parser.add_argument('--ignore-vehicles', type=int, default=80, help='忽略车辆百分比')
    
    # 噪声参数
    parser.add_argument('--noise', action='store_true', help='启用噪声注入')
    parser.add_argument('--noise-ratio', type=float, default=0.4, help='噪声时间占比')
    parser.add_argument('--max-steer-offset', type=float, default=0.35, help='最大转向偏移')
    
    args = parser.parse_args()
    
    # 创建配置
    noise_config = NoiseConfig(
        enabled=args.noise,
        noise_ratio=args.noise_ratio,
        max_steer_offset=args.max_steer_offset
    )
    
    config = CollectorConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        target_speed=args.target_speed,
        simulation_fps=args.fps,
        ignore_traffic_lights=args.ignore_lights,
        ignore_signs=args.ignore_signs,
        ignore_vehicles_percentage=args.ignore_vehicles,
        noise=noise_config
    )
    
    # 运行收集器
    collector = InteractiveCollector(config)
    collector.run(num_frames=args.max_frames, save_path=args.save_path)


if __name__ == '__main__':
    main()
