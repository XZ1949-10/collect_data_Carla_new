#!/usr/bin/env python
# coding=utf-8
"""
红绿灯管理器独立使用示例

演示如何独立使用 TrafficLightManager 模块，
不依赖数据收集器，可安全调用不会造成卡顿。

使用方法:
    python -m collect_data_new.scripts.traffic_light_demo --host localhost --port 2000
"""

import argparse
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import carla
except ImportError:
    print("❌ 无法导入 CARLA 模块，请确保已安装 CARLA Python API")
    sys.exit(1)

from collect_data_new.core import (
    TrafficLightManager,
    TrafficLightTiming,
    TrafficLightState,
    TRAFFIC_LIGHT_PRESETS,
    configure_traffic_lights,
)


def demo_basic_usage(world):
    """基础用法演示"""
    print("\n" + "="*60)
    print("📖 基础用法演示")
    print("="*60)
    
    # 创建管理器
    tl_manager = TrafficLightManager(world, verbose=True)
    
    # 打印当前状态
    tl_manager.print_status()
    
    # 设置红绿灯时间
    print("\n🔧 设置红绿灯时间...")
    tl_manager.set_timing(red=5.0, green=10.0, yellow=2.0)
    
    # 等待一下让设置生效
    time.sleep(1.0)
    
    # 再次打印状态
    tl_manager.print_status()
    
    return tl_manager


def demo_presets(tl_manager):
    """预设配置演示"""
    print("\n" + "="*60)
    print("📖 预设配置演示")
    print("="*60)
    
    print("\n可用的预设配置:")
    for name, timing in TRAFFIC_LIGHT_PRESETS.items():
        print(f"  {name}: 红={timing.red_time}s, 绿={timing.green_time}s, 黄={timing.yellow_time}s")
    
    # 使用预设
    print("\n🔧 使用 'fast' 预设...")
    tl_manager.set_timing_preset('fast')
    
    time.sleep(1.0)
    tl_manager.print_status()


def demo_freeze_unfreeze(tl_manager):
    """冻结/解冻演示"""
    print("\n" + "="*60)
    print("📖 冻结/解冻演示")
    print("="*60)
    
    # 冻结为绿灯
    print("\n🔧 冻结所有红绿灯为绿灯...")
    tl_manager.freeze_all(TrafficLightState.GREEN)
    
    time.sleep(2.0)
    tl_manager.print_status()
    
    # 解冻
    print("\n🔧 解冻所有红绿灯...")
    tl_manager.unfreeze_all()
    
    time.sleep(1.0)
    tl_manager.print_status()


def demo_query_info(tl_manager):
    """查询信息演示"""
    print("\n" + "="*60)
    print("📖 查询信息演示")
    print("="*60)
    
    # 获取所有红绿灯信息
    infos = tl_manager.get_traffic_lights_info()
    
    print(f"\n找到 {len(infos)} 个红绿灯:")
    for i, info in enumerate(infos[:5]):  # 只显示前5个
        print(f"  [{i+1}] ID={info.actor_id}, 状态={info.state.value}, "
              f"位置=({info.location[0]:.1f}, {info.location[1]:.1f})")
    
    if len(infos) > 5:
        print(f"  ... 还有 {len(infos) - 5} 个")


def demo_area_operation(tl_manager, world):
    """区域操作演示"""
    print("\n" + "="*60)
    print("📖 区域操作演示")
    print("="*60)
    
    # 获取一个生成点作为中心
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        center = spawn_points[0].location
        center_tuple = (center.x, center.y, center.z)
        
        print(f"\n以位置 ({center.x:.1f}, {center.y:.1f}) 为中心，半径 50m 内的红绿灯:")
        
        nearby = tl_manager.get_traffic_lights_in_radius(center_tuple, 50.0)
        print(f"  找到 {len(nearby)} 个红绿灯")
        
        if nearby:
            print("\n🔧 设置这些红绿灯为快速周期...")
            tl_manager.set_timing_in_radius(center_tuple, 50.0, red=2.0, green=3.0, yellow=1.0)


def demo_convenience_function(world):
    """便捷函数演示"""
    print("\n" + "="*60)
    print("📖 便捷函数演示")
    print("="*60)
    
    print("\n使用一次性配置函数...")
    success = configure_traffic_lights(world, red=6.0, green=12.0, yellow=2.0)
    print(f"配置结果: {'成功' if success else '失败'}")


def main():
    parser = argparse.ArgumentParser(description='红绿灯管理器演示')
    parser.add_argument('--host', default='localhost', help='CARLA 服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA 服务器端口')
    parser.add_argument('--demo', choices=['all', 'basic', 'presets', 'freeze', 'query', 'area', 'convenience'],
                        default='all', help='要运行的演示')
    args = parser.parse_args()
    
    print("="*60)
    print("🚦 红绿灯管理器演示")
    print("="*60)
    
    # 连接 CARLA
    print(f"\n正在连接到 CARLA 服务器 {args.host}:{args.port}...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        map_name = world.get_map().name.split('/')[-1]
        print(f"✅ 已连接到地图: {map_name}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return 1
    
    try:
        if args.demo in ['all', 'basic']:
            tl_manager = demo_basic_usage(world)
        else:
            tl_manager = TrafficLightManager(world, verbose=True)
        
        if args.demo in ['all', 'presets']:
            demo_presets(tl_manager)
        
        if args.demo in ['all', 'freeze']:
            demo_freeze_unfreeze(tl_manager)
        
        if args.demo in ['all', 'query']:
            demo_query_info(tl_manager)
        
        if args.demo in ['all', 'area']:
            demo_area_operation(tl_manager, world)
        
        if args.demo in ['all', 'convenience']:
            demo_convenience_function(world)
        
        print("\n" + "="*60)
        print("✅ 演示完成")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
