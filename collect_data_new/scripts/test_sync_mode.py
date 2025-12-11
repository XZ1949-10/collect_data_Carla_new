#!/usr/bin/env python
# coding=utf-8
"""
同步模式测试脚本

用于诊断 CARLA 同步模式问题。
运行方式: python -m collect_data_new.scripts.test_sync_mode
"""

import os
import sys
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("❌ CARLA 模块不可用")
    sys.exit(1)


def test_basic_connection():
    """测试基本连接"""
    print("\n" + "="*60)
    print("测试 1: 基本连接")
    print("="*60)
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    print(f"✅ 连接成功！当前地图: {world.get_map().name}")
    
    return client, world


def test_sync_mode_switch(world):
    """测试同步模式切换"""
    print("\n" + "="*60)
    print("测试 2: 同步模式切换")
    print("="*60)
    
    settings = world.get_settings()
    print(f"当前同步模式: {settings.synchronous_mode}")
    
    # 切换到异步模式
    print("切换到异步模式...")
    settings.synchronous_mode = False
    world.apply_settings(settings)
    time.sleep(0.5)
    
    settings = world.get_settings()
    print(f"切换后: {settings.synchronous_mode}")
    assert not settings.synchronous_mode, "切换到异步模式失败"
    print("✅ 异步模式切换成功")
    
    # 切换到同步模式
    print("切换到同步模式...")
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    time.sleep(0.5)
    
    settings = world.get_settings()
    print(f"切换后: {settings.synchronous_mode}")
    assert settings.synchronous_mode, "切换到同步模式失败"
    print("✅ 同步模式切换成功")


def test_tick(world):
    """测试 tick 调用"""
    print("\n" + "="*60)
    print("测试 3: tick 调用")
    print("="*60)
    
    print("执行 10 次 tick...")
    for i in range(10):
        try:
            world.tick(2.0)
            print(f"  tick {i+1}/10 成功")
        except Exception as e:
            print(f"  tick {i+1}/10 失败: {e}")
            return False
    
    print("✅ tick 测试通过")
    return True


def test_get_actors(world):
    """测试 get_actors 调用"""
    print("\n" + "="*60)
    print("测试 4: get_actors 调用（同步模式下）")
    print("="*60)
    
    # 先执行一次 tick
    print("先执行 tick...")
    world.tick(2.0)
    
    # 然后调用 get_actors
    print("调用 get_actors...")
    start = time.time()
    actors = world.get_actors()
    elapsed = time.time() - start
    print(f"✅ get_actors 成功，耗时: {elapsed:.3f}s，共 {len(actors)} 个 actor")
    
    # 过滤车辆
    print("过滤车辆...")
    start = time.time()
    vehicles = actors.filter("*vehicle*")
    elapsed = time.time() - start
    print(f"✅ 过滤成功，耗时: {elapsed:.3f}s，共 {len(vehicles)} 辆车")
    
    return True


def test_spawn_vehicle(world, blueprint_library):
    """测试生成车辆"""
    print("\n" + "="*60)
    print("测试 5: 生成车辆")
    print("="*60)
    
    spawn_points = world.get_map().get_spawn_points()
    print(f"共 {len(spawn_points)} 个生成点")
    
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = spawn_points[0]
    
    print("生成车辆...")
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    
    if vehicle is None:
        print("❌ 生成车辆失败")
        return None
    
    print(f"✅ 车辆生成成功！ID: {vehicle.id}")
    
    # 等待稳定
    print("等待物理稳定（5 次 tick）...")
    for i in range(5):
        world.tick(2.0)
        time.sleep(0.05)
    
    return vehicle


def test_vehicle_control(world, vehicle):
    """测试车辆控制"""
    print("\n" + "="*60)
    print("测试 6: 车辆控制")
    print("="*60)
    
    control = carla.VehicleControl()
    control.throttle = 0.5
    control.steer = 0.0
    
    print("应用控制（油门 0.5）...")
    vehicle.apply_control(control)
    
    print("执行 20 次 tick，观察速度变化...")
    for i in range(20):
        world.tick(2.0)
        velocity = vehicle.get_velocity()
        speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        print(f"  tick {i+1}/20: 速度 {speed:.1f} km/h")
        
        if speed > 1.0:
            print("✅ 车辆开始移动！")
            break
    else:
        print("⚠️ 车辆速度一直很低，可能存在问题")
    
    return True


def cleanup(world, vehicle):
    """清理资源"""
    print("\n" + "="*60)
    print("清理资源")
    print("="*60)
    
    # 切换到异步模式
    print("切换到异步模式...")
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    time.sleep(0.5)
    
    # 销毁车辆
    if vehicle:
        print("销毁车辆...")
        vehicle.destroy()
    
    print("✅ 清理完成")


def main():
    print("="*60)
    print("CARLA 同步模式诊断测试")
    print("="*60)
    
    client = None
    world = None
    vehicle = None
    
    try:
        # 测试 1: 基本连接
        client, world = test_basic_connection()
        blueprint_library = world.get_blueprint_library()
        
        # 测试 2: 同步模式切换
        test_sync_mode_switch(world)
        
        # 测试 3: tick 调用
        if not test_tick(world):
            print("❌ tick 测试失败，停止后续测试")
            return
        
        # 测试 4: get_actors 调用
        if not test_get_actors(world):
            print("❌ get_actors 测试失败，停止后续测试")
            return
        
        # 测试 5: 生成车辆
        vehicle = test_spawn_vehicle(world, blueprint_library)
        if vehicle is None:
            print("❌ 生成车辆失败，停止后续测试")
            return
        
        # 测试 6: 车辆控制
        test_vehicle_control(world, vehicle)
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if world:
            cleanup(world, vehicle)


if __name__ == '__main__':
    main()
