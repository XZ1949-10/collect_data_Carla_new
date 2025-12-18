#!/usr/bin/env python
# coding=utf-8
"""
智能卡住检测器测试脚本

测试场景：
1. 正常行驶 - 不应触发卡住
2. 等红灯 - 不应触发卡住（除非超时）
3. 有油门但不动 - 应触发卡住
4. 位置长时间无变化 - 应触发卡住
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import importlib.util

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接从文件导入配置类（避免包级别的 CARLA 导入）
spec = importlib.util.spec_from_file_location(
    "settings", 
    project_root / "collect_data_new" / "config" / "settings.py"
)
settings_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings_module)
AnomalyConfig = settings_module.AnomalyConfig

# 直接从文件导入检测器（避免包级别的 CARLA 导入）
spec = importlib.util.spec_from_file_location(
    "anomaly_detector", 
    project_root / "collect_data_new" / "detection" / "anomaly_detector.py"
)
anomaly_module = importlib.util.module_from_spec(spec)
# 注入 AnomalyConfig 到模块命名空间
sys.modules['collect_data_new.config'] = type(sys)('config')
sys.modules['collect_data_new.config'].AnomalyConfig = AnomalyConfig
spec.loader.exec_module(anomaly_module)

AnomalyDetector = anomaly_module.AnomalyDetector
VehicleState = anomaly_module.VehicleState
AnomalyType = anomaly_module.AnomalyType
StuckReason = anomaly_module.StuckReason


def test_normal_driving():
    """测试正常行驶场景"""
    print("\n" + "="*60)
    print("测试1: 正常行驶")
    print("="*60)
    
    config = AnomalyConfig()
    detector = AnomalyDetector(config)
    
    # 模拟正常行驶：速度正常，位置变化
    for i in range(50):
        state = VehicleState(
            speed=5.0,  # 5 m/s
            throttle=0.5,
            x=float(i),  # 位置变化
            y=0.0,
            timestamp=time.time()
        )
        result = detector.check(state)
        if result:
            print(f"❌ 误判为卡住！帧 {i}")
            return False
    
    print("✅ 正常行驶未触发卡住检测")
    return True


def test_waiting_at_red_light():
    """测试等红灯场景（无 world，简化测试）"""
    print("\n" + "="*60)
    print("测试2: 低速停车（模拟等红灯，无油门）")
    print("="*60)
    
    config = AnomalyConfig()
    config.stuck_time_threshold = 3.0  # 缩短测试时间
    detector = AnomalyDetector(config)
    
    # 模拟等红灯：速度为0，无油门，位置不变
    start_time = time.time()
    for i in range(100):
        state = VehicleState(
            speed=0.0,
            throttle=0.0,  # 无油门 = 主动停车
            brake=0.5,
            x=10.0,  # 位置不变
            y=20.0,
            timestamp=time.time()
        )
        result = detector.check(state)
        
        # 无油门停车，即使位置不变也不应该立即判定为卡住
        # （因为可能是正常停车）
        if result and (time.time() - start_time) < config.stuck_time_threshold * 1.5:
            print(f"⚠️ 过早判定为卡住（{time.time() - start_time:.1f}秒）")
        
        time.sleep(0.05)
        
        if time.time() - start_time > 6.0:
            break
    
    status = detector.get_status()
    print(f"检测状态: {status['anomaly_type']}")
    print("✅ 低速无油门停车测试完成")
    return True


def test_throttle_but_no_movement():
    """测试有油门但不动的场景"""
    print("\n" + "="*60)
    print("测试3: 有油门但不动（真正卡住）")
    print("="*60)
    
    config = AnomalyConfig()
    config.stuck_time_threshold = 2.0  # 缩短测试时间
    config.stuck_consecutive_attempts = 3
    detector = AnomalyDetector(config)
    
    # 模拟卡住：有油门但速度为0，位置不变
    start_time = time.time()
    stuck_detected = False
    
    for i in range(200):
        state = VehicleState(
            speed=0.0,
            throttle=0.5,  # 有油门
            brake=0.0,
            x=10.0,  # 位置不变
            y=20.0,
            timestamp=time.time()
        )
        result = detector.check(state)
        
        if result:
            elapsed = time.time() - start_time
            print(f"✅ 检测到卡住！耗时 {elapsed:.1f}秒")
            stuck_detected = True
            
            # 检查卡住原因
            analysis = detector.last_stuck_analysis
            if analysis:
                print(f"   原因: {analysis.reason.name}")
                print(f"   详情: {analysis.details}")
            break
        
        time.sleep(0.05)
    
    if not stuck_detected:
        print("❌ 未能检测到卡住")
        return False
    
    return True


def test_position_no_change():
    """测试位置长时间无变化的场景"""
    print("\n" + "="*60)
    print("测试4: 位置长时间无变化")
    print("="*60)
    
    config = AnomalyConfig()
    config.stuck_time_threshold = 2.0
    detector = AnomalyDetector(config)
    
    # 先正常行驶一段
    for i in range(20):
        state = VehicleState(
            speed=3.0,
            throttle=0.3,
            x=float(i),
            y=0.0,
            timestamp=time.time()
        )
        detector.check(state)
        time.sleep(0.02)
    
    # 然后停止但有微小油门
    start_time = time.time()
    stuck_detected = False
    
    for i in range(200):
        state = VehicleState(
            speed=0.1,  # 极低速度
            throttle=0.15,  # 有油门
            x=20.0,  # 位置不变
            y=0.0,
            timestamp=time.time()
        )
        result = detector.check(state)
        
        if result:
            elapsed = time.time() - start_time
            print(f"✅ 检测到卡住！耗时 {elapsed:.1f}秒")
            stuck_detected = True
            break
        
        time.sleep(0.05)
    
    if not stuck_detected:
        print("❌ 未能检测到卡住")
        return False
    
    return True


def test_status_report():
    """测试状态报告功能"""
    print("\n" + "="*60)
    print("测试5: 状态报告")
    print("="*60)
    
    config = AnomalyConfig()
    detector = AnomalyDetector(config)
    
    # 添加一些状态
    for i in range(10):
        state = VehicleState(
            speed=2.0,
            throttle=0.3,
            x=float(i),
            y=0.0,
            timestamp=time.time()
        )
        detector.check(state)
    
    status = detector.get_status()
    print(f"状态报告:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("✅ 状态报告测试完成")
    return True


def main():
    """运行所有测试"""
    print("="*60)
    print("智能卡住检测器测试")
    print("="*60)
    
    tests = [
        ("正常行驶", test_normal_driving),
        ("等红灯（无油门停车）", test_waiting_at_red_light),
        ("有油门但不动", test_throttle_but_no_movement),
        ("位置无变化", test_position_no_change),
        ("状态报告", test_status_report),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 测试 '{name}' 出错: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 通过")
    
    return passed == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
