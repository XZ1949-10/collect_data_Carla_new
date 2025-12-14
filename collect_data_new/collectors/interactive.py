#!/usr/bin/env python
# coding=utf-8
"""
交互式数据收集启动器

整合生成点可视化和数据收集功能，提供友好的交互式界面。

使用说明：
    1. 启动CARLA服务器
    2. 运行交互式数据收集
    3. 按照提示操作：
       - 首先会看到所有生成点的彩色标记
       - 输入起点索引（例如：0）
       - 输入终点索引（例如：105）
       - 查看蓝色导航路径
       - 输入"开始"开始收集数据
       - 收集完成后选择是否继续

特点：
    ✅ 保持CARLA视角不变
    ✅ 所有可视化标记统一显示时间（30秒）
    ✅ 统一的倒计时进度条，简洁清晰
    ✅ 路径规划失败时自动重新选择
    ✅ 可以连续收集多条路线
"""

import os
import sys
import time
from typing import Optional, Tuple

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

from ..config import CollectorConfig
from ..utils.carla_visualizer import CarlaWorldVisualizer, CountdownTimer
from ..core import SyncModeManager, SyncModeConfig, ResourceLifecycleHelper
from ..core.actor_utils import (
    is_actor_alive,
    safe_destroy_sensor,
    safe_destroy_actor,
    destroy_all_resources,
)
from .command_based import CommandBasedCollector


class InteractiveCollector:
    """交互式数据收集器"""
    
    # 默认可视化持续时间
    DEFAULT_MARKER_DURATION = 30.0
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        
        self.client = None
        self.world = None
        self.spawn_points = []
        
        self.collector: Optional[CommandBasedCollector] = None
        self.route_planner = None
        self.world_visualizer: Optional[CarlaWorldVisualizer] = None
        
        # 同步模式管理器
        self._sync_manager: Optional[SyncModeManager] = None
        self._lifecycle_helper: Optional[ResourceLifecycleHelper] = None
    
    def connect(self):
        """连接到CARLA服务器"""
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        print("\n" + "="*70)
        print("🚗 CARLA 交互式数据收集器")
        print("="*70)
        
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(10.0)
        
        self.world = self.client.get_world()
        current_map = self.world.get_map().name.split('/')[-1]
        
        if current_map != self.config.town:
            print(f"正在加载地图 {self.config.town}...")
            self.world = self.client.load_world(self.config.town)
        else:
            print(f"✅ 已连接到地图 {self.config.town}")
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"✅ 找到 {len(self.spawn_points)} 个生成点")
        
        # 初始化可视化器
        self.world_visualizer = CarlaWorldVisualizer(self.world)
        
        # 初始化同步模式管理器
        sync_config = SyncModeConfig(simulation_fps=self.config.simulation_fps)
        self._sync_manager = SyncModeManager(self.world, sync_config)
        self._lifecycle_helper = ResourceLifecycleHelper(self._sync_manager)
        
        # 初始化路径规划器
        if AGENTS_AVAILABLE:
            try:
                self.route_planner = GlobalRoutePlanner(
                    self.world.get_map(), sampling_resolution=2.0
                )
                print("✅ 路径规划器初始化成功")
            except Exception as e:
                print(f"⚠️ 路径规划器初始化失败: {e}")
    
    def visualize_spawn_points(self, duration: float = None) -> Tuple[float, float]:
        """
        可视化所有生成点
        
        返回:
            Tuple[float, float]: (开始时间, 持续时间)
        """
        duration = duration or self.DEFAULT_MARKER_DURATION
        
        if self.world_visualizer:
            return self.world_visualizer.visualize_spawn_points(duration)
        return time.time(), 0
    
    def get_user_route(self) -> Optional[Tuple[int, int]]:
        """获取用户输入的起点和终点"""
        print(f"\n可用索引范围: 0 到 {len(self.spawn_points) - 1}")
        print("输入 'q' 退出\n")
        
        while True:
            try:
                start_input = input("请输入起点索引: ").strip()
                if start_input.lower() in ['q', 'quit']:
                    return None
                start_idx = int(start_input)
                
                if not (0 <= start_idx < len(self.spawn_points)):
                    print(f"❌ 索引无效！范围: 0-{len(self.spawn_points)-1}")
                    continue
                
                end_input = input("请输入终点索引: ").strip()
                if end_input.lower() in ['q', 'quit']:
                    return None
                end_idx = int(end_input)
                
                if not (0 <= end_idx < len(self.spawn_points)):
                    print(f"❌ 索引无效！范围: 0-{len(self.spawn_points)-1}")
                    continue
                
                if start_idx == end_idx:
                    print(f"❌ 起点和终点不能相同！")
                    continue
                
                return start_idx, end_idx
                
            except ValueError:
                print("❌ 请输入数字！")
            except KeyboardInterrupt:
                return None
    
    def visualize_route(self, start_idx: int, end_idx: int, 
                        duration: float = None) -> bool:
        """
        可视化路径
        
        参数:
            start_idx: 起点索引
            end_idx: 终点索引
            duration: 显示持续时间
            
        返回:
            bool: 是否成功规划路径
        """
        duration = duration or self.DEFAULT_MARKER_DURATION
        
        if self.world_visualizer:
            return self.world_visualizer.visualize_route(
                start_idx, end_idx, self.route_planner, duration
            )
        return False
    
    def wait_for_markers(self, duration: float = None):
        """等待标记消失（带进度条）"""
        duration = duration or self.DEFAULT_MARKER_DURATION
        
        if self.world_visualizer:
            self.world_visualizer.wait_for_markers_to_clear(
                duration, "等待可视化标记消失"
            )
        else:
            print(f"\n⏳ 等待 {duration} 秒...")
            time.sleep(duration)
    
    def wait_for_start(self) -> Optional[bool]:
        """等待用户确认开始"""
        print("\n请选择:")
        print("  '开始' 或 's' → 开始收集")
        print("  '重选' 或 'r' → 重新选择路线")
        print("  'q' → 退出")
        
        while True:
            try:
                cmd = input("\n👉 ").strip().lower()
                
                if cmd in ['开始', 'start', 's']:
                    return True
                elif cmd in ['重选', 'reselect', 'r']:
                    return False
                elif cmd in ['q', 'quit']:
                    return None
                else:
                    print("❌ 无效命令！")
            except KeyboardInterrupt:
                return None
    
    def collect_data(self, start_idx: int, end_idx: int,
                     num_frames: int = 10000, save_path: str = './carla_data') -> bool:
        """收集数据"""
        print(f"\n📊 开始数据收集")
        print(f"  起点: {start_idx}, 终点: {end_idx}")
        print(f"  最大帧数: {num_frames}")
        print(f"  保存路径: {save_path}")
        
        self.collector = CommandBasedCollector(self.config)
        
        # 复用连接
        self.collector.client = self.client
        self.collector.world = self.world
        self.collector.blueprint_library = self.world.get_blueprint_library()
        
        # 传递同步模式管理器和资源生命周期辅助
        self.collector._sync_manager = self._sync_manager
        self.collector._lifecycle_helper = self._lifecycle_helper
        
        # 【v2.0】使用 ensure_sync_mode 确保同步模式
        if self._sync_manager is not None:
            if not self._sync_manager.ensure_sync_mode():
                print("⚠️ 无法启用同步模式")
                return False
        else:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.config.simulation_fps
                self.world.apply_settings(settings)
        
        try:
            if not self.collector.spawn_vehicle(start_idx, end_idx):
                print("❌ 无法生成车辆！")
                return False
            
            self.collector.setup_camera()
            self.collector.setup_collision_sensor()
            time.sleep(1.0)
            
            self.collector.collect_data_interactive(
                max_frames=num_frames,
                save_path=save_path,
                visualize=True
            )
            
            return True
            
        except Exception as e:
            print(f"❌ 收集出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup_collector()
    
    def _cleanup_collector(self):
        """清理收集器资源（使用统一的 actor_utils）"""
        if self.collector is None:
            return
        
        print("\n正在清理...")
        
        try:
            self.collector.agent = None
        except:
            pass
        
        # 收集需要销毁的传感器（只收集有效的）
        sensors = []
        if hasattr(self.collector, 'collision_sensor') and self.collector.collision_sensor:
            if is_actor_alive(self.collector.collision_sensor):
                sensors.append(self.collector.collision_sensor)
        if self.collector.camera:
            if is_actor_alive(self.collector.camera):
                sensors.append(self.collector.camera)
        
        # 检查车辆是否有效
        vehicle_to_destroy = None
        if self.collector.vehicle and is_actor_alive(self.collector.vehicle):
            vehicle_to_destroy = self.collector.vehicle
        
        # 使用 ResourceLifecycleHelper 安全清理资源
        if self._lifecycle_helper is not None:
            self._lifecycle_helper.destroy_all_safe(
                sensors=sensors,
                vehicle=vehicle_to_destroy,
                restore_sync=False
            )
        else:
            # 降级方案：使用统一的 actor_utils
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                time.sleep(0.3)
            except:
                pass
            
            # 使用统一的资源销毁工具
            destroy_all_resources(
                client=None,
                sensors=sensors,
                vehicle=vehicle_to_destroy,
                wait_time=0.5,
                silent=True
            )
        
        # 清理引用
        try:
            self.collector.collision_sensor = None
            self.collector.camera = None
            self.collector.vehicle = None
        except:
            pass
        
        print("✅ 清理完成")
    
    def run(self, num_frames: int = 10000, save_path: str = './carla_data'):
        """
        运行交互式收集流程
        
        流程：
        1. 查看所有生成点（彩色柱体+索引数字）
        2. 输入起点索引 -> 输入终点索引
        3. 规划路径并显示标记
        4. 倒计时等待标记消失
        5. 确认后开始收集
        6. 收集完成后选择继续或退出
        """
        try:
            self.connect()
            
            while True:
                # 显示生成点
                self.visualize_spawn_points()
                
                # 获取用户选择的路线
                route = self.get_user_route()
                if route is None:
                    print("\n👋 退出")
                    break
                
                start_idx, end_idx = route
                
                # 可视化路径
                route_valid = self.visualize_route(start_idx, end_idx)
                
                if not route_valid:
                    print("⚠️ 路径规划失败，请重新选择")
                    continue
                
                # 等待标记消失（带进度条）
                self.wait_for_markers()
                
                # 等待用户确认
                start_cmd = self.wait_for_start()
                
                if start_cmd is None:
                    break
                elif start_cmd is False:
                    continue
                
                # 开始收集数据
                self.collect_data(start_idx, end_idx, num_frames, save_path)
                
                # 询问是否继续
                print("\n是否继续收集？(y/n)")
                try:
                    if input().strip().lower() not in ['y', 'yes', '']:
                        break
                except KeyboardInterrupt:
                    break
            
            print("\n✅ 收集结束")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 收到中断信号，正在退出...")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 【v2.0】使用 ensure_async_mode 确保异步模式
            if self._sync_manager is not None:
                try:
                    self._sync_manager.ensure_async_mode()
                    print("✅ 已恢复CARLA异步模式")
                except:
                    pass
            elif self.world is not None:
                try:
                    settings = self.world.get_settings()
                    if settings.synchronous_mode:
                        settings.synchronous_mode = False
                        self.world.apply_settings(settings)
                        print("✅ 已恢复CARLA异步模式")
                except:
                    pass
