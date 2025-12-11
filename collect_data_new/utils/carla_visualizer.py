#!/usr/bin/env python
# coding=utf-8
"""
CARLA 世界可视化工具

提供生成点标记、路径可视化、倒计时等功能。
"""

import time
import colorsys
from typing import List, Optional, Tuple

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class SpawnPointVisualizer:
    """生成点可视化器"""
    
    def __init__(self, debug_helper):
        """
        初始化
        
        参数:
            debug_helper: carla.DebugHelper 实例
        """
        self.debug = debug_helper
    
    def visualize_all(self, spawn_points: List, duration: float = 30.0):
        """
        可视化所有生成点
        
        参数:
            spawn_points: 生成点列表
            duration: 显示持续时间（秒）
        """
        if not CARLA_AVAILABLE:
            return
        
        print(f"\n🎨 可视化 {len(spawn_points)} 个生成点...")
        
        for idx, spawn_point in enumerate(spawn_points):
            location = spawn_point.location
            color = self._get_rainbow_color(idx, len(spawn_points))
            
            # 绘制柱体
            self.debug.draw_arrow(
                begin=carla.Location(x=location.x, y=location.y, z=location.z + 0.1),
                end=carla.Location(x=location.x, y=location.y, z=location.z + 3.0),
                thickness=0.15, arrow_size=0.0, color=color, life_time=duration
            )
            
            # 绘制索引数字
            self.debug.draw_string(
                location=carla.Location(x=location.x, y=location.y, z=location.z + 3.5),
                text=f"{idx}", draw_shadow=True,
                color=carla.Color(255, 255, 255), life_time=duration
            )
        
        print(f"✅ 生成点可视化完成！")
    
    def _get_rainbow_color(self, idx: int, total: int) -> 'carla.Color':
        """获取彩虹色"""
        hue = (idx / total) * 360
        r, g, b = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
        return carla.Color(int(r * 255), int(g * 255), int(b * 255), 255)


class RouteVisualizer:
    """路径可视化器"""
    
    def __init__(self, debug_helper):
        """
        初始化
        
        参数:
            debug_helper: carla.DebugHelper 实例
        """
        self.debug = debug_helper
    
    def visualize_route(self, start_point, end_point, route: Optional[List] = None,
                        duration: float = 30.0):
        """
        可视化路径
        
        参数:
            start_point: 起点 Transform
            end_point: 终点 Transform
            route: 路径点列表 [(waypoint, road_option), ...]
            duration: 显示持续时间（秒）
        """
        if not CARLA_AVAILABLE:
            return
        
        # 起点标记（绿色）
        self._draw_endpoint_marker(start_point.location, carla.Color(0, 255, 0), 
                                   "START", duration)
        
        # 终点标记（红色）
        self._draw_endpoint_marker(end_point.location, carla.Color(255, 0, 0), 
                                   "END", duration)
        
        # 起点到终点的直线（黄色）
        self.debug.draw_line(
            begin=carla.Location(x=start_point.location.x, y=start_point.location.y, 
                               z=start_point.location.z + 2.0),
            end=carla.Location(x=end_point.location.x, y=end_point.location.y, 
                             z=end_point.location.z + 2.0),
            thickness=0.1, color=carla.Color(255, 255, 0), life_time=duration
        )
        
        # 绘制路径（蓝色）
        if route:
            self._draw_route_path(route, duration)
    
    def _draw_endpoint_marker(self, location, color, text: str, duration: float):
        """绘制端点标记"""
        # 柱体
        self.debug.draw_arrow(
            begin=carla.Location(x=location.x, y=location.y, z=location.z + 0.1),
            end=carla.Location(x=location.x, y=location.y, z=location.z + 8.0),
            thickness=0.3, arrow_size=0.0, color=color, life_time=duration
        )
        
        # 文字标签
        self.debug.draw_string(
            location=carla.Location(x=location.x, y=location.y, z=location.z + 9.0),
            text=text, draw_shadow=True,
            color=carla.Color(255, 255, 255), life_time=duration
        )
    
    def _draw_route_path(self, route: List, duration: float):
        """绘制路径线"""
        for i in range(len(route) - 1):
            wp1 = route[i][0].transform.location
            wp2 = route[i+1][0].transform.location
            
            self.debug.draw_line(
                begin=carla.Location(x=wp1.x, y=wp1.y, z=wp1.z + 1.0),
                end=carla.Location(x=wp2.x, y=wp2.y, z=wp2.z + 1.0),
                thickness=0.2, color=carla.Color(0, 150, 255), life_time=duration
            )


class CountdownTimer:
    """倒计时器"""
    
    def __init__(self, total_seconds: float):
        """
        初始化
        
        参数:
            total_seconds: 总倒计时秒数
        """
        self.total_seconds = total_seconds
        self.start_time = None
    
    def start(self):
        """开始倒计时"""
        self.start_time = time.time()
    
    def get_remaining(self) -> float:
        """获取剩余时间（秒）"""
        if self.start_time is None:
            return self.total_seconds
        elapsed = time.time() - self.start_time
        return max(0, self.total_seconds - elapsed)
    
    def is_finished(self) -> bool:
        """是否已完成"""
        return self.get_remaining() <= 0
    
    def get_progress(self) -> float:
        """获取进度（0-1）"""
        if self.start_time is None:
            return 0
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.total_seconds)
    
    def wait_with_progress(self, message: str = "等待中"):
        """
        带进度条的等待
        
        参数:
            message: 显示的消息
        """
        self.start()
        
        while not self.is_finished():
            remaining = self.get_remaining()
            progress = self.get_progress()
            
            # 绘制进度条
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r⏳ {message}: [{bar}] {progress*100:.0f}% ({remaining:.1f}s)", 
                  end='', flush=True)
            
            time.sleep(0.1)
        
        print(f"\r✅ {message}: [{'█' * 40}] 100%                    ")


class CarlaWorldVisualizer:
    """CARLA 世界可视化器（整合所有可视化功能）"""
    
    def __init__(self, world):
        """
        初始化
        
        参数:
            world: carla.World 实例
        """
        self.world = world
        self.debug = world.debug if world else None
        
        self.spawn_visualizer = SpawnPointVisualizer(self.debug) if self.debug else None
        self.route_visualizer = RouteVisualizer(self.debug) if self.debug else None
    
    def visualize_spawn_points(self, duration: float = 30.0) -> Tuple[float, float]:
        """
        可视化所有生成点
        
        返回:
            Tuple[float, float]: (开始时间, 持续时间)
        """
        if not self.spawn_visualizer:
            return time.time(), 0
        
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_visualizer.visualize_all(spawn_points, duration)
        return time.time(), duration
    
    def visualize_route(self, start_idx: int, end_idx: int, 
                        route_planner=None, duration: float = 30.0) -> bool:
        """
        可视化路径
        
        参数:
            start_idx: 起点索引
            end_idx: 终点索引
            route_planner: GlobalRoutePlanner 实例
            duration: 显示持续时间
            
        返回:
            bool: 是否成功规划路径
        """
        import math
        
        if not self.route_visualizer:
            return False
        
        spawn_points = self.world.get_map().get_spawn_points()
        
        if start_idx >= len(spawn_points) or end_idx >= len(spawn_points):
            print(f"❌ 索引超出范围！")
            return False
        
        start_point = spawn_points[start_idx]
        end_point = spawn_points[end_idx]
        
        # 计算直线距离
        dx = end_point.location.x - start_point.location.x
        dy = end_point.location.y - start_point.location.y
        straight_distance = math.sqrt(dx**2 + dy**2)
        
        print(f"\n📏 起点 #{start_idx}: ({start_point.location.x:.2f}, {start_point.location.y:.2f})")
        print(f"📏 终点 #{end_idx}: ({end_point.location.x:.2f}, {end_point.location.y:.2f})")
        print(f"📏 直线距离: {straight_distance:.2f} 米")
        
        # 尝试规划路径
        route = None
        if route_planner:
            try:
                route = route_planner.trace_route(
                    start_point.location, end_point.location
                )
                if route and len(route) > 0:
                    # 计算实际路径长度
                    route_distance = 0.0
                    for i in range(len(route) - 1):
                        wp1 = route[i][0].transform.location
                        wp2 = route[i+1][0].transform.location
                        route_distance += wp1.distance(wp2)
                    
                    print(f"\n✅ 路径规划成功！")
                    print(f"📏 实际路径长度: {route_distance:.2f} 米")
                    print(f"📏 路点数量: {len(route)} 个")
                    print(f"📊 路径/直线比: {route_distance/straight_distance:.2f}x")
                    
                    # 路线质量评估
                    print(f"\n📝 路线评估:")
                    if straight_distance < 50:
                        print(f"   ⚠️  距离较短 ({straight_distance:.0f}m)")
                    elif straight_distance < 150:
                        print(f"   ✅ 距离适中 ({straight_distance:.0f}m)")
                    elif straight_distance < 300:
                        print(f"   ✅ 距离较长 ({straight_distance:.0f}m)")
                    else:
                        print(f"   ⭐ 距离很长 ({straight_distance:.0f}m)")
                    
                    ratio = route_distance / straight_distance
                    if ratio > 2.5:
                        print(f"   ⚠️  路径曲折度较高 ({ratio:.2f}x)")
                    elif ratio > 1.5:
                        print(f"   ✅ 路径有适当的转弯 ({ratio:.2f}x)")
                    else:
                        print(f"   ✅ 路径较为直接 ({ratio:.2f}x)")
                else:
                    print(f"\n❌ 路径规划失败！这两个点之间可能不可达")
                    return False
            except Exception as e:
                print(f"\n⚠️ 路径规划失败: {e}")
                return False
        
        self.route_visualizer.visualize_route(start_point, end_point, route, duration)
        return route is not None
    
    def wait_for_markers_to_clear(self, duration: float, message: str = "等待标记消失"):
        """
        等待标记消失（带进度条）
        
        参数:
            duration: 等待时间（秒）
            message: 显示的消息
        """
        timer = CountdownTimer(duration)
        timer.wait_with_progress(message)
