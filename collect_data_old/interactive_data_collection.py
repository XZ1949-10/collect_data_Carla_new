#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant
日期: 2025-11-03
说明: 交互式CARLA数据收集启动器
      整合生成点可视化和数据收集功能，提供友好的交互式界面
'''

import glob
import os
import sys
import time
import numpy as np
import colorsys

# 添加CARLA Python API路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# 导入数据收集器
from command_based_data_collection import CommandBasedDataCollector

# 尝试导入agents模块（用于路径规划）
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    print(f"⚠️  警告: 无法导入agents模块，路径可视化功能将受限: {e}")


class InteractiveDataCollector:
    """交互式数据收集器"""
    
    def __init__(self, host='localhost', port=2000, town='Town01',
                 ignore_traffic_lights=True, ignore_signs=True,
                 ignore_vehicles_percentage=80, target_speed=10.0, simulation_fps=20,
                 noise_enabled=False, lateral_noise=True, longitudinal_noise=False,
                 noise_ratio=0.4, max_steer_offset=0.35, max_throttle_offset=0.2,
                 noise_modes=None):
        """
        初始化交互式收集器
        
        参数:
            host (str): CARLA服务器地址
            port (int): CARLA服务器端口
            town (str): 地图名称
            ignore_traffic_lights (bool): 是否忽略红绿灯
            ignore_signs (bool): 是否忽略停车标志
            ignore_vehicles_percentage (int): 忽略其他车辆的百分比
            target_speed (float): 目标速度 (km/h)
            simulation_fps (int): 模拟帧率
            noise_enabled (bool): 是否启用噪声注入
            lateral_noise (bool): 是否启用横向噪声（转向）
            longitudinal_noise (bool): 是否启用纵向噪声（油门/刹车）
            noise_ratio (float): 噪声时间占比 (0-1)
            max_steer_offset (float): 最大转向偏移 (0-1)
            max_throttle_offset (float): 最大油门偏移 (0-1)
            noise_modes (dict): 噪声模式配置
        """
        self.host = host
        self.port = port
        self.town = town
        
        # 交通规则配置
        self.ignore_traffic_lights = ignore_traffic_lights
        self.ignore_signs = ignore_signs
        self.ignore_vehicles_percentage = ignore_vehicles_percentage
        
        # 速度和帧率配置
        self.target_speed = target_speed
        self.simulation_fps = simulation_fps
        
        # ========== 噪声配置 ==========
        self.noise_enabled = noise_enabled
        self.lateral_noise_enabled = lateral_noise
        self.longitudinal_noise_enabled = longitudinal_noise
        self.noise_ratio = noise_ratio
        self.max_steer_offset = max_steer_offset
        self.max_throttle_offset = max_throttle_offset
        self.noise_modes = noise_modes
        
        # CARLA对象
        self.client = None
        self.world = None
        self.debug = None
        self.spawn_points = []
        
        # 数据收集器（延迟初始化）
        self.collector = None
        
        # 路径规划器
        self.route_planner = None
        
    def connect(self):
        """连接到CARLA服务器"""
        print("\n" + "="*70)
        print("🚗 CARLA 交互式数据收集器")
        print("="*70)
        print(f"正在连接到CARLA服务器 {self.host}:{self.port}...")
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        
        # 获取当前世界，避免重新加载导致视角重置
        self.world = self.client.get_world()
        current_map_name = self.world.get_map().name.split('/')[-1]
        
        # 只有在地图不同时才重新加载
        if current_map_name != self.town:
            print(f"当前地图: {current_map_name}, 需要切换到: {self.town}")
            print(f"正在加载地图 {self.town}...")
            self.world = self.client.load_world(self.town)
        else:
            print(f"✅ 已连接到地图 {self.town}（保持当前视角）")
        
        # 获取debug helper
        self.debug = self.world.debug
        
        # 获取生成点
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"✅ 成功连接！共找到 {len(self.spawn_points)} 个生成点")
        
        # 显示交通规则配置
        print(f"\n📋 交通规则配置:")
        print(f"  • 忽略红绿灯: {'✅ 是' if self.ignore_traffic_lights else '❌ 否'}")
        print(f"  • 忽略停车标志: {'✅ 是' if self.ignore_signs else '❌ 否'}")
        print(f"  • 忽略其他车辆: {self.ignore_vehicles_percentage}%")
        print()
        
        # 初始化路径规划器
        if AGENTS_AVAILABLE:
            try:
                self.route_planner = GlobalRoutePlanner(
                    self.world.get_map(), 
                    sampling_resolution=2.0
                )
                print("✅ 路径规划器初始化成功")
            except Exception as e:
                print(f"⚠️  路径规划器初始化失败: {e}")
                self.route_planner = None
        
    def visualize_all_spawn_points(self, duration=30.0):
        """
        可视化所有生成点（彩虹渐变）
        
        参数:
            duration (float): 显示持续时间（秒），默认30秒（统一显示时间）
        """
        print("\n" + "="*70)
        print("🎨 步骤 1/4: 可视化所有生成点")
        print("="*70)
        print(f"正在绘制 {len(self.spawn_points)} 个生成点...")
        print("提示: 彩色柱体标记生成点位置，白色数字是索引")
        print(f"⏰ 注意: 生成点（包括索引数字）将在{duration:.0f}秒后自动消失（数据收集前会自动清除）\n")
        
        def get_color_by_index(idx, total):
            """根据索引返回渐变颜色"""
            hue = (idx / total) * 360
            r, g, b = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
            return carla.Color(int(r * 255), int(g * 255), int(b * 255), 255)
        
        # 绘制每个生成点
        for idx, spawn_point in enumerate(self.spawn_points):
            location = spawn_point.location
            color = get_color_by_index(idx, len(self.spawn_points))
            
            # 绘制柱体
            begin = carla.Location(x=location.x, y=location.y, z=location.z + 0.1)
            end = carla.Location(x=location.x, y=location.y, z=location.z + 30.0)
            
            self.debug.draw_arrow(
                begin=begin,
                end=end,
                thickness=0.15,
                arrow_size=0.0,
                color=color,
                life_time=duration
            )
            
            # 绘制索引编号
            text_location = carla.Location(x=location.x, y=location.y, z=location.z + 3.5)
            self.debug.draw_string(
                location=text_location,
                text=f"{idx}",
                draw_shadow=True,
                color=carla.Color(255, 255, 255),
                life_time=duration
            )
            
            # 绘制方向箭头
            rotation = spawn_point.rotation
            forward = rotation.get_forward_vector()
            arrow_begin = carla.Location(x=location.x, y=location.y, z=location.z + 0.5)
            arrow_end = carla.Location(
                x=location.x + forward.x * 2.0,
                y=location.y + forward.y * 2.0,
                z=location.z + 0.5
            )
            
            self.debug.draw_arrow(
                begin=arrow_begin,
                end=arrow_end,
                thickness=0.1,
                arrow_size=0.3,
                color=carla.Color(255, 255, 0),
                life_time=duration
            )
            
            # 显示进度
            if (idx + 1) % 20 == 0 or idx == len(self.spawn_points) - 1:
                progress = (idx + 1) / len(self.spawn_points) * 100
                print(f"   进度: {progress:.1f}% ({idx + 1}/{len(self.spawn_points)})")
        
        print(f"\n✅ 所有生成点已可视化！")
        print(f"💡 图例：彩色柱体=位置 | 白色数字=索引 | 黄色箭头=朝向")
        print(f"⏰ 所有标记将在{duration:.0f}秒后自动消失，请尽快选择起点和终点\n")
        
        # 返回绘制时间，供后续判断是否需要等待消失
        return time.time(), duration
        
    def get_user_input_route(self):
        """
        获取用户输入的起点和终点
        
        返回:
            tuple: (start_idx, end_idx) 或 None（如果用户想退出）
        """
        print("\n" + "="*70)
        print("📍 步骤 2/4: 选择起点和终点")
        print("="*70)
        print(f"可用索引范围: 0 到 {len(self.spawn_points) - 1}")
        print("提示: 输入 'q' 或 'quit' 退出程序\n")
        
        while True:
            try:
                # 获取起点
                start_input = input("请输入起点索引: ").strip()
                if start_input.lower() in ['q', 'quit', 'exit']:
                    return None
                    
                start_idx = int(start_input)
                
                # 验证起点
                if start_idx < 0 or start_idx >= len(self.spawn_points):
                    print(f"❌ 起点索引无效！请输入 0-{len(self.spawn_points)-1} 之间的数字")
                    continue
                
                # 获取终点
                end_input = input("请输入终点索引: ").strip()
                if end_input.lower() in ['q', 'quit', 'exit']:
                    return None
                    
                end_idx = int(end_input)
                
                # 验证终点
                if end_idx < 0 or end_idx >= len(self.spawn_points):
                    print(f"❌ 终点索引无效！请输入 0-{len(self.spawn_points)-1} 之间的数字")
                    continue
                
                if start_idx == end_idx:
                    print(f"❌ 起点和终点不能相同！")
                    continue
                
                return start_idx, end_idx
                
            except ValueError:
                print("❌ 输入格式错误！请输入数字索引")
            except KeyboardInterrupt:
                print("\n\n⚠️  收到中断信号")
                return None
    
    def visualize_and_plan_route(self, start_idx, end_idx, duration=30.0):
        """
        可视化并规划路径（所有标记统一显示时间）
        
        参数:
            start_idx (int): 起点索引
            end_idx (int): 终点索引
            duration (float): 所有标记显示时间（秒），默认30秒（与生成点统一）
            
        返回:
            tuple: (是否成功, 路径数据, 标记时间, duration) 或 (False, None, None, None)
        """
        print("\n" + "="*70)
        print("🗺️  步骤 3/4: 规划并可视化路径")
        print("="*70)
        
        start_point = self.spawn_points[start_idx]
        end_point = self.spawn_points[end_idx]
        
        # 计算直线距离
        dx = end_point.location.x - start_point.location.x
        dy = end_point.location.y - start_point.location.y
        straight_distance = np.sqrt(dx**2 + dy**2)
        
        print(f"📏 起点 #{start_idx}: ({start_point.location.x:.2f}, {start_point.location.y:.2f})")
        print(f"📏 终点 #{end_idx}: ({end_point.location.x:.2f}, {end_point.location.y:.2f})")
        print(f"📏 直线距离: {straight_distance:.2f} 米\n")
        
        # 记录标记开始时间（用于后续计算等待时间）
        markers_draw_time = time.time()
        
        # 标记起点（绿色大柱体）
        self.debug.draw_arrow(
            begin=carla.Location(x=start_point.location.x, y=start_point.location.y, 
                               z=start_point.location.z + 0.1),
            end=carla.Location(x=start_point.location.x, y=start_point.location.y, 
                             z=start_point.location.z + 8.0),
            thickness=0.3,
            arrow_size=0.0,
            color=carla.Color(0, 255, 0),  # 绿色
            life_time=duration
        )
        
        self.debug.draw_string(
            location=carla.Location(x=start_point.location.x, y=start_point.location.y, 
                                  z=start_point.location.z + 9.0),
            text=f"起点 #{start_idx}",
            draw_shadow=True,
            color=carla.Color(0, 255, 0),
            life_time=duration
        )
        
        # 标记终点（红色大柱体）
        self.debug.draw_arrow(
            begin=carla.Location(x=end_point.location.x, y=end_point.location.y, 
                               z=end_point.location.z + 0.1),
            end=carla.Location(x=end_point.location.x, y=end_point.location.y, 
                             z=end_point.location.z + 8.0),
            thickness=0.3,
            arrow_size=0.0,
            color=carla.Color(255, 0, 0),  # 红色
            life_time=duration
        )
        
        self.debug.draw_string(
            location=carla.Location(x=end_point.location.x, y=end_point.location.y, 
                                  z=end_point.location.z + 9.0),
            text=f"终点 #{end_idx}",
            draw_shadow=True,
            color=carla.Color(255, 0, 0),
            life_time=duration
        )
        
        # 绘制直线连接（黄色虚线）
        num_segments = max(int(straight_distance / 10), 1)
        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments
            
            loc1 = carla.Location(
                x=start_point.location.x + dx * t1,
                y=start_point.location.y + dy * t1,
                z=start_point.location.z + 2.0
            )
            loc2 = carla.Location(
                x=start_point.location.x + dx * t2,
                y=start_point.location.y + dy * t2,
                z=start_point.location.z + 2.0
            )
            
            if i % 2 == 0:  # 虚线效果
                self.debug.draw_line(
                    begin=loc1,
                    end=loc2,
                    thickness=0.1,
                    color=carla.Color(255, 255, 0),
                    life_time=duration
                )
        
        # 尝试规划实际路径
        if not AGENTS_AVAILABLE or self.route_planner is None:
            print("⚠️  路径规划器不可用，只显示直线距离")
            print("   建议安装CARLA agents模块以获得完整功能\n")
            return False, None, None, None
        
        try:
            print("🚗 正在计算导航路径...")
            route = self.route_planner.trace_route(start_point.location, end_point.location)
            
            if not route or len(route) == 0:
                print("❌ 路径规划失败！这两个点之间可能不可达")
                print("   请重新选择起点和终点\n")
                return False, None, None, None
            
            # 计算实际路径长度
            route_distance = 0.0
            for i in range(len(route) - 1):
                wp1 = route[i][0].transform.location
                wp2 = route[i+1][0].transform.location
                route_distance += wp1.distance(wp2)
            
            print(f"✅ 路径规划成功！")
            print(f"📏 实际路径长度: {route_distance:.2f} 米")
            print(f"📏 路点数量: {len(route)} 个")
            print(f"📊 路径/直线比: {route_distance/straight_distance:.2f}x\n")
            
            # 评估路线质量
            print(f"📝 路线评估:")
            if straight_distance < 50:
                print(f"   ⚠️  距离较短 ({straight_distance:.0f}m)")
            elif straight_distance < 150:
                print(f"   ✅ 距离适中 ({straight_distance:.0f}m)")
            elif straight_distance < 300:
                print(f"   ✅ 距离较长 ({straight_distance:.0f}m)")
            else:
                print(f"   ⭐ 距离很长 ({straight_distance:.0f}m)")
            
            if route_distance / straight_distance > 2.5:
                print(f"   ⚠️  路径曲折度较高")
            elif route_distance / straight_distance > 1.5:
                print(f"   ✅ 路径有适当的转弯")
            else:
                print(f"   ✅ 路径较为直接")
            
            # 绘制蓝色导航路径（使用统一的life_time）
            print(f"\n🎨 正在绘制蓝色导航路径...")
            
            for i in range(len(route) - 1):
                waypoint1 = route[i][0]
                waypoint2 = route[i+1][0]
                
                loc1 = carla.Location(
                    x=waypoint1.transform.location.x,
                    y=waypoint1.transform.location.y,
                    z=waypoint1.transform.location.z + 1.0
                )
                loc2 = carla.Location(
                    x=waypoint2.transform.location.x,
                    y=waypoint2.transform.location.y,
                    z=waypoint2.transform.location.z + 1.0
                )
                
                self.debug.draw_line(
                    begin=loc1,
                    end=loc2,
                    thickness=0.2,
                    color=carla.Color(0, 150, 255),  # 蓝色路径
                    life_time=duration  # 统一显示时间
                )
            
            print(f"✅ 已绘制蓝色导航路径（{len(route)-1} 段）")
            
            # 保存路径供后续使用（用于设置到 LocalPlanner）
            self._current_route = route
            
            print(f"\n💡 说明：")
            print(f"   - 🟢 绿色高柱体 = 起点")
            print(f"   - 🔴 红色高柱体 = 终点")
            print(f"   - 🟡 黄色虚线 = 直线距离（参考）")
            print(f"   - 🔵 蓝色实线 = 导航路径")
            print(f"\n⏰ 统一显示时间：")
            print(f"   → 所有标记（生成点、索引数字、起点/终点、黄线、蓝色路径）")
            print(f"   → 将在{duration:.0f}秒后同时消失")
            print(f"\n✨ 接下来：")
            print(f"   → 系统将自动开始倒计时{duration:.0f}秒，等待所有标记消失")
            print(f"   → 请利用这段时间在CARLA中仔细观察路线")
            print(f"   → 倒计时结束后，你可以决定是否开始收集数据\n")
            
            # 返回路径数据和时间信息
            return True, route, markers_draw_time, duration
            
        except Exception as e:
            print(f"❌ 路径规划失败: {e}")
            print("   请重新选择起点和终点\n")
            return False, None, None, None
    
    def wait_for_start_command(self):
        """
        等待用户输入"开始"命令
        
        返回:
            bool: True=开始收集, False=重新选择路线, None=退出
        """
        print("\n" + "="*70)
        print("⏸️  步骤 5/6: 等待确认")
        print("="*70)
        print("\n✅ 所有可视化标记已清除完毕，CARLA画面已干净")
        print("💡 现在你可以决定：")
        print("   - 如果对路线满意 → 输入'开始'立即收集数据")
        print("   - 如果想换路线 → 输入'重选'重新选择起点终点")
        print("   - 如果想退出 → 输入'q'\n")
        print("📋 可用命令:")
        print("  ✅ '开始' 或 'start' → 立即开始数据收集")
        print("  🔄 '重选' 或 'reselect' → 重新选择路线")
        print("  ❌ 'q' 或 'quit' → 退出程序\n")
        
        while True:
            try:
                command = input("👉 请输入命令: ").strip().lower()
                
                if command in ['开始', 'start', 's']:
                    print("✅ 收到'开始'命令，正在准备...")
                    return True
                elif command in ['重选', 'reselect', 'r', '重新选择']:
                    print("🔄 收到'重选'命令，返回路线选择...")
                    return False
                elif command in ['q', 'quit', 'exit', '退出']:
                    print("👋 收到退出命令...")
                    return None
                else:
                    print(f"❌ 无效命令：'{command}' ！请输入 '开始'、'重选' 或 'q'\n")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  收到中断信号")
                return None
    
    def collect_data(self, start_idx, end_idx, num_frames=10000, 
                    save_path='./carla_data', visualize=False):
        """
        收集数据（基于命令分段的交互式收集）
        
        新功能：
        1. 开始时询问是否保存初始命令段
        2. 命令变化时暂停并询问是否保存
        3. 每段数据按200条切片保存
        4. 文件名包含命令类型
        
        参数:
            start_idx (int): 起点索引
            end_idx (int): 终点索引
            num_frames (int): 最大帧数
            save_path (str): 保存路径
            visualize (bool): 是否启用可视化（交互式模式下强制启用）
        """
        print("\n" + "="*70)
        print("📊 步骤 6/6: 开始基于命令的交互式数据收集")
        print("="*70)
        
        # 交互式模式下强制启用可视化
        visualize = True
        
        # 创建基于命令的数据收集器（传递所有配置参数，包括噪声参数）
        self.collector = CommandBasedDataCollector(
            host=self.host,
            port=self.port,
            town=self.town,
            ignore_traffic_lights=self.ignore_traffic_lights,
            ignore_signs=self.ignore_signs,
            ignore_vehicles_percentage=self.ignore_vehicles_percentage,
            target_speed=self.target_speed,
            simulation_fps=self.simulation_fps,
            noise_enabled=self.noise_enabled,
            lateral_noise=self.lateral_noise_enabled,
            longitudinal_noise=self.longitudinal_noise_enabled,
            noise_ratio=self.noise_ratio,
            max_steer_offset=self.max_steer_offset,
            max_throttle_offset=self.max_throttle_offset,
            noise_modes=self.noise_modes
        )
        
        # 复用已有的连接
        self.collector.client = self.client
        self.collector.world = self.world
        self.collector.blueprint_library = self.world.get_blueprint_library()
        
        # 设置同步模式
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.simulation_fps
            self.world.apply_settings(settings)
        
        print(f"配置:")
        print(f"  起点索引: {start_idx}")
        print(f"  终点索引: {end_idx}")
        print(f"  最大帧数: {num_frames}")
        print(f"  保存路径: {save_path}")
        print(f"  目标速度: {self.target_speed} km/h")
        print(f"  模拟帧率: {self.simulation_fps} FPS")
        print(f"  实时可视化: ✅ 已启用")
        print(f"  交通规则:")
        print(f"    • 忽略红绿灯: {'✅ 是' if self.ignore_traffic_lights else '❌ 否'}")
        print(f"    • 忽略停车标志: {'✅ 是' if self.ignore_signs else '❌ 否'}")
        print(f"    • 忽略其他车辆: {self.ignore_vehicles_percentage}%")
        print(f"  噪声配置:")
        print(f"    • 噪声注入: {'✅ 启用' if self.noise_enabled else '❌ 禁用'}")
        if self.noise_enabled:
            print(f"    • 横向噪声: {'✅' if self.lateral_noise_enabled else '❌'} "
                  f"(ratio={self.noise_ratio}, max_offset={self.max_steer_offset})")
            print(f"    • 纵向噪声: {'✅' if self.longitudinal_noise_enabled else '❌'} "
                  f"(max_offset={self.max_throttle_offset})")
        print(f"  收集模式: 📋 简化的循环式收集")
        print(f"    • 询问是否收集当前命令段")
        print(f"    • 选择'保存' → 收集200帧 → 自动保存")
        print(f"    • 自动保存后继续询问下一段")
        print(f"    • 文件名使用选择时的命令类型\n")
        
        try:
            # 生成车辆（必须先生成车辆，才能创建局部规划器）
            if not self.collector.spawn_vehicle(start_idx, end_idx):
                print("❌ 无法生成车辆！")
                return False
            
            # 设置摄像头
            # 注意：spawn_vehicle() 已经创建了 BasicAgent，它内部有自己的 LocalPlanner
            # 不需要再创建额外的 local_planner
            self.collector.setup_camera()
            self.collector.setup_collision_sensor()  # 设置碰撞传感器
            
            # 等待传感器准备
            print("\n等待传感器准备...")
            time.sleep(1.0)
            
            # 噪声已在构造函数中配置，无需再次调用 configure_noise()
            
            # 开始交互式收集数据
            print("\n🎬 准备开始交互式数据收集...")
            print("="*70)
            print("💡 简化工作流程：")
            print("   1. 系统检测当前导航命令（Follow/Left/Right/Straight）")
            print("   2. 询问你是否收集该命令段")
            print("   3. 你选择：保存 / 跳过 / 停止")
            print("   4. 选择'保存' → 自动收集200帧 → 自动保存")
            print("   5. 保存完成后 → 自动询问下一段")
            print("   6. 循环执行，直到用户停止或到达终点")
            print("="*70)
            print()
            
            self.collector.collect_data_interactive(
                max_frames=num_frames,
                save_path=save_path,
                visualize=visualize
            )
            
            print("\n✅ 数据收集完成！")
            return True
            
        except Exception as e:
            print(f"\n❌ 数据收集出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理资源
            if self.collector:
                print("\n正在清理车辆和传感器...")
                try:
                    if self.collector.agent is not None:
                        self.collector.agent = None
                except:
                    pass
                
                try:
                    if self.collector.collision_sensor is not None:
                        self.collector.collision_sensor.stop()
                        self.collector.collision_sensor.destroy()
                except:
                    pass
                
                try:
                    if self.collector.camera is not None:
                        self.collector.camera.stop()
                        self.collector.camera.destroy()
                except:
                    pass
                    
                try:
                    if self.collector.vehicle is not None:
                        self.collector.vehicle.destroy()
                except:
                    pass
                
                # 恢复异步模式
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    self.world.apply_settings(settings)
                    print("✅ 已恢复CARLA异步模式（画面可正常运行）")
                except:
                    pass
                
                print("✅ 清理完成（保留路径可视化）")
    
    def run(self, num_frames=10000, save_path='./carla_data', visualize=False):
        """
        运行交互式数据收集流程
        
        参数:
            num_frames (int): 每次收集的帧数
            save_path (str): 数据保存路径
            visualize (bool): 是否启用实时可视化
        """
        try:
            # 步骤1: 连接CARLA
            self.connect()
            
            # 主循环：选择路线 -> 收集数据 -> 继续或退出
            while True:
                # 步骤2: 可视化所有生成点（每次循环都重新绘制）
                spawn_points_time, spawn_points_duration = self.visualize_all_spawn_points(duration=30.0)
                
                # 步骤3: 获取用户输入的起点和终点
                route_input = self.get_user_input_route()
                if route_input is None:
                    print("\n👋 用户选择退出")
                    break
                
                start_idx, end_idx = route_input
                
                # 步骤4: 规划并可视化路径（统一显示时间30秒）
                route_valid, route_data, markers_time, markers_duration = self.visualize_and_plan_route(
                    start_idx, end_idx, 
                    duration=30.0  # 所有标记统一显示30秒
                )
                
                if not route_valid:
                    print("⚠️  路径规划失败，请重新选择起点和终点")
                    continue
                
                # 步骤4.5: 立即开始倒计时等待所有标记消失
                print("\n" + "="*70)
                print("⏳ 步骤 4/5: 自动清除可视化标记")
                print("="*70)
                print("💡 请利用这段时间在CARLA中仔细观察路线规划")
                print("   倒计时结束后，你可以决定是否开始收集数据\n")
                
                # 计算需要等待的时间（所有标记统一显示时间）
                current_time = time.time()
                elapsed = current_time - markers_time
                remaining_time = markers_duration - elapsed
                
                if remaining_time > 0:
                        # 需要等待标记消失
                    wait_time = remaining_time + 3.0  # 额外等待3秒确保完全消失
                    print(f"⏳ 正在清除所有可视化标记（包括所有文字）...")
                    print(f"   所有标记已显示 {elapsed:.1f}秒")
                    print(f"   还需等待 {wait_time:.1f}秒确保全部消失...\n")
                        
                    # 倒计时显示（带进度条）
                    total_seconds = int(wait_time)
                    for i in range(total_seconds):
                        remaining = total_seconds - i
                        progress = (i / total_seconds) * 100
                        
                        # 绘制进度条（50个字符宽度）
                        bar_length = 50
                        filled_length = int(bar_length * i / total_seconds)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        
                        print(f"   🕐 [{bar}] {progress:.1f}% | 剩余: {remaining}秒     ", end='\r', flush=True)
                        time.sleep(1.0)
                    
                    # 最后显示100%完成
                    bar = '█' * 50
                    print(f"   ✅ [{bar}] 100.0% | 完成！             ")
                    print("\n✅ 所有可视化标记已完全消失（CARLA画面已清空）\n")
                else:
                        # 所有标记都已经消失了
                    print(f"\n✅ 太好了！所有可视化标记已自动消失")
                    print(f"✅ CARLA画面已清空，可以直接开始收集数据\n")
                
                # 步骤5: 倒计时完成，等待用户确认开始
                start_command = self.wait_for_start_command()
                
                if start_command is None:
                    print("\n👋 用户选择退出")
                    break
                elif start_command is False:
                    print("\n🔄 重新选择路线...")
                    continue
                
                # 步骤6: 开始收集数据
                success = self.collect_data(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_frames=num_frames,
                    save_path=save_path,
                    visualize=visualize
                )
                
                # 步骤7: 询问是否继续收集
                print("\n" + "="*70)
                print("✅ 本次收集完成")
                print("="*70)
                continue_input = input("是否继续收集下一条路线？(y/n): ").strip().lower()
                
                if continue_input not in ['y', 'yes', '是', 'continue']:
                    print("\n👋 结束数据收集")
                    break
                
                print("\n🔄 开始新的收集任务...\n")
            
            print("\n" + "="*70)
            print("📊 数据收集会话结束")
            print("="*70)
            print("提示: 生成点和路径标记将继续显示在CARLA中")
            print("      你可以在CARLA中自由观察或调整视角\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  收到中断信号，正在退出...")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保程序退出时恢复异步模式
            if self.world is not None:
                try:
                    settings = self.world.get_settings()
                    if settings.synchronous_mode:
                        settings.synchronous_mode = False
                        self.world.apply_settings(settings)
                        print("✅ 已恢复CARLA异步模式")
                except:
                    pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CARLA交互式数据收集')
    
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA服务器地址（默认：localhost）')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA服务器端口（默认：2000）')
    parser.add_argument('--town', type=str, default='Town01',
                       help='地图名称（默认：Town01）')
    parser.add_argument('--num-frames', type=int, default=10000,
                       help='每次收集的帧数（默认：10000）')
    parser.add_argument('--save-path', type=str, default='./carla_data',
                       help='数据保存路径（默认：./carla_data）')
    parser.add_argument('--visualize', action='store_true',
                       help='启用实时可视化（交互式模式下默认启用）')
    parser.add_argument('--respect-traffic-lights', action='store_true',
                       help='遵守红绿灯（默认忽略）')
    parser.add_argument('--respect-signs', action='store_true',
                       help='遵守停车标志（默认忽略）')
    parser.add_argument('--ignore-vehicles', type=int, default=80,
                       help='忽略其他车辆的百分比 0-100（默认：80）')
    parser.add_argument('--target-speed', type=float, default=10.0,
                       help='目标速度 km/h（默认：10.0）')
    parser.add_argument('--fps', type=int, default=20,
                       help='模拟帧率（默认：20）')
    # 噪声相关参数
    parser.add_argument('--noise', action='store_true',
                       help='启用噪声注入（DAgger风格）')
    parser.add_argument('--no-lateral-noise', action='store_true',
                       help='禁用横向噪声（默认启用）')
    parser.add_argument('--longitudinal-noise', action='store_true',
                       help='启用纵向噪声（默认禁用）')
    parser.add_argument('--noise-ratio', type=float, default=0.4,
                       help='噪声时间占比 0-1（默认：0.4）')
    parser.add_argument('--max-steer-offset', type=float, default=0.35,
                       help='最大转向偏移 0-1（默认：0.35）')
    parser.add_argument('--max-throttle-offset', type=float, default=0.2,
                       help='最大油门偏移 0-1（默认：0.2）')
    
    args = parser.parse_args()
    
    # 创建交互式收集器
    collector = InteractiveDataCollector(
        host=args.host,
        port=args.port,
        town=args.town,
        ignore_traffic_lights=not args.respect_traffic_lights,
        ignore_signs=not args.respect_signs,
        ignore_vehicles_percentage=args.ignore_vehicles,
        target_speed=args.target_speed,
        simulation_fps=args.fps,
        # 噪声配置
        noise_enabled=args.noise,
        lateral_noise=not args.no_lateral_noise,
        longitudinal_noise=args.longitudinal_noise,
        noise_ratio=args.noise_ratio,
        max_steer_offset=args.max_steer_offset,
        max_throttle_offset=args.max_throttle_offset
    )
    
    # 运行
    collector.run(
        num_frames=args.num_frames,
        save_path=args.save_path,
        visualize=args.visualize
    )


if __name__ == '__main__':
    """
    使用说明：
    
    1. 启动CARLA服务器：
       F:\\CARLA_0.9.16\\CarlaUE4.exe
    
    2. 运行交互式数据收集：
       python interactive_data_collection.py
    
    3. 按照提示操作：
       - 首先会看到所有生成点的彩色标记
       - 输入起点索引（例如：0）
       - 输入终点索引（例如：105）
       - 查看蓝色导航路径
       - 输入"开始"开始收集数据
       - 收集完成后选择是否继续
    
    参数说明：
       --town: 地图名称（默认Town01）
       --num-frames: 每次收集的帧数（默认10000）
       --save-path: 保存路径（默认./carla_data）
       --visualize: 参数保留（交互式模式下强制启用）
       --respect-traffic-lights: 遵守红绿灯
       --respect-signs: 遵守停车标志
       --ignore-vehicles: 忽略其他车辆的百分比（默认80）
    
    示例命令：
       # 基础使用（自动启用可视化）
       python interactive_data_collection.py
       
       # 切换到Town02，收集5000帧
       python interactive_data_collection.py --town Town02 --num-frames 5000
       
       # 遵守所有交通规则
       python interactive_data_collection.py --respect-traffic-lights --respect-signs --ignore-vehicles 0
    
    交互流程：
       1. 查看所有生成点（彩色柱体+索引数字）
       2. 输入起点索引 -> 输入终点索引
       3. 规划路径并显示标记（供用户查看路线）
          - 起点/终点标记、黄线、蓝色路径
       4. 自动开始倒计时清除标记（统一30秒）
          - ⏰ 所有标记（生成点、索引、起点/终点、黄线、蓝色路径）统一显示30秒
          - 📊 一个统一的进度条倒计时，实时显示百分比和剩余秒数
          - ✅ 倒计时结束后所有标记同时完全消失
          - 💡 用户可以利用这段时间在CARLA中观察路线
       5. 倒计时结束后，输入"开始"命令确认
          - 可选择：开始收集 / 重选路线 / 退出
       6. ⭐ 简化的循环式数据收集（新流程）
          - 🎯 检测当前导航命令（Follow/Left/Right/Straight）
          - ❓ 询问是否收集该命令段
          - ✅ 选择"保存" → 自动收集200帧 → 自动保存 → 继续询问
          - ❌ 选择"跳过" → 等待命令变化 → 继续询问
          - ⏹️  选择"停止" → 结束收集
          - 📊 每段固定200帧，自动保存
          - 🔄 保存后立即询问下一段（无需等待）
          - 📁 文件名使用选择时的命令类型（如：carla_cmd3_Left_xxx_part001.h5）
       7. 数据收集完成后，选择继续或退出
       8. 重复步骤1-7收集更多路线（每次都会重新显示生成点）
    
    特点：
       ✅ 保持CARLA视角不变
       ✅ 优化的流程：规划路径 → 自动倒计时 → 确认后收集
       ✅ 所有可视化标记统一显示时间（30秒）
       ✅ 统一的倒计时进度条，简洁清晰
       ✅ 所有标记（生成点、索引、起点/终点、黄线、蓝色路径）同时消失
       ✅ 每次新任务都会重新显示生成点（确保始终可见）
       ✅ 规划路径后立即开始倒计时（让用户观察路线）
       ✅ 精美的进度条显示清除进度（█/░字符）
       ✅ 实时显示百分比和剩余秒数（flush=True确保实时更新）
       ✅ 智能计算等待时间，确保画面完全清空
       ✅ 倒计时结束后再确认是否开始（避免尴尬等待）
       ✅ 所有文字标记（索引数字、起点/终点文字）完全清除
       ✅ 路径规划失败时自动重新选择
       ✅ 可以连续收集多条路线
       ✅ 强制启用实时可视化（显示收集过程）
       ✅ 可视化窗口显示图像、速度、控制信号、进度等
       ✅ 按ESC可关闭可视化窗口（数据收集继续）
       
       ⭐ 新增：简化的循环式交互收集
       ✅ 询问是否收集 → 收集200帧 → 自动保存 → 循环
       ✅ 文件名使用用户选择时的命令类型
       ✅ 每段固定200帧，流程清晰简洁
       ✅ 自动保存后立即询问下一段
       ✅ 跳过模式自动等待命令变化
       ✅ 精准控制收集哪些场景的数据
       ✅ 避免收集不需要的命令段
       ✅ 完美解决数据不平衡问题
    """
    main()
