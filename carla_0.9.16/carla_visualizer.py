#!/usr/bin/env python
# coding=utf-8
'''
CARLA 可视化模块
方案A：创建跟随摄像头，将模型输入图像叠加到左上角，用OpenCV显示合成图像
新增：右上角实时路线图显示
'''

import cv2
import time
import math
import numpy as np
import carla
from collections import deque
from carla_config import *


class RouteMapRenderer:
    """路线图渲染器 - 车头朝上的导航风格实时路线图"""
    
    def __init__(self, map_size=300, margin=20, padding=20):
        """
        初始化路线图渲染器
        
        参数:
            map_size: 小地图尺寸（像素）
            margin: 距离屏幕边缘的边距
            padding: 地图内部边距
        """
        self.map_size = map_size
        self.margin = margin
        self.padding = padding
        
        # 视野范围（世界坐标，米）- 以车辆为中心显示多大范围
        self.view_range = 80.0  # 显示车辆周围 80米 x 80米 的范围
        
        # 路线数据缓存
        self._route_waypoints = []  # [(x, y), ...]
        
        # 路网数据缓存
        self._road_network = []  # [[(x1,y1), (x2,y2), ...], ...] 道路段列表
        self._road_network_initialized = False
        
        # 导航风格配色 (BGR) - 夜间模式（类似高德/百度导航）
        self.color_bg = (40, 40, 45)           # 深色背景
        
        # 道路颜色（分层次）
        self.color_road_main = (70, 70, 75)    # 主干道（较亮）
        self.color_road_normal = (55, 55, 60)  # 普通道路
        self.color_road_outline = (35, 35, 40) # 道路边缘（深色）
        self.color_road_center = (50, 50, 55)  # 道路中心线（虚线）
        self.color_junction = (60, 60, 65)     # 路口区域
        
        self.color_border = (80, 80, 90)       # 边框
        self.color_route_done = (60, 60, 65)   # 已走过的路线（暗灰）
        self.color_route_remaining = (255, 180, 60)  # 剩余路线（亮橙色/导航色）
        self.color_route_glow = (255, 220, 120)  # 路线发光效果
        self.color_route_outline = (200, 140, 40)  # 路线边缘
        self.color_vehicle = (100, 255, 100)   # 车辆位置（亮绿色）
        self.color_vehicle_glow = (50, 200, 50)  # 车辆发光
        self.color_destination = (80, 80, 255)  # 目的地（红色）
        self.color_dest_glow = (50, 50, 180)   # 目的地发光
        self.color_start = (255, 180, 50)      # 起点（橙黄色）
        self.color_text = (230, 230, 230)      # 文字颜色
        self.color_text_shadow = (25, 25, 30)  # 文字阴影
        self.color_north = (120, 200, 255)     # 北向指示器
        
        # 路口位置缓存
        self._junctions = []  # [(x, y, radius), ...]
        
    def set_route(self, route_waypoints):
        """
        设置路线数据
        
        参数:
            route_waypoints: 路线路点列表 [(waypoint, road_option), ...]
        """
        if not route_waypoints:
            self._route_waypoints = []
            return
            
        # 提取坐标
        self._route_waypoints = []
        for wp, _ in route_waypoints:
            loc = wp.transform.location
            self._route_waypoints.append((loc.x, loc.y))
    
    def set_road_network(self, world):
        """
        从CARLA世界获取路网数据（使用generate_waypoints获取连续道路）
        
        参数:
            world: CARLA world 对象
        """
        if self._road_network_initialized:
            return
            
        try:
            carla_map = world.get_map()
            
            self._road_network = []
            self._junctions = []
            junction_set = set()
            
            # 使用 generate_waypoints 获取所有道路上的 waypoint
            # 参数是采样间距（米）
            all_waypoints = carla_map.generate_waypoints(2.0)
            
            # 按 (road_id, lane_id) 分组，形成连续的车道
            lane_dict = {}  # {(road_id, lane_id): [waypoints]}
            
            for wp in all_waypoints:
                key = (wp.road_id, wp.lane_id)
                if key not in lane_dict:
                    lane_dict[key] = []
                lane_dict[key].append(wp)
            
            # 对每条车道的 waypoint 按位置排序，形成连续路径
            for (road_id, lane_id), waypoints in lane_dict.items():
                if len(waypoints) < 2:
                    continue
                
                # 按 s 值（沿道路的距离）排序
                waypoints.sort(key=lambda w: w.s)
                
                # 提取坐标和宽度
                segment_points = []
                segment_widths = []
                is_junction_road = waypoints[0].is_junction
                
                for wp in waypoints:
                    loc = wp.transform.location
                    segment_points.append((loc.x, loc.y))
                    segment_widths.append(wp.lane_width)
                    
                    # 收集路口信息
                    if wp.is_junction:
                        junction = wp.get_junction()
                        if junction is not None:
                            jid = junction.id
                            if jid not in junction_set:
                                junction_set.add(jid)
                                bb = junction.bounding_box
                                jx = bb.location.x
                                jy = bb.location.y
                                jr = max(bb.extent.x, bb.extent.y)
                                self._junctions.append((jx, jy, jr))
                
                if len(segment_points) >= 2:
                    # 计算平均宽度
                    avg_width = sum(segment_widths) / len(segment_widths)
                    # 存储道路段信息
                    self._road_network.append({
                        'points': segment_points,
                        'width': avg_width,
                        'is_junction': is_junction_road
                    })
            
            self._road_network_initialized = True
            print(f"✅ 路网数据已加载: {len(self._road_network)} 条车道, {len(self._junctions)} 个路口")
            
        except Exception as e:
            print(f"⚠️ 加载路网数据失败: {e}")
            import traceback
            traceback.print_exc()
            self._road_network = []
            self._junctions = []
    
    def _world_to_map_rotated(self, world_x, world_y, center_x, center_y, yaw_deg, scale):
        """
        将世界坐标转换为旋转后的地图像素坐标（车头朝上）
        
        CARLA 坐标系（俯视图）：
        - X 轴：向右（东）
        - Y 轴：向下（南）（注意：CARLA 使用左手坐标系）
        - yaw = 0°：车头朝 X+ 方向（东）
        - yaw = 90°：车头朝 Y+ 方向（南）
        - yaw = -90°：车头朝 Y- 方向（北）
        
        屏幕坐标系：
        - X 轴：向右
        - Y 轴：向下
        - 我们要让车头朝上（屏幕 Y- 方向）
        
        转换步骤：
        1. 计算相对于车辆的位置
        2. 旋转使车头方向对齐屏幕上方
        3. 转换到屏幕像素坐标
        
        参数:
            world_x, world_y: 世界坐标
            center_x, center_y: 车辆位置（世界坐标）
            yaw_deg: 车辆朝向角度（度）
            scale: 缩放比例（像素/米）
        """
        # 相对于车辆的位置
        rel_x = world_x - center_x
        rel_y = world_y - center_y
        
        # 旋转角度：需要将车头方向旋转到屏幕上方
        # 车头方向向量：(cos(yaw), sin(yaw))
        # 要让这个向量指向屏幕上方 (0, -1)
        # 需要旋转 -(yaw + 90°) 或等价地 (-yaw - 90°)
        rotation_rad = math.radians(-yaw_deg - 90)
        
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        
        # 旋转坐标
        rot_x = rel_x * cos_r - rel_y * sin_r
        rot_y = rel_x * sin_r + rel_y * cos_r
        
        # 转换到地图像素坐标（车辆在中心）
        map_center = self.map_size // 2
        map_x = int(map_center + rot_x * scale)
        map_y = int(map_center + rot_y * scale)  # 注意：不翻转Y，因为旋转已经处理了方向
        
        return (map_x, map_y)
    
    def render(self, vehicle_location, vehicle_yaw, current_waypoint_index=0, 
               remaining_distance=0, progress=0):
        """
        渲染路线图（车头朝上模式）
        
        参数:
            vehicle_location: 车辆位置 (x, y)
            vehicle_yaw: 车辆朝向角度（度）
            current_waypoint_index: 当前路点索引
            remaining_distance: 剩余距离（米）
            progress: 进度百分比
            
        返回:
            numpy数组: 路线图图像 (map_size, map_size, 3)
        """
        # 创建背景
        map_img = np.full((self.map_size, self.map_size, 3), self.color_bg, dtype=np.uint8)
        
        # 计算缩放比例
        draw_size = self.map_size - 2 * self.padding
        scale = draw_size / self.view_range  # 像素/米
        
        veh_x, veh_y = vehicle_location
        
        # 1. 绘制路网（周围道路）
        self._draw_road_network(map_img, veh_x, veh_y, vehicle_yaw, scale)
        
        # 2. 绘制导航路线
        if self._route_waypoints and len(self._route_waypoints) >= 2:
            self._draw_navigation_route(map_img, veh_x, veh_y, vehicle_yaw, scale, 
                                        current_waypoint_index)
        
        # 3. 绘制终点标记（如果在视野内）
        if self._route_waypoints:
            end_x, end_y = self._route_waypoints[-1]
            end_pt = self._world_to_map_rotated(end_x, end_y, veh_x, veh_y, vehicle_yaw, scale)
            if self._is_in_view(end_pt):
                self._draw_destination_marker(map_img, end_pt)
        
        # 4. 绘制车辆（固定在中心，朝上）
        center = (self.map_size // 2, self.map_size // 2)
        self._draw_vehicle_marker(map_img, center)
        
        # 5. 绘制边框
        self._draw_border(map_img)
        
        # 6. 绘制北向指示器
        self._draw_north_indicator(map_img, vehicle_yaw)
        
        # 7. 绘制顶部信息栏
        self._draw_header(map_img, remaining_distance, progress)
        
        return map_img
    
    def _is_in_view(self, pt):
        """检查点是否在视野内"""
        margin = 10
        return (margin < pt[0] < self.map_size - margin and 
                margin < pt[1] < self.map_size - margin)
    
    def _draw_road_network(self, img, center_x, center_y, yaw_deg, scale):
        """绘制周围路网（增强版）"""
        if not self._road_network:
            return
        
        # 第一遍：绘制路口区域（底层）
        for jx, jy, jr in self._junctions:
            dist = math.sqrt((jx - center_x)**2 + (jy - center_y)**2)
            if dist < self.view_range + jr:
                jpt = self._world_to_map_rotated(jx, jy, center_x, center_y, yaw_deg, scale)
                jr_px = int(jr * scale)
                if jr_px > 3:
                    # 路口区域（圆形）
                    cv2.circle(img, jpt, jr_px, self.color_junction, -1)
        
        # 第二遍：绘制道路边缘（阴影层）
        for segment in self._road_network:
            points = segment['points']
            width = segment['width']
            
            map_points = []
            in_view = False
            
            for wx, wy in points:
                dist = math.sqrt((wx - center_x)**2 + (wy - center_y)**2)
                if dist < self.view_range * 0.9:
                    in_view = True
                pt = self._world_to_map_rotated(wx, wy, center_x, center_y, yaw_deg, scale)
                map_points.append(pt)
            
            if in_view and len(map_points) >= 2:
                points_array = np.array(map_points, dtype=np.int32)
                # 根据道路宽度计算绘制宽度
                road_width_px = max(4, int(width * scale * 0.8))
                # 道路边缘（深色阴影）
                cv2.polylines(img, [points_array], False, self.color_road_outline, 
                              road_width_px + 4, cv2.LINE_AA)
        
        # 第三遍：绘制道路主体
        for segment in self._road_network:
            points = segment['points']
            width = segment['width']
            is_junction = segment['is_junction']
            
            map_points = []
            in_view = False
            
            for wx, wy in points:
                dist = math.sqrt((wx - center_x)**2 + (wy - center_y)**2)
                if dist < self.view_range * 0.9:
                    in_view = True
                pt = self._world_to_map_rotated(wx, wy, center_x, center_y, yaw_deg, scale)
                map_points.append(pt)
            
            if in_view and len(map_points) >= 2:
                points_array = np.array(map_points, dtype=np.int32)
                road_width_px = max(4, int(width * scale * 0.8))
                
                # 选择道路颜色（主干道更亮）
                road_color = self.color_road_main if width > 3.5 else self.color_road_normal
                
                # 道路主体
                cv2.polylines(img, [points_array], False, road_color, 
                              road_width_px, cv2.LINE_AA)
                
                # 道路中心虚线（仅非路口道路）
                if not is_junction and road_width_px >= 6:
                    self._draw_dashed_line(img, map_points, self.color_road_center, 1)
    
    def _draw_dashed_line(self, img, points, color, thickness, dash_length=8, gap_length=6):
        """绘制虚线"""
        if len(points) < 2:
            return
        
        total_drawn = 0
        drawing = True
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_len = math.sqrt(dx*dx + dy*dy)
            
            if segment_len < 1:
                continue
            
            # 单位向量
            ux = dx / segment_len
            uy = dy / segment_len
            
            pos = 0
            while pos < segment_len:
                if drawing:
                    # 绘制实线段
                    start_x = int(p1[0] + ux * pos)
                    start_y = int(p1[1] + uy * pos)
                    end_pos = min(pos + dash_length, segment_len)
                    end_x = int(p1[0] + ux * end_pos)
                    end_y = int(p1[1] + uy * end_pos)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
                    pos = end_pos
                    total_drawn += dash_length
                    if total_drawn >= dash_length:
                        drawing = False
                        total_drawn = 0
                else:
                    # 跳过间隙
                    pos += gap_length
                    total_drawn += gap_length
                    if total_drawn >= gap_length:
                        drawing = True
                        total_drawn = 0
    
    def _draw_navigation_route(self, img, center_x, center_y, yaw_deg, scale, 
                                current_waypoint_index):
        """绘制导航路线（增强版，带箭头指示方向）"""
        if not self._route_waypoints:
            return
            
        # 转换所有路点
        map_points = []
        for wx, wy in self._route_waypoints:
            pt = self._world_to_map_rotated(wx, wy, center_x, center_y, yaw_deg, scale)
            map_points.append(pt)
        
        # 绘制已走过的路线（暗灰虚线）
        if current_waypoint_index > 0:
            done_points = map_points[:current_waypoint_index+1]
            if len(done_points) >= 2:
                done_array = np.array(done_points, dtype=np.int32)
                cv2.polylines(img, [done_array], False, self.color_route_done, 3, cv2.LINE_AA)
        
        # 绘制剩余路线（带发光效果和边框）
        if current_waypoint_index < len(map_points) - 1:
            remaining_points = map_points[current_waypoint_index:]
            if len(remaining_points) >= 2:
                remaining_array = np.array(remaining_points, dtype=np.int32)
                # 外发光层（最粗）
                cv2.polylines(img, [remaining_array], False, self.color_route_glow, 12, cv2.LINE_AA)
                # 边框层
                cv2.polylines(img, [remaining_array], False, self.color_route_outline, 7, cv2.LINE_AA)
                # 主路线
                cv2.polylines(img, [remaining_array], False, self.color_route_remaining, 5, cv2.LINE_AA)
                
                # 绘制方向箭头（每隔一段距离）
                self._draw_route_arrows(img, remaining_points)
    
    def _draw_route_arrows(self, img, points, arrow_spacing=40):
        """在路线上绘制方向箭头"""
        if len(points) < 2:
            return
        
        accumulated_dist = 0
        last_arrow_dist = 0
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_len = math.sqrt(dx*dx + dy*dy)
            
            if segment_len < 1:
                continue
            
            # 检查是否需要在这段上绘制箭头
            while accumulated_dist + segment_len > last_arrow_dist + arrow_spacing:
                # 计算箭头位置
                arrow_pos_in_segment = last_arrow_dist + arrow_spacing - accumulated_dist
                if arrow_pos_in_segment < 0 or arrow_pos_in_segment > segment_len:
                    break
                
                ratio = arrow_pos_in_segment / segment_len
                ax = int(p1[0] + dx * ratio)
                ay = int(p1[1] + dy * ratio)
                
                # 检查是否在视野内
                if 20 < ax < self.map_size - 20 and 35 < ay < self.map_size - 20:
                    # 计算箭头方向
                    angle = math.atan2(dy, dx)
                    arrow_size = 6
                    
                    # 箭头三个顶点
                    tip_x = int(ax + arrow_size * math.cos(angle))
                    tip_y = int(ay + arrow_size * math.sin(angle))
                    
                    left_x = int(ax - arrow_size * 0.7 * math.cos(angle - math.pi/4))
                    left_y = int(ay - arrow_size * 0.7 * math.sin(angle - math.pi/4))
                    
                    right_x = int(ax - arrow_size * 0.7 * math.cos(angle + math.pi/4))
                    right_y = int(ay - arrow_size * 0.7 * math.sin(angle + math.pi/4))
                    
                    # 绘制箭头
                    arrow_pts = np.array([[tip_x, tip_y], [left_x, left_y], [right_x, right_y]], dtype=np.int32)
                    cv2.fillPoly(img, [arrow_pts], (255, 255, 255))
                
                last_arrow_dist += arrow_spacing
            
            accumulated_dist += segment_len
    
    def _draw_border(self, img):
        """绘制现代风格边框（圆角毛玻璃效果）"""
        size = self.map_size
        border_radius = 12  # 圆角半径
        border_width = 2
        
        # 创建圆角遮罩，裁剪四角
        self._apply_rounded_corners(img, border_radius)
        
        # 外层柔和阴影（多层渐变模拟）
        for i in range(4, 0, -1):
            alpha = 0.15 * (5 - i) / 4
            shadow_color = (int(15 * alpha), int(15 * alpha), int(18 * alpha))
            self._draw_rounded_rect_outline(img, i, i, size - i*2, size - i*2, 
                                            border_radius - i + 2, shadow_color, 1)
        
        # 主边框 - 渐变金属质感
        # 顶部高光
        for y in range(border_width):
            brightness = 95 - y * 15
            color = (brightness, brightness, brightness + 10)
            cv2.line(img, (border_radius, y), (size - border_radius, y), color, 1)
        
        # 左侧高光
        for x in range(border_width):
            brightness = 85 - x * 15
            color = (brightness, brightness, brightness + 8)
            cv2.line(img, (x, border_radius), (x, size - border_radius), color, 1)
        
        # 底部阴影
        for y in range(border_width):
            brightness = 45 + y * 5
            color = (brightness, brightness, brightness + 5)
            cv2.line(img, (border_radius, size - 1 - y), (size - border_radius, size - 1 - y), color, 1)
        
        # 右侧阴影
        for x in range(border_width):
            brightness = 50 + x * 5
            color = (brightness, brightness, brightness + 5)
            cv2.line(img, (size - 1 - x, border_radius), (size - 1 - x, size - border_radius), color, 1)
        
        # 圆角部分 - 使用弧线
        # 左上角弧线（高光）
        cv2.ellipse(img, (border_radius, border_radius), (border_radius, border_radius), 
                    180, 0, 90, (90, 90, 100), border_width)
        # 右上角弧线（高光）
        cv2.ellipse(img, (size - border_radius - 1, border_radius), (border_radius, border_radius), 
                    270, 0, 90, (85, 85, 95), border_width)
        # 左下角弧线（阴影）
        cv2.ellipse(img, (border_radius, size - border_radius - 1), (border_radius, border_radius), 
                    90, 0, 90, (55, 55, 60), border_width)
        # 右下角弧线（阴影）
        cv2.ellipse(img, (size - border_radius - 1, size - border_radius - 1), (border_radius, border_radius), 
                    0, 0, 90, (50, 50, 55), border_width)
        
        # 内层微弱发光边缘（增加深度感）
        inner_offset = 4
        inner_color = (55, 55, 60)
        self._draw_rounded_rect_outline(img, inner_offset, inner_offset, 
                                        size - inner_offset*2, size - inner_offset*2,
                                        border_radius - 3, inner_color, 1)
    
    def _apply_rounded_corners(self, img, radius):
        """应用圆角效果（将四角裁剪为圆角）"""
        size = self.map_size
        bg_color = self.color_bg
        
        # 左上角
        for y in range(radius):
            for x in range(radius):
                dist = math.sqrt((radius - x - 1)**2 + (radius - y - 1)**2)
                if dist > radius:
                    img[y, x] = bg_color
        
        # 右上角
        for y in range(radius):
            for x in range(size - radius, size):
                dist = math.sqrt((x - (size - radius))**2 + (radius - y - 1)**2)
                if dist > radius:
                    img[y, x] = bg_color
        
        # 左下角
        for y in range(size - radius, size):
            for x in range(radius):
                dist = math.sqrt((radius - x - 1)**2 + (y - (size - radius))**2)
                if dist > radius:
                    img[y, x] = bg_color
        
        # 右下角
        for y in range(size - radius, size):
            for x in range(size - radius, size):
                dist = math.sqrt((x - (size - radius))**2 + (y - (size - radius))**2)
                if dist > radius:
                    img[y, x] = bg_color
    
    def _draw_rounded_rect_outline(self, img, x, y, w, h, radius, color, thickness):
        """绘制圆角矩形轮廓"""
        # 四条边
        cv2.line(img, (x + radius, y), (x + w - radius, y), color, thickness)  # 顶
        cv2.line(img, (x + radius, y + h), (x + w - radius, y + h), color, thickness)  # 底
        cv2.line(img, (x, y + radius), (x, y + h - radius), color, thickness)  # 左
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), color, thickness)  # 右
        
        # 四个圆角
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def _draw_north_indicator(self, img, vehicle_yaw):
        """绘制北向指示器（现代罗盘风格）"""
        # 位置（左下角）
        cx = 32
        cy = self.map_size - 38
        radius = 22
        
        # 计算北方在屏幕上的方向
        north_screen_angle = -(90 + vehicle_yaw)
        north_screen_rad = math.radians(north_screen_angle)
        
        # 外层阴影
        cv2.circle(img, (cx+1, cy+1), radius + 3, (15, 15, 18), -1)
        
        # 背景圆（渐变效果模拟）
        cv2.circle(img, (cx, cy), radius + 2, (35, 35, 40), -1)
        cv2.circle(img, (cx, cy), radius, (28, 28, 32), -1)
        
        # 刻度线（8个方向）
        for i in range(8):
            angle = north_screen_rad + math.radians(i * 45)
            inner_r = radius - 6 if i % 2 == 0 else radius - 4
            outer_r = radius - 2
            
            x1 = int(cx + inner_r * math.sin(angle))
            y1 = int(cy - inner_r * math.cos(angle))
            x2 = int(cx + outer_r * math.sin(angle))
            y2 = int(cy - outer_r * math.cos(angle))
            
            tick_color = (80, 80, 90) if i % 2 == 0 else (60, 60, 70)
            cv2.line(img, (x1, y1), (x2, y2), tick_color, 1)
        
        # 北向箭头（红色三角形）
        arrow_len = radius - 8
        north_x = int(cx + arrow_len * math.sin(north_screen_rad))
        north_y = int(cy - arrow_len * math.cos(north_screen_rad))
        
        # 北向三角形
        n_size = 8
        n_tip_x = int(cx + (arrow_len + 2) * math.sin(north_screen_rad))
        n_tip_y = int(cy - (arrow_len + 2) * math.cos(north_screen_rad))
        n_left_x = int(cx + (arrow_len - n_size) * math.sin(north_screen_rad - 0.3))
        n_left_y = int(cy - (arrow_len - n_size) * math.cos(north_screen_rad - 0.3))
        n_right_x = int(cx + (arrow_len - n_size) * math.sin(north_screen_rad + 0.3))
        n_right_y = int(cy - (arrow_len - n_size) * math.cos(north_screen_rad + 0.3))
        
        north_tri = np.array([[n_tip_x, n_tip_y], [n_left_x, n_left_y], [n_right_x, n_right_y]], dtype=np.int32)
        cv2.fillPoly(img, [north_tri], (100, 120, 255))  # 红色（BGR）
        cv2.polylines(img, [north_tri], True, (150, 170, 255), 1)
        
        # 南向箭头（白色三角形）
        s_tip_x = int(cx - (arrow_len + 2) * math.sin(north_screen_rad))
        s_tip_y = int(cy + (arrow_len + 2) * math.cos(north_screen_rad))
        s_left_x = int(cx - (arrow_len - n_size) * math.sin(north_screen_rad - 0.3))
        s_left_y = int(cy + (arrow_len - n_size) * math.cos(north_screen_rad - 0.3))
        s_right_x = int(cx - (arrow_len - n_size) * math.sin(north_screen_rad + 0.3))
        s_right_y = int(cy + (arrow_len - n_size) * math.cos(north_screen_rad + 0.3))
        
        south_tri = np.array([[s_tip_x, s_tip_y], [s_left_x, s_left_y], [s_right_x, s_right_y]], dtype=np.int32)
        cv2.fillPoly(img, [south_tri], (180, 180, 180))
        
        # 中心点
        cv2.circle(img, (cx, cy), 3, (60, 60, 70), -1)
        cv2.circle(img, (cx, cy), 3, (100, 100, 110), 1)
        
        # N 标记
        n_label_x = int(cx + (radius + 8) * math.sin(north_screen_rad)) - 4
        n_label_y = int(cy - (radius + 8) * math.cos(north_screen_rad)) + 4
        cv2.putText(img, "N", (n_label_x, n_label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 150, 255), 1)
        
        # 外圈边框
        cv2.circle(img, (cx, cy), radius + 2, (70, 70, 80), 1)
    
    def _draw_text_with_shadow(self, img, text, pos, scale, color, thickness=1):
        """绘制带阴影的文字"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 阴影
        cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, scale, self.color_text_shadow, thickness)
        # 主文字
        cv2.putText(img, text, pos, font, scale, color, thickness)
    
    def _draw_header(self, img, remaining_distance, progress):
        """绘制顶部标题栏（简洁现代风格，无图标）"""
        header_height = 36
        size = self.map_size
        border_radius = 12
        
        # 渐变背景（从深灰到更深，带透明感）
        for y in range(header_height):
            alpha = y / header_height
            base = 38 - int(12 * alpha)
            color = (base, base, base + 3)
            # 考虑圆角，左右留出空间
            start_x = border_radius if y < border_radius else 5
            end_x = size - border_radius if y < border_radius else size - 6
            if y < border_radius:
                # 圆角区域内的渐变
                for x in range(start_x, end_x):
                    img[y, x] = color
            else:
                cv2.line(img, (5, y), (size-6, y), color, 1)
        
        # 底部分隔线（柔和渐变线）
        # 主分隔线
        cv2.line(img, (15, header_height), (size-16, header_height), (60, 60, 65), 1)
        # 下方阴影线
        cv2.line(img, (15, header_height+1), (size-16, header_height+1), (32, 32, 36), 1)
        
        # 剩余距离显示（居中，大字体，醒目）
        if remaining_distance > 0:
            if remaining_distance >= 1000:
                dist_text = f"{remaining_distance/1000:.1f} km"
            else:
                dist_text = f"{int(remaining_distance)} m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            
            # 计算文字尺寸以居中
            text_size = cv2.getTextSize(dist_text, font, font_scale, thickness)[0]
            text_x = (size - text_size[0]) // 2
            text_y = 24
            
            # 文字阴影（增加可读性）
            cv2.putText(img, dist_text, (text_x + 1, text_y + 1), 
                        font, font_scale, (20, 20, 25), thickness)
            # 主文字（亮白色，醒目）
            cv2.putText(img, dist_text, (text_x, text_y), 
                        font, font_scale, (240, 245, 255), thickness)
    
    def _draw_destination_marker(self, img, pt):
        """绘制终点标记"""
        # 检查是否在有效范围内
        if not (0 < pt[0] < self.map_size and 0 < pt[1] < self.map_size):
            return
        # 发光效果
        cv2.circle(img, pt, 14, self.color_dest_glow, 2)
        # 外圈
        cv2.circle(img, pt, 10, self.color_destination, -1)
        # 内圈（白色）
        cv2.circle(img, pt, 6, (255, 255, 255), -1)
        # 中心点
        cv2.circle(img, pt, 3, self.color_destination, -1)
    
    def _draw_vehicle_marker(self, img, center, size=14):
        """绘制车辆标记（固定朝上的箭头）"""
        # 车辆始终朝上（屏幕上方）
        # 发光效果（外圈）
        cv2.circle(img, center, size + 5, self.color_vehicle_glow, 2)
        
        # 箭头形状（朝上）
        # 前方顶点（上方）
        front_x = center[0]
        front_y = center[1] - size
        
        # 后方两个顶点
        back1_x = center[0] - int(size * 0.7)
        back1_y = center[1] + int(size * 0.5)
        
        back2_x = center[0] + int(size * 0.7)
        back2_y = center[1] + int(size * 0.5)
        
        # 后方中心凹陷点
        back_center_x = center[0]
        back_center_y = center[1] + int(size * 0.1)
        
        # 绘制箭头（四边形，带凹陷）
        arrow = np.array([
            [front_x, front_y],
            [back1_x, back1_y],
            [back_center_x, back_center_y],
            [back2_x, back2_y]
        ], dtype=np.int32)
        
        # 填充
        cv2.fillPoly(img, [arrow], self.color_vehicle)
        # 白色边框
        cv2.polylines(img, [arrow], True, (255, 255, 255), 2)


class CarlaVisualizer:
    """CARLA 推理可视化器 - 第三人称跟随 + HUD叠加"""
    
    def __init__(self, mode='spectator'):
        """
        初始化可视化器
        
        参数:
            mode: 可视化模式
                - 'spectator': 第三人称跟随模式（推荐）
                - 'opencv': OpenCV独立窗口模式（旧模式）
        """
        self.mode = mode
        self.window_name = 'CARLA Autonomous Driving'
        self.start_time = None
        
        # Spectator模式相关
        self.world = None
        self.vehicle = None
        self.spectator = None
        
        # 跟随摄像头
        self.follow_camera = None
        self.follow_image_buffer = deque(maxlen=1)
        
        # Spectator跟随参数（第三人称斜俯视 chase cam 效果）
        self.spectator_distance = 6.0      # 后方距离（米）
        self.spectator_height = 3.0        # 高度（米）
        self.spectator_pitch = -20.0       # 俯视角度（度，负值向下看）
        
        # 跟随摄像头渲染分辨率（与CARLA UE4窗口一致）
        self.render_width = 1920
        self.render_height = 1080
        
        # HUD叠加参数
        self.hud_scale = 2.5               # 模型输入图像放大倍数（适配高分辨率）
        self.hud_margin = 20               # 边距
        self.hud_border = 3                # 边框宽度
        
        # 路线图渲染器
        self.route_map_renderer = RouteMapRenderer(map_size=300, margin=20)
        self._route_initialized = False
        
    def set_start_time(self, start_time):
        """设置开始时间（用于FPS计算）"""
        self.start_time = start_time
        
    def setup_spectator_mode(self, world, vehicle):
        """
        设置第三人称跟随模式
        
        参数:
            world: CARLA世界对象
            vehicle: 要跟随的车辆
        """
        self.world = world
        self.vehicle = vehicle
        self.spectator = world.get_spectator()
        
        # 创建跟随摄像头
        self._setup_follow_camera()
        
        print(f"✅ 第三人称跟随模式已启用")
        print(f"   渲染分辨率: {self.render_width}x{self.render_height}")
        print(f"   跟随距离: {self.spectator_distance}m, 高度: {self.spectator_height}m")
        
    def _setup_follow_camera(self):
        """创建跟随摄像头（第三人称chase cam效果）"""
        if self.world is None or self.vehicle is None:
            return
            
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # 设置摄像头参数
        camera_bp.set_attribute('image_size_x', str(self.render_width))
        camera_bp.set_attribute('image_size_y', str(self.render_height))
        camera_bp.set_attribute('fov', '100')  # 稍大的FOV，更有临场感
        
        # 摄像头位置（相对于车辆中心）
        # x: 负值=后方, z: 正值=上方
        # 第三人称chase cam: 在车辆后上方，向下看向车辆
        camera_transform = carla.Transform(
            carla.Location(x=-self.spectator_distance, z=self.spectator_height),
            carla.Rotation(pitch=self.spectator_pitch)
        )
        
        # 附加到车辆（使用Rigid固定连接，跟随更稳定）
        self.follow_camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # 注册回调
        self.follow_camera.listen(lambda image: self._on_follow_camera_update(image))
        
        print(f"   跟随摄像头已创建（chase cam模式）")
        
    def _on_follow_camera_update(self, image):
        """跟随摄像头图像回调"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 移除Alpha通道
        # 保持BGR格式，OpenCV直接使用
        self.follow_image_buffer.append(array.copy())
        
    def _update_spectator(self):
        """更新Spectator位置（可选，主要用于CARLA窗口预览）"""
        if self.vehicle is None or self.spectator is None:
            return
            
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        yaw_rad = math.radians(vehicle_rotation.yaw)
        
        offset_x = -self.spectator_distance * math.cos(yaw_rad)
        offset_y = -self.spectator_distance * math.sin(yaw_rad)
        
        spectator_location = carla.Location(
            x=vehicle_location.x + offset_x,
            y=vehicle_location.y + offset_y,
            z=vehicle_location.z + self.spectator_height
        )
        
        spectator_rotation = carla.Rotation(
            pitch=self.spectator_pitch,
            yaw=vehicle_rotation.yaw,
            roll=0
        )
        
        self.spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
        
    def set_route(self, route_waypoints):
        """
        设置路线数据（用于路线图显示）
        
        参数:
            route_waypoints: 路线路点列表 [(waypoint, road_option), ...]
        """
        self.route_map_renderer.set_route(route_waypoints)
        self._route_initialized = True
    
    def init_road_network(self, world):
        """
        初始化路网数据（用于显示周围道路）
        
        参数:
            world: CARLA world 对象
        """
        self.route_map_renderer.set_road_network(world)
        
    def visualize(self, model_input_image, control_result, actual_speed, route_info, frame_count,
                  vehicle_location=None, vehicle_yaw=0, current_waypoint_index=0):
        """
        可视化当前状态
        
        参数:
            model_input_image: 模型输入图像 (numpy array, RGB)
            control_result: 控制预测结果字典
            actual_speed: 实际速度（归一化值 0-1）
            route_info: 路线信息字典
            frame_count: 当前帧数
            vehicle_location: 车辆位置 (x, y)，用于路线图
            vehicle_yaw: 车辆朝向角度（度），用于路线图
            current_waypoint_index: 当前路点索引，用于路线图
        """
        if self.mode == 'spectator':
            self._visualize_spectator_mode(model_input_image, control_result, 
                                           actual_speed, route_info, frame_count,
                                           vehicle_location, vehicle_yaw, current_waypoint_index)
        else:
            self._visualize_opencv_mode(model_input_image, control_result, 
                                        actual_speed, route_info, frame_count)
            
    def _visualize_spectator_mode(self, model_input_image, control_result, 
                                   actual_speed, route_info, frame_count,
                                   vehicle_location=None, vehicle_yaw=0, current_waypoint_index=0):
        """第三人称跟随模式可视化"""
        # 更新spectator位置（同步CARLA窗口视角）
        self._update_spectator()
        
        # 等待跟随摄像头图像
        if len(self.follow_image_buffer) == 0:
            return
            
        # 获取跟随摄像头图像（已经是BGR格式）
        main_image = self.follow_image_buffer[-1].copy()
        
        # 准备模型输入图像（放大并添加边框）
        hud_image = self._prepare_hud_image(model_input_image)
        
        # 叠加HUD到左上角
        self._overlay_hud(main_image, hud_image)
        
        # 绘制状态信息
        self._draw_status_info(main_image, control_result, actual_speed, route_info, frame_count)
        
        # 绘制路线图到右上角
        if self._route_initialized and vehicle_location is not None:
            self._overlay_route_map(main_image, vehicle_location, vehicle_yaw, current_waypoint_index,
                                    route_info.get('remaining_distance', 0),
                                    route_info.get('progress', 0))
        
        # 显示
        cv2.imshow(self.window_name, main_image)
        key = cv2.waitKey(1)
        
        # ESC退出
        if key == 27:
            raise KeyboardInterrupt("用户按下ESC退出")
            
    def _prepare_hud_image(self, model_input_image):
        """准备HUD图像（模型输入图像）"""
        # 放大模型输入图像
        h, w = model_input_image.shape[:2]
        new_w = int(w * self.hud_scale)
        new_h = int(h * self.hud_scale)
        
        hud = cv2.resize(model_input_image, (new_w, new_h))
        
        # RGB转BGR
        hud = cv2.cvtColor(hud, cv2.COLOR_RGB2BGR)
        
        # 添加边框
        hud = cv2.copyMakeBorder(hud, self.hud_border, self.hud_border, 
                                  self.hud_border, self.hud_border,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        return hud
        
    def _overlay_hud(self, main_image, hud_image):
        """将HUD图像叠加到主图像左上角"""
        hud_h, hud_w = hud_image.shape[:2]
        
        # 叠加位置
        x1 = self.hud_margin
        y1 = self.hud_margin
        x2 = x1 + hud_w
        y2 = y1 + hud_h
        
        # 确保不超出边界
        if x2 <= main_image.shape[1] and y2 <= main_image.shape[0]:
            main_image[y1:y2, x1:x2] = hud_image
            
    def _overlay_route_map(self, main_image, vehicle_location, vehicle_yaw, current_waypoint_index,
                           remaining_distance=0, progress=0):
        """将路线图叠加到主图像右上角"""
        # 渲染路线图（传递距离和进度信息）
        route_map = self.route_map_renderer.render(
            vehicle_location, vehicle_yaw, current_waypoint_index,
            remaining_distance, progress
        )
        
        map_h, map_w = route_map.shape[:2]
        
        # 右上角位置
        x1 = main_image.shape[1] - map_w - self.hud_margin
        y1 = self.hud_margin
        x2 = x1 + map_w
        y2 = y1 + map_h
        
        # 确保不超出边界
        if x1 >= 0 and y2 <= main_image.shape[0]:
            # 半透明叠加效果
            alpha = 0.9
            roi = main_image[y1:y2, x1:x2]
            blended = cv2.addWeighted(route_map, alpha, roi, 1 - alpha, 0)
            main_image[y1:y2, x1:x2] = blended
            
    def _draw_status_info(self, image, control_result, actual_speed, route_info, frame_count):
        """绘制状态信息（在HUD图像下方）"""
        # 计算实际速度
        actual_speed_kmh = actual_speed * SPEED_NORMALIZATION_MPS
        command_en = COMMAND_NAMES_EN.get(route_info['current_command'], 'Unknown')
        
        # 计算HUD图像的底部位置
        hud_h = int(IMAGE_HEIGHT * self.hud_scale) + 2 * self.hud_border
        text_start_y = self.hud_margin + hud_h + 35
        
        # 文字参数（适配1920x1080分辨率）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_height = 30
        x_pos = self.hud_margin
        
        # 第一行：命令和速度（绿色）
        text1 = f"Cmd: {command_en} | Spd: {actual_speed_kmh:.1f} km/h"
        cv2.putText(image, text1, (x_pos, text_start_y), 
                    font, font_scale, (0, 255, 0), thickness)
        
        # 第二行：控制信号（黄色）
        text2 = f"Str: {control_result['steer']:+.2f} | Thr: {control_result['throttle']:.2f} | Brk: {control_result['brake']:.2f}"
        cv2.putText(image, text2, (x_pos, text_start_y + line_height), 
                    font, font_scale, (0, 255, 255), thickness)
        
        # 第三行：进度信息（青色）
        text3 = f"Progress: {route_info['progress']:.1f}% | Dist: {route_info['remaining_distance']:.0f}m"
        cv2.putText(image, text3, (x_pos, text_start_y + 2 * line_height), 
                    font, font_scale, (255, 255, 0), thickness)
        
        # FPS（顶部居中，带半透明背景）
        if self.start_time is not None and frame_count > 0:
            fps = frame_count / (time.time() - self.start_time)
            fps_text = f"FPS: {fps:.1f}"
            text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
            
            # 居中位置
            fps_x = (image.shape[1] - text_size[0]) // 2
            fps_y = 35
            
            # 绘制半透明背景框
            bg_padding = 10
            bg_x1 = fps_x - bg_padding
            bg_y1 = fps_y - text_size[1] - bg_padding
            bg_x2 = fps_x + text_size[0] + bg_padding
            bg_y2 = fps_y + bg_padding
            
            # 创建半透明背景
            overlay = image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            
            # 绘制边框
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (80, 80, 80), 1)
            
            # 绘制FPS文字
            cv2.putText(image, fps_text, (fps_x, fps_y), 
                        font, font_scale, (255, 255, 255), thickness)
            
    def _visualize_opencv_mode(self, image, control_result, actual_speed, route_info, frame_count):
        """OpenCV独立窗口可视化（旧模式）"""
        vis_image = image.copy()
        vis_image = cv2.resize(vis_image, (VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))
        
        actual_speed_kmh = actual_speed * SPEED_NORMALIZATION_MPS 
        command_en = COMMAND_NAMES_EN.get(route_info['current_command'], 'Unknown')
        
        y_pos = 20
        line_height = 20
        
        cv2.putText(vis_image, f"Command: {command_en}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(vis_image, f"Progress: {route_info['progress']:.1f}%", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(vis_image, f"Remaining: {route_info['remaining_distance']:.0f}m", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(vis_image, f"Speed: {actual_speed_kmh:.1f} km/h", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(vis_image, f"Steer: {control_result['steer']:+.3f}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(vis_image, f"Throttle: {control_result['throttle']:.3f}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        cv2.putText(vis_image, f"Brake: {control_result['brake']:.3f}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        fps_text = f"FPS: {frame_count / (time.time() - self.start_time):.1f}" \
                   if self.start_time is not None else "FPS: --"
        cv2.putText(vis_image, fps_text, 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self._draw_steering_indicator(vis_image, control_result['steer'])
        
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
        
    def _draw_steering_indicator(self, image, steer_value):
        """绘制方向盘指示器"""
        center_x = VISUALIZATION_WIDTH // 2
        bar_y = VISUALIZATION_HEIGHT - 30
        
        cv2.line(image, (100, bar_y), (300, bar_y), (100, 100, 100), 2)
        cv2.circle(image, (center_x, bar_y), 3, (255, 255, 255), -1)
        
        steer_x = int(center_x + steer_value * 100)
        steer_x = max(100, min(300, steer_x))
        cv2.circle(image, (steer_x, bar_y), 5, (0, 0, 255), -1)
        
    def close(self):
        """关闭可视化"""
        # 销毁跟随摄像头
        if self.follow_camera is not None:
            try:
                self.follow_camera.stop()
                self.follow_camera.destroy()
            except:
                pass
            self.follow_camera = None
            
        cv2.destroyAllWindows()
