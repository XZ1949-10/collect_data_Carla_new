#!/usr/bin/env python
# coding=utf-8
'''
CARLA 可视化模块
方案A：创建跟随摄像头，将模型输入图像叠加到左上角，用OpenCV显示合成图像
'''

import cv2
import time
import math
import numpy as np
import carla
from collections import deque
from carla_config import *


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
        
    def visualize(self, model_input_image, control_result, actual_speed, route_info, frame_count):
        """
        可视化当前状态
        
        参数:
            model_input_image: 模型输入图像 (numpy array, RGB)
            control_result: 控制预测结果字典
            actual_speed: 实际速度（归一化值 0-1）
            route_info: 路线信息字典
            frame_count: 当前帧数
        """
        if self.mode == 'spectator':
            self._visualize_spectator_mode(model_input_image, control_result, 
                                           actual_speed, route_info, frame_count)
        else:
            self._visualize_opencv_mode(model_input_image, control_result, 
                                        actual_speed, route_info, frame_count)
            
    def _visualize_spectator_mode(self, model_input_image, control_result, 
                                   actual_speed, route_info, frame_count):
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
        
        # FPS（右上角，白色）
        if self.start_time is not None and frame_count > 0:
            fps = frame_count / (time.time() - self.start_time)
            fps_text = f"FPS: {fps:.1f}"
            text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
            fps_x = image.shape[1] - text_size[0] - 30
            cv2.putText(image, fps_text, (fps_x, 40), 
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
