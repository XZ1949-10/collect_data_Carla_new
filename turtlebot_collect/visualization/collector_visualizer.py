#!/usr/bin/env python
# coding=utf-8
'''
数据收集可视化模块 (美化版 v3)
负责收集过程中的界面显示

界面布局示意图:
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────┬────────────────────────┐  │
│  │                                                       │  ┌──────────────────┐  │  │
│  │                                                       │  │   运行状态       │  │  │
│  │                                                       │  │  ● 录制中        │  │  │
│  │                                                       │  │  ● 同步正常      │  │  │
│  │                 摄像头实时画面                         │  ├──────────────────┤  │  │
│  │              (200x88 等比放大显示)                     │  │   导航命令       │  │  │
│  │                                                       │  │  [ 跟随 ]        │  │  │
│  │                                                       │  ├──────────────────┤  │  │
│  │                                                       │  │   数据统计       │  │  │
│  │                                                       │  │  片段: 5         │  │  │
│  │                                                       │  │  帧数: 123       │  │  │
│  └───────────────────────────────────────────────────────┴────────────────────────┘  │
│  ┌─────────────────────────────────────────┬──────────────────────────────────────┐  │
│  │  CARLA 控制信号                         │  TurtleBot 速度                      │  │
│  │  ┌───────────────────────────────────┐  │  ┌────────────────────────────────┐  │  │
│  │  │ 转向 [◀━━━━━━━━│━━━━━━━━▶] -0.35  │  │  │ 线速度  [▓▓▓▓▓▓░░░░] +0.22    │  │  │
│  │  │ 油门 [▓▓▓▓▓▓▓▓░░░░░░░░░░]  0.75  │  │  │ 角速度  [◀━━━│━━━━▶] -0.15    │  │  │
│  │  │ 刹车 [░░░░░░░░░░░░░░░░░░]  0.00  │  │  │ 速度    2.5 km/h               │  │  │
│  │  └───────────────────────────────────┘  │  └────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┴──────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘

颜色方案 (现代深色主题):
    - 背景: 深灰蓝 (30, 32, 36)
    - 面板: 稍浅灰 (45, 48, 54)
    - 强调色: 青色/橙色/绿色
    - 录制: 红色脉冲动画
'''

import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

from config import DisplayConfig, CommandConfig, ImageConfig


def get_chinese_font(size=16):
    """
    获取支持中文的字体
    
    参数:
        size (int): 字体大小
    
    返回:
        ImageFont: PIL字体对象
    """
    # 尝试常见的中文字体路径
    font_paths = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    
    # 如果都找不到，使用默认字体
    return ImageFont.load_default()


def put_chinese_text(img, text, position, font_size=16, color=(255, 255, 255)):
    """
    在OpenCV图像上绘制中文文字
    
    参数:
        img (np.ndarray): BGR格式的OpenCV图像
        text (str): 要绘制的文字
        position (tuple): 文字位置 (x, y)
        font_size (int): 字体大小
        color (tuple): BGR颜色
    
    返回:
        np.ndarray: 绘制后的图像
    """
    # 转换为PIL图像 (BGR -> RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 获取字体
    font = get_chinese_font(font_size)
    
    # BGR -> RGB 颜色转换
    color_rgb = (color[2], color[1], color[0])
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color_rgb)
    
    # 转换回OpenCV格式 (RGB -> BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class CollectorVisualizer:
    """
    数据收集可视化器 (美化版 v3)
    
    特点:
    - 现代深色主题
    - 中文界面标题
    - 底部分两栏: CARLA控制信号 + TurtleBot速度
    - 动态录制指示器
    - 清晰的进度条显示
    
    注意: 颜色和命令名称定义统一使用 DisplayConfig，避免重复定义
    """
    
    # ============ 布局常量 ============
    PANEL_WIDTH = 200           # 右侧信息面板宽度
    CONTROL_HEIGHT = 120        # 底部控制面板高度
    PADDING = 10                # 内边距
    LINE_HEIGHT = 22            # 行高
    SECTION_GAP = 6             # 区块间距
    
    # 进度条常量
    BAR_HEIGHT = 16             # 进度条高度
    BAR_WIDTH = 200             # 进度条宽度 (缩小以适应两栏)
    BAR_WIDTH_SMALL = 140       # 小进度条宽度
    BAR_GAP = 24                # 进度条间距
    
    # ============ 颜色定义 - 引用 DisplayConfig ============
    # 背景色
    COLOR_BG_DARK = DisplayConfig.COLOR_BG_DARK
    COLOR_BG_PANEL = DisplayConfig.COLOR_BG_PANEL
    COLOR_BG_CARD = DisplayConfig.COLOR_BG_CARD
    COLOR_BG_HEADER = DisplayConfig.COLOR_BG_HEADER
    
    # 边框和分隔线
    COLOR_BORDER = DisplayConfig.COLOR_BORDER
    COLOR_BORDER_ACCENT = DisplayConfig.COLOR_BORDER_ACCENT
    COLOR_DIVIDER = DisplayConfig.COLOR_DIVIDER
    
    # 文字颜色
    COLOR_TEXT_PRIMARY = DisplayConfig.COLOR_TEXT_PRIMARY
    COLOR_TEXT_SECONDARY = DisplayConfig.COLOR_TEXT_SECONDARY
    COLOR_TEXT_MUTED = DisplayConfig.COLOR_TEXT_MUTED
    COLOR_TITLE = DisplayConfig.COLOR_TITLE
    COLOR_TITLE_CN = DisplayConfig.COLOR_TITLE_CN
    
    # 标题颜色 (扩展)
    COLOR_TITLE_CARLA = DisplayConfig.COLOR_TITLE_CARLA
    COLOR_TITLE_TURTLE = DisplayConfig.COLOR_TITLE_TURTLE
    
    # 状态颜色
    COLOR_RECORDING = DisplayConfig.COLOR_RECORDING
    COLOR_RECORDING_GLOW = DisplayConfig.COLOR_RECORDING_GLOW
    COLOR_STANDBY = DisplayConfig.COLOR_STANDBY
    COLOR_SYNC_OK = DisplayConfig.COLOR_SYNC_OK
    COLOR_SYNC_FAIL = DisplayConfig.COLOR_SYNC_FAIL
    COLOR_WARNING = DisplayConfig.COLOR_WARNING
    
    # 控制信号颜色 (CARLA)
    COLOR_STEER = DisplayConfig.COLOR_STEER
    COLOR_THROTTLE = DisplayConfig.COLOR_THROTTLE
    COLOR_BRAKE = DisplayConfig.COLOR_BRAKE
    
    # TurtleBot 速度颜色
    COLOR_LINEAR = DisplayConfig.COLOR_LINEAR
    COLOR_ANGULAR = DisplayConfig.COLOR_ANGULAR
    COLOR_SPEED = DisplayConfig.COLOR_SPEED
    
    # 命令颜色
    COLOR_CMD_FOLLOW = DisplayConfig.COLOR_CMD_FOLLOW
    COLOR_CMD_LEFT = DisplayConfig.COLOR_CMD_LEFT
    COLOR_CMD_RIGHT = DisplayConfig.COLOR_CMD_RIGHT
    COLOR_CMD_STRAIGHT = DisplayConfig.COLOR_CMD_STRAIGHT
    
    # 进度条
    COLOR_BAR_BG = DisplayConfig.COLOR_BAR_BG
    COLOR_BAR_BORDER = DisplayConfig.COLOR_BAR_BORDER
    COLOR_BAR_TICK = DisplayConfig.COLOR_BAR_TICK
    
    # 中文命令名称 - 引用 DisplayConfig
    COMMAND_NAMES_CN = DisplayConfig.COMMAND_NAMES_CN
    
    def __init__(self, window_name=None, scale_factor=3.0):
        """
        初始化可视化器
        
        参数:
            window_name (str): 窗口名称
            scale_factor (float): 图像放大倍数 (默认 3.0)
        """
        self.window_name = window_name or DisplayConfig.WINDOW_NAME
        self.scale_factor = scale_factor
        self._window_created = False
        self._start_time = time.time()
        
        # 计算显示尺寸 (保持 200x88 宽高比)
        self.image_width = int(ImageConfig.OUTPUT_WIDTH * scale_factor)
        self.image_height = int(ImageConfig.OUTPUT_HEIGHT * scale_factor)
        
        # 总窗口尺寸
        self.display_width = self.image_width + self.PANEL_WIDTH + self.PADDING * 3
        self.display_height = self.image_height + self.CONTROL_HEIGHT + self.PADDING * 3
        
    def create_window(self):
        """创建显示窗口"""
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            self._window_created = True
    
    def destroy_window(self):
        """销毁显示窗口"""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    
    def create_display(self, image, is_collecting, current_command, 
                       frame_count=0, speed=0.0, linear_vel=0.0, angular_vel=0.0,
                       sync_status=True, episode_count=0,
                       steer=0.0, throttle=0.0, brake=0.0, control_enabled=False):
        """
        创建显示图像
        
        参数:
            image (np.ndarray): 当前图像 (RGB格式, 88x200)
            is_collecting (bool): 是否正在录制
            current_command (float): 当前导航命令
            frame_count (int): 当前帧数
            speed (float): 当前速度 (m/s)
            linear_vel (float): 线速度 (m/s)
            angular_vel (float): 角速度 (rad/s)
            sync_status (bool): 传感器同步状态
            episode_count (int): 片段编号
            steer (float): 转向 (-1.0 ~ 1.0)
            throttle (float): 油门 (0.0 ~ 1.0)
            brake (float): 刹车 (0.0 ~ 1.0)
            control_enabled (bool): 控制是否启用 (防误触)
            
        返回:
            np.ndarray: BGR格式显示图像
        """
        # 创建画布
        canvas = np.full((self.display_height, self.display_width, 3), 
                        self.COLOR_BG_DARK, dtype=np.uint8)
        
        # 绘制外边框
        cv2.rectangle(canvas, (2, 2), (self.display_width-3, self.display_height-3), 
                     self.COLOR_BORDER, 1)
        
        # ============ 左侧: 摄像头画面 ============
        img_x, img_y = self.PADDING, self.PADDING
        self._draw_camera_view(canvas, image, img_x, img_y, is_collecting, control_enabled)
        
        # ============ 右侧: 信息面板 ============
        panel_x = self.image_width + self.PADDING * 2
        self._draw_info_panel(canvas, panel_x, self.PADDING, 
                             is_collecting, current_command, frame_count,
                             sync_status, episode_count, control_enabled)
        
        # ============ 底部: 双栏控制面板 ============
        ctrl_y = self.image_height + self.PADDING * 2
        self._draw_dual_control_panel(canvas, self.PADDING, ctrl_y, 
                                      steer, throttle, brake,
                                      linear_vel, angular_vel, speed)
        
        return canvas
    
    def _draw_camera_view(self, canvas, image, x, y, is_collecting, control_enabled=False):
        """绘制摄像头画面区域"""
        # 外边框
        border = 4
        cv2.rectangle(canvas, 
                     (x - border, y - border), 
                     (x + self.image_width + border, y + self.image_height + border), 
                     self.COLOR_BORDER_ACCENT, -1)
        cv2.rectangle(canvas, 
                     (x - border, y - border), 
                     (x + self.image_width + border, y + self.image_height + border), 
                     self.COLOR_BORDER, 2)
        
        # 图像区域
        if image is not None:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_scaled = cv2.resize(img_bgr, (self.image_width, self.image_height), 
                                   interpolation=cv2.INTER_LINEAR)
            canvas[y:y+self.image_height, x:x+self.image_width] = img_scaled
        else:
            cv2.rectangle(canvas, (x, y), (x + self.image_width, y + self.image_height),
                         self.COLOR_BG_PANEL, -1)
            self._draw_centered_chinese_text(canvas, "等待摄像头...", 
                                    x, y, self.image_width, self.image_height,
                                    self.COLOR_TEXT_MUTED, 18)
        
        # 录制指示器
        if is_collecting:
            self._draw_recording_badge(canvas, x + 8, y + 8)
        
        # 控制禁用提示 (显示在画面底部)
        if not control_enabled:
            self._draw_control_disabled_badge(canvas, x + 8, y + self.image_height - 34)
    
    def _draw_recording_badge(self, canvas, x, y):
        """绘制录制徽章 (带脉冲动画)"""
        pulse = (np.sin(time.time() * 5) + 1) / 2
        
        # 背景框
        badge_w, badge_h = 70, 26
        cv2.rectangle(canvas, (x, y), (x + badge_w, y + badge_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + badge_w, y + badge_h), self.COLOR_RECORDING, 2)
        
        # 圆点 (脉冲)
        radius = int(6 + pulse * 2)
        cv2.circle(canvas, (x + 15, y + 13), radius + 2, self.COLOR_RECORDING_GLOW, -1)
        cv2.circle(canvas, (x + 15, y + 13), radius, self.COLOR_RECORDING, -1)
        
        # 文字
        cv2.putText(canvas, "REC", (x + 28, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_RECORDING, 2, cv2.LINE_AA)
    
    def _draw_control_disabled_badge(self, canvas, x, y):
        """绘制控制禁用提示徽章"""
        # 背景框 (半透明黄色警告)
        badge_w, badge_h = 140, 26
        cv2.rectangle(canvas, (x, y), (x + badge_w, y + badge_h), (20, 40, 60), -1)
        cv2.rectangle(canvas, (x, y), (x + badge_w, y + badge_h), self.COLOR_WARNING, 2)
        
        # 警告图标 (三角形)
        pts = np.array([[x + 14, y + 6], [x + 6, y + 20], [x + 22, y + 20]], np.int32)
        cv2.fillPoly(canvas, [pts], self.COLOR_WARNING)
        cv2.putText(canvas, "!", (x + 12, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 40, 60), 1, cv2.LINE_AA)
        
        # 文字
        self._put_chinese_text(canvas, "控制已禁用", (x + 28, y + 5), 13, self.COLOR_WARNING)
    
    def _draw_info_panel(self, canvas, x, y, is_collecting, current_command, frame_count,
                        sync_status, episode_count, control_enabled=False):
        """绘制右侧信息面板"""
        panel_w = self.PANEL_WIDTH
        panel_h = self.image_height + 8
        
        # 面板背景
        cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), self.COLOR_BG_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), self.COLOR_BORDER, 1)
        
        content_x = x + 12
        content_y = y + 8
        inner_w = panel_w - 24
        
        # ═══════ 运行状态 ═══════
        content_y = self._draw_section_title(canvas, content_x, content_y, inner_w, "运行状态")
        
        # 控制启用状态
        if control_enabled:
            self._draw_status_dot(canvas, content_x + 4, content_y + 8, self.COLOR_SYNC_OK, False)
            self._put_chinese_text(canvas, "控制启用", (content_x + 20, content_y + 2), 13, self.COLOR_SYNC_OK)
        else:
            self._draw_status_dot(canvas, content_x + 4, content_y + 8, self.COLOR_WARNING, True)
            self._put_chinese_text(canvas, "控制禁用", (content_x + 20, content_y + 2), 13, self.COLOR_WARNING)
        content_y += 20
        
        # 录制状态
        if is_collecting:
            self._draw_status_dot(canvas, content_x + 4, content_y + 8, self.COLOR_RECORDING, True)
            self._put_chinese_text(canvas, "录制中", (content_x + 20, content_y + 2), 13, self.COLOR_RECORDING)
        else:
            self._draw_status_dot(canvas, content_x + 4, content_y + 8, self.COLOR_STANDBY, False)
            self._put_chinese_text(canvas, "待机", (content_x + 20, content_y + 2), 13, self.COLOR_STANDBY)
        content_y += 20
        
        # 同步状态
        sync_color = self.COLOR_SYNC_OK if sync_status else self.COLOR_SYNC_FAIL
        sync_text = "同步正常" if sync_status else "同步异常"
        self._draw_status_dot(canvas, content_x + 4, content_y + 8, sync_color, False)
        self._put_chinese_text(canvas, sync_text, (content_x + 20, content_y + 2), 13, sync_color)
        content_y += self.SECTION_GAP + 22
        
        # ═══════ 导航命令 ═══════
        content_y = self._draw_section_title(canvas, content_x, content_y, inner_w, "导航命令")
        
        cmd_name = self.COMMAND_NAMES_CN.get(current_command, '未知')
        cmd_color = self._get_command_color(current_command)
        
        # 命令框
        cmd_y = content_y + 2
        cv2.rectangle(canvas, (content_x, cmd_y), (content_x + inner_w, cmd_y + 30), 
                     self.COLOR_BG_CARD, -1)
        cv2.rectangle(canvas, (content_x, cmd_y), (content_x + inner_w, cmd_y + 30), 
                     cmd_color, 2)
        
        # 命令文字居中
        self._draw_centered_chinese_text(canvas, cmd_name, content_x, cmd_y, inner_w, 30, 
                                cmd_color, 18)
        content_y += self.SECTION_GAP + 38
        
        # ═══════ 数据统计 ═══════
        content_y = self._draw_section_title(canvas, content_x, content_y, inner_w, "数据统计")
        
        self._draw_info_item(canvas, content_x, content_y, "片段", str(episode_count),
                            self.COLOR_TEXT_PRIMARY)
        content_y += 20
        if is_collecting:
            self._draw_info_item(canvas, content_x, content_y, "帧数", str(frame_count),
                                self.COLOR_WARNING)
    
    def _draw_section_title(self, canvas, x, y, width, title):
        """绘制区块标题"""
        cv2.rectangle(canvas, (x - 4, y), (x + width + 4, y + 20), self.COLOR_BG_HEADER, -1)
        self._put_chinese_text(canvas, title, (x, y + 2), 14, self.COLOR_TITLE_CN)
        cv2.line(canvas, (x - 4, y + 20), (x + width + 4, y + 20), self.COLOR_DIVIDER, 1)
        return y + 26
    
    def _draw_status_dot(self, canvas, x, y, color, pulse=False):
        """绘制状态圆点"""
        radius = 5
        if pulse:
            p = (np.sin(time.time() * 4) + 1) / 2
            radius = int(5 + p * 2)
            cv2.circle(canvas, (x, y), radius + 3, tuple(c//3 for c in color), -1)
        cv2.circle(canvas, (x, y), radius, color, -1)
    
    def _draw_info_item(self, canvas, x, y, label, value, value_color):
        """绘制信息项"""
        self._put_chinese_text(canvas, f"{label}:", (x + 4, y), 13, self.COLOR_TEXT_MUTED)
        cv2.putText(canvas, str(value), (x + 60, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, value_color, 1, cv2.LINE_AA)
    
    def _draw_centered_text(self, canvas, text, x, y, w, h, color, scale, thickness=1):
        """绘制居中英文文字"""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    
    def _draw_centered_chinese_text(self, canvas, text, x, y, w, h, color, font_size):
        """绘制居中中文文字"""
        font = get_chinese_font(font_size)
        # 获取文字尺寸 (兼容旧版 Pillow)
        try:
            # Pillow 8.0+ 使用 textbbox
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            # 旧版 Pillow 使用 textsize
            text_w, text_h = font.getsize(text)
        text_x = x + (w - text_w) // 2
        text_y = y + (h - text_h) // 2
        self._put_chinese_text(canvas, text, (text_x, text_y), font_size, color)
    
    def _put_chinese_text(self, canvas, text, position, font_size, color):
        """在画布上绘制中文文字 (原地修改)"""
        result = put_chinese_text(canvas, text, position, font_size, color)
        canvas[:] = result[:]
    
    def _get_command_color(self, command):
        """获取命令颜色"""
        return {
            CommandConfig.CMD_FOLLOW: self.COLOR_CMD_FOLLOW,
            CommandConfig.CMD_LEFT: self.COLOR_CMD_LEFT,
            CommandConfig.CMD_RIGHT: self.COLOR_CMD_RIGHT,
            CommandConfig.CMD_STRAIGHT: self.COLOR_CMD_STRAIGHT,
        }.get(command, self.COLOR_TEXT_PRIMARY)

    
    def _draw_dual_control_panel(self, canvas, x, y, steer, throttle, brake,
                                  linear_vel, angular_vel, speed):
        """绘制双栏控制面板: 左边CARLA控制信号，右边TurtleBot速度"""
        panel_w = self.display_width - self.PADDING * 2
        panel_h = self.CONTROL_HEIGHT - self.PADDING
        
        # 面板背景
        cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), self.COLOR_BG_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), self.COLOR_BORDER, 1)
        
        # 计算两栏宽度
        left_w = (panel_w - 10) // 2
        right_w = panel_w - left_w - 10
        right_x = x + left_w + 8
        
        # 中间分隔线
        cv2.line(canvas, (x + left_w + 4, y + 4), (x + left_w + 4, y + panel_h - 4), 
                self.COLOR_DIVIDER, 1)
        
        # ═══════ 左栏: CARLA 控制信号 ═══════
        self._draw_carla_controls(canvas, x, y, left_w, steer, throttle, brake)
        
        # ═══════ 右栏: TurtleBot 速度 ═══════
        self._draw_turtlebot_velocity(canvas, right_x, y, right_w, linear_vel, angular_vel, speed)
    
    def _draw_carla_controls(self, canvas, x, y, width, steer, throttle, brake):
        """绘制 CARLA 控制信号栏"""
        # 标题栏
        cv2.rectangle(canvas, (x, y), (x + width, y + 22), self.COLOR_BG_HEADER, -1)
        self._put_chinese_text(canvas, "CARLA 控制信号", (x + 10, y + 4), 13, self.COLOR_TITLE_CARLA)
        cv2.line(canvas, (x, y + 22), (x + width, y + 22), self.COLOR_DIVIDER, 1)
        
        # 进度条区域
        bar_x = x + 55
        bar_y = y + 30
        label_x = x + 10
        bar_w = self.BAR_WIDTH
        value_x = bar_x + bar_w + 8
        
        # 转向 (中心对称)
        self._put_chinese_text(canvas, "转向", (label_x, bar_y), 13, self.COLOR_TEXT_SECONDARY)
        self._draw_steer_bar(canvas, bar_x, bar_y, steer, bar_w)
        steer_color = self.COLOR_STEER if abs(steer) > 0.05 else self.COLOR_TEXT_MUTED
        cv2.putText(canvas, f"{steer:+.3f}", (value_x, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, steer_color, 1, cv2.LINE_AA)
        
        # 油门
        bar_y += self.BAR_GAP
        self._put_chinese_text(canvas, "油门", (label_x, bar_y), 13, self.COLOR_TEXT_SECONDARY)
        self._draw_progress_bar(canvas, bar_x, bar_y, throttle, self.COLOR_THROTTLE, bar_w)
        throttle_color = self.COLOR_THROTTLE if throttle > 0.05 else self.COLOR_TEXT_MUTED
        cv2.putText(canvas, f"{throttle:.3f}", (value_x, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, throttle_color, 1, cv2.LINE_AA)
        
        # 刹车
        bar_y += self.BAR_GAP
        self._put_chinese_text(canvas, "刹车", (label_x, bar_y), 13, self.COLOR_TEXT_SECONDARY)
        self._draw_progress_bar(canvas, bar_x, bar_y, brake, self.COLOR_BRAKE, bar_w)
        brake_color = self.COLOR_BRAKE if brake > 0.05 else self.COLOR_TEXT_MUTED
        cv2.putText(canvas, f"{brake:.3f}", (value_x, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, brake_color, 1, cv2.LINE_AA)
    
    def _draw_turtlebot_velocity(self, canvas, x, y, width, linear_vel, angular_vel, speed):
        """绘制 TurtleBot 速度栏"""
        # 标题栏
        cv2.rectangle(canvas, (x, y), (x + width, y + 22), self.COLOR_BG_HEADER, -1)
        self._put_chinese_text(canvas, "TurtleBot 速度", (x + 10, y + 4), 13, self.COLOR_TITLE_TURTLE)
        cv2.line(canvas, (x, y + 22), (x + width, y + 22), self.COLOR_DIVIDER, 1)
        
        # 进度条区域
        bar_x = x + 65
        bar_y = y + 30
        label_x = x + 10
        bar_w = self.BAR_WIDTH_SMALL
        value_x = bar_x + bar_w + 8
        
        # 线速度 (可正可负，但通常为正)
        self._put_chinese_text(canvas, "线速度", (label_x, bar_y), 13, self.COLOR_TEXT_SECONDARY)
        # 归一化到 0~1 (使用配置中的最大值)
        linear_norm = min(1.0, max(0.0, abs(linear_vel) / DisplayConfig.NORM_MAX_LINEAR_VEL))
        self._draw_progress_bar(canvas, bar_x, bar_y, linear_norm, self.COLOR_LINEAR, bar_w)
        linear_color = self.COLOR_LINEAR if abs(linear_vel) > 0.01 else self.COLOR_TEXT_MUTED
        cv2.putText(canvas, f"{linear_vel:+.2f} m/s", (value_x, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.36, linear_color, 1, cv2.LINE_AA)
        
        # 角速度 (中心对称)
        bar_y += self.BAR_GAP
        self._put_chinese_text(canvas, "角速度", (label_x, bar_y), 13, self.COLOR_TEXT_SECONDARY)
        # 归一化到 -1~1 (使用配置中的最大值)
        angular_norm = max(-1.0, min(1.0, angular_vel / DisplayConfig.NORM_MAX_ANGULAR_VEL))
        self._draw_steer_bar(canvas, bar_x, bar_y, angular_norm, bar_w, self.COLOR_ANGULAR)
        angular_color = self.COLOR_ANGULAR if abs(angular_vel) > 0.01 else self.COLOR_TEXT_MUTED
        cv2.putText(canvas, f"{angular_vel:+.2f} r/s", (value_x, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.36, angular_color, 1, cv2.LINE_AA)
        
        # 速度 (km/h)
        bar_y += self.BAR_GAP
        speed_kmh = speed * 3.6
        self._put_chinese_text(canvas, "速度", (label_x, bar_y), 13, self.COLOR_TEXT_SECONDARY)
        # 归一化到 0~1 (使用配置中的最大值)
        speed_norm = min(1.0, max(0.0, speed_kmh / DisplayConfig.NORM_MAX_SPEED_KMH))
        self._draw_progress_bar(canvas, bar_x, bar_y, speed_norm, self.COLOR_SPEED, bar_w)
        speed_color = self.COLOR_SPEED if speed_kmh > 0.1 else self.COLOR_TEXT_MUTED
        cv2.putText(canvas, f"{speed_kmh:.1f} km/h", (value_x, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.36, speed_color, 1, cv2.LINE_AA)
    
    def _draw_steer_bar(self, canvas, x, y, value, width, color=None):
        """绘制转向进度条 (中心对称)"""
        if color is None:
            color = self.COLOR_STEER
        w, h = width, self.BAR_HEIGHT
        
        # 背景
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BAR_BG, -1)
        
        # 刻度线
        for i in range(5):
            tick_x = x + int(w * i / 4)
            tick_color = self.COLOR_BORDER if i == 2 else self.COLOR_BAR_TICK
            cv2.line(canvas, (tick_x, y + h - 4), (tick_x, y + h), tick_color, 1)
        
        # 中心线
        center_x = x + w // 2
        cv2.line(canvas, (center_x, y + 2), (center_x, y + h - 2), self.COLOR_BORDER, 2)
        
        # 值条
        value = max(-1.0, min(1.0, value))
        bar_half = int(abs(value) * (w // 2))
        
        if bar_half > 1:
            if value < 0:
                cv2.rectangle(canvas, (center_x - bar_half, y + 3), (center_x - 1, y + h - 3),
                             color, -1)
                cv2.line(canvas, (center_x - bar_half, y + 3), (center_x - bar_half, y + h - 3),
                        tuple(min(255, c + 40) for c in color), 2)
            else:
                cv2.rectangle(canvas, (center_x + 1, y + 3), (center_x + bar_half, y + h - 3),
                             color, -1)
                cv2.line(canvas, (center_x + bar_half, y + 3), (center_x + bar_half, y + h - 3),
                        tuple(min(255, c + 40) for c in color), 2)
        
        # 边框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BAR_BORDER, 1)
        
        # 左右箭头
        cv2.putText(canvas, "<", (x + 3, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, ">", (x + w - 10, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    
    def _draw_progress_bar(self, canvas, x, y, value, color, width):
        """绘制普通进度条 (0.0 ~ 1.0)"""
        w, h = width, self.BAR_HEIGHT
        
        # 背景
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BAR_BG, -1)
        
        # 刻度线
        for i in range(5):
            tick_x = x + int(w * i / 4)
            cv2.line(canvas, (tick_x, y + h - 4), (tick_x, y + h), self.COLOR_BAR_TICK, 1)
        
        # 值条
        value = max(0.0, min(1.0, value))
        bar_w = int(value * w)
        if bar_w > 1:
            cv2.rectangle(canvas, (x + 1, y + 3), (x + bar_w, y + h - 3), color, -1)
            cv2.line(canvas, (x + bar_w - 1, y + 3), (x + bar_w - 1, y + h - 3),
                    tuple(min(255, c + 50) for c in color), 2)
        
        # 边框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.COLOR_BAR_BORDER, 1)
    
    def show(self, display_image):
        """显示图像"""
        if display_image is None:
            return -1
        self.create_window()
        cv2.imshow(self.window_name, display_image)
        return cv2.waitKey(1) & 0xFF
    
    def cleanup(self):
        """清理资源"""
        self.destroy_window()
        cv2.destroyAllWindows()
