#!/usr/bin/env python
# coding=utf-8
'''
CARLA 传感器管理模块
负责摄像头的创建和管理
'''

import numpy as np
import carla
from collections import deque
from carla_config import (CAMERA_RAW_WIDTH, CAMERA_RAW_HEIGHT, CAMERA_FOV, 
                          CAMERA_LOCATION_X, CAMERA_LOCATION_Z, CAMERA_PITCH)


class SensorManager:
    """传感器管理器"""
    
    def __init__(self, world, vehicle):
        """
        初始化传感器管理器
        
        参数:
            world: CARLA世界对象
            vehicle: 车辆对象
        """
        self.world = world
        self.vehicle = vehicle
        
        # 传感器对象
        self.camera = None
        
        # 数据缓冲区
        self.image_buffer = deque(maxlen=1)
        
    def setup_camera(self):
        """设置RGB摄像头"""
        # 如果已存在摄像头，先销毁
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None
        
        print("正在设置摄像头...")
        
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # 设置摄像头参数（使用高分辨率采集，后续会裁剪和resize）
        camera_bp.set_attribute('image_size_x', str(CAMERA_RAW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_RAW_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        
        # 摄像头位置
        camera_transform = carla.Transform(
            carla.Location(x=CAMERA_LOCATION_X, z=CAMERA_LOCATION_Z),
            carla.Rotation(pitch=CAMERA_PITCH)
        )
        
        # 附加到车辆
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # 注册回调函数
        self.camera.listen(lambda image: self._on_camera_update(image))
        
        print("摄像头设置完成！")
        
    def _on_camera_update(self, image):
        """摄像头图像回调函数"""
        # 转换为numpy数组
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 移除Alpha通道
        array = array[:, :, ::-1].copy()  # BGR -> RGB，使用copy()避免负步长问题
        
        # 存储到缓冲区
        self.image_buffer.append(array)
        
    def get_latest_image(self):
        """获取最新图像"""
        if len(self.image_buffer) > 0:
            return self.image_buffer[-1]
        return None
        
    def has_image(self):
        """检查是否有图像数据"""
        return len(self.image_buffer) > 0
        
    def cleanup(self):
        """清理传感器资源"""
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()

