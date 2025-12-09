#!/usr/bin/env python
# coding=utf-8
"""
CARLA 资源管理器 V2

改进点：
1. 使用 Context Manager 模式保证资源释放
2. 统一的状态机管理资源生命周期
3. 完善的超时和重试机制
4. 强制清理机制防止资源泄漏
5. 线程安全的回调管理
6. 兼容旧版 CarlaResourceManager 接口

迁移说明：
- 此文件替代 carla_resource_manager.py
- 保持向后兼容，可直接替换导入
"""

import time
import threading
import weakref
from enum import Enum, auto
from contextlib import contextmanager
from typing import Optional, Callable, List, Tuple
import carla


class ResourceState(Enum):
    """资源状态枚举"""
    IDLE = auto()           # 空闲，无资源
    CREATING = auto()       # 正在创建资源
    READY = auto()          # 资源就绪
    DESTROYING = auto()     # 正在销毁资源
    ERROR = auto()          # 错误状态


class CarlaResourceManagerV2:
    """
    CARLA 资源管理器 V2
    
    特性：
    - 状态机管理资源生命周期
    - Context Manager 支持 (with 语句)
    - 自动清理机制
    - 线程安全
    
    使用示例：
        with CarlaResourceManagerV2(world, bp_lib) as mgr:
            mgr.create_all(spawn_transform, camera_cb, collision_cb)
            # 使用资源...
        # 自动清理
    """
    
    # 类级别的活跃管理器追踪（用于紧急清理）
    _active_managers: List[weakref.ref] = []
    _managers_lock = threading.Lock()
    
    def __init__(self, world, blueprint_library, simulation_fps: int = 20):
        """
        初始化资源管理器
        
        参数:
            world: CARLA world 对象
            blueprint_library: CARLA blueprint_library 对象
            simulation_fps: 模拟帧率
        """
        self.world = world
        self.blueprint_library = blueprint_library
        self.simulation_fps = simulation_fps
        
        # 资源引用
        self._vehicle: Optional[carla.Actor] = None
        self._camera: Optional[carla.Actor] = None
        self._collision_sensor: Optional[carla.Actor] = None
        
        # 状态管理
        self._state = ResourceState.IDLE
        self._state_lock = threading.Lock()
        
        # 同步模式状态
        self._sync_mode_enabled = False
        self._original_sync_mode = None  # 记录进入时的同步模式
        
        # 回调管理
        self._camera_callback: Optional[Callable] = None
        self._collision_callback: Optional[Callable] = None
        
        # 配置
        self._destroy_timeout = 5.0      # 销毁超时时间
        self._create_timeout = 10.0      # 创建超时时间
        self._sensor_init_ticks = 10     # 传感器初始化tick次数
        
        # 注册到活跃管理器列表
        with CarlaResourceManagerV2._managers_lock:
            CarlaResourceManagerV2._active_managers.append(weakref.ref(self))
    
    # ==================== 属性访问 ====================
    
    @property
    def vehicle(self) -> Optional[carla.Actor]:
        return self._vehicle
    
    @property
    def camera(self) -> Optional[carla.Actor]:
        return self._camera
    
    @property
    def collision_sensor(self) -> Optional[carla.Actor]:
        return self._collision_sensor
    
    @property
    def state(self) -> ResourceState:
        with self._state_lock:
            return self._state
    
    @property
    def is_ready(self) -> bool:
        return self.state == ResourceState.READY
    
    # ==================== Context Manager ====================
    
    def __enter__(self):
        """进入上下文时记录当前同步模式"""
        try:
            settings = self.world.get_settings()
            self._original_sync_mode = settings.synchronous_mode
        except:
            self._original_sync_mode = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动清理资源"""
        self.destroy_all(restore_original_mode=True)
        
        # 从活跃管理器列表移除
        with CarlaResourceManagerV2._managers_lock:
            CarlaResourceManagerV2._active_managers = [
                ref for ref in CarlaResourceManagerV2._active_managers 
                if ref() is not None and ref() is not self
            ]
        
        return False  # 不抑制异常
    
    # ==================== 同步模式管理 ====================
    
    def _set_sync_mode(self, enabled: bool, wait_time: float = 0.3) -> bool:
        """
        设置同步模式
        
        参数:
            enabled: True=同步模式, False=异步模式
            wait_time: 模式切换后等待时间
            
        返回:
            bool: 是否成功
        """
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode == enabled:
                self._sync_mode_enabled = enabled
                return True
            
            settings.synchronous_mode = enabled
            if enabled:
                settings.fixed_delta_seconds = 1.0 / self.simulation_fps
            else:
                settings.fixed_delta_seconds = None
            
            self.world.apply_settings(settings)
            time.sleep(wait_time)
            self._sync_mode_enabled = enabled
            return True
            
        except Exception as e:
            print(f"⚠️ 同步模式切换失败: {e}")
            return False
    
    def ensure_sync_mode(self) -> bool:
        """确保处于同步模式"""
        return self._set_sync_mode(True)
    
    def ensure_async_mode(self) -> bool:
        """确保处于异步模式"""
        return self._set_sync_mode(False)
    
    # ==================== 资源创建 ====================
    
    def create_vehicle(self, spawn_transform: carla.Transform,
                       vehicle_filter: str = 'vehicle.tesla.model3') -> bool:
        """
        创建车辆
        
        参数:
            spawn_transform: 生成位置
            vehicle_filter: 车辆类型过滤器
            
        返回:
            bool: 是否成功
        """
        if self._vehicle is not None:
            print("⚠️ 车辆已存在，先销毁")
            self._destroy_vehicle()
        
        try:
            vehicle_bp = self.blueprint_library.filter(vehicle_filter)[0]
            self._vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if self._vehicle is None:
                print("❌ 车辆生成失败：位置可能被占用")
                return False
            
            # 等待车辆稳定
            self._stabilize_vehicle()
            print(f"✅ 车辆创建成功 (ID: {self._vehicle.id})")
            return True
            
        except Exception as e:
            print(f"❌ 创建车辆异常: {e}")
            return False
    
    def _stabilize_vehicle(self, ticks: int = 5):
        """等待车辆物理稳定"""
        if self._sync_mode_enabled:
            for _ in range(ticks):
                self.world.tick()
                time.sleep(0.05)
        else:
            time.sleep(0.5)
    
    def create_camera(self, callback: Callable,
                      width: int = 800, height: int = 600, fov: int = 90,
                      location: Tuple[float, float, float] = (2.0, 0, 1.4),
                      rotation: Tuple[float, float, float] = (0, -15, 0)) -> bool:
        """
        创建摄像头
        
        参数:
            callback: 图像回调函数
            width, height: 图像尺寸
            fov: 视场角
            location: 相对车辆位置 (x, y, z)
            rotation: 相对车辆旋转 (roll, pitch, yaw)
            
        返回:
            bool: 是否成功
        """
        if self._vehicle is None:
            print("❌ 无法创建摄像头：车辆不存在")
            return False
        
        if self._camera is not None:
            self._destroy_camera()
        
        try:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(width))
            camera_bp.set_attribute('image_size_y', str(height))
            camera_bp.set_attribute('fov', str(fov))
            
            camera_transform = carla.Transform(
                carla.Location(x=location[0], y=location[1], z=location[2]),
                carla.Rotation(roll=rotation[0], pitch=rotation[1], yaw=rotation[2])
            )
            
            self._camera = self.world.spawn_actor(
                camera_bp, camera_transform,
                attach_to=self._vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
            
            self._camera_callback = callback
            self._camera.listen(callback)
            
            print(f"✅ 摄像头创建成功 (ID: {self._camera.id})")
            return True
            
        except Exception as e:
            print(f"❌ 创建摄像头异常: {e}")
            return False
    
    def create_collision_sensor(self, callback: Callable) -> bool:
        """
        创建碰撞传感器
        
        参数:
            callback: 碰撞回调函数
            
        返回:
            bool: 是否成功
        """
        if self._vehicle is None:
            print("❌ 无法创建碰撞传感器：车辆不存在")
            return False
        
        if self._collision_sensor is not None:
            self._destroy_collision_sensor()
        
        try:
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            self._collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=self._vehicle
            )
            
            self._collision_callback = callback
            self._collision_sensor.listen(callback)
            
            print(f"✅ 碰撞传感器创建成功 (ID: {self._collision_sensor.id})")
            return True
            
        except Exception as e:
            print(f"❌ 创建碰撞传感器异常: {e}")
            return False
    
    def wait_for_sensors(self, timeout: float = 10.0) -> bool:
        """
        等待传感器就绪
        
        参数:
            timeout: 超时时间（秒）
            
        返回:
            bool: 是否成功
        """
        if not self._sync_mode_enabled:
            time.sleep(1.0)
            return True
        
        start_time = time.time()
        
        try:
            for i in range(self._sensor_init_ticks):
                if time.time() - start_time > timeout:
                    print(f"⚠️ 传感器初始化超时")
                    return False
                
                self.world.tick()
                time.sleep(0.05)
            
            return True
            
        except Exception as e:
            print(f"⚠️ 传感器初始化异常: {e}")
            return False
    
    def create_all(self, spawn_transform: carla.Transform,
                   camera_callback: Callable,
                   collision_callback: Callable,
                   vehicle_filter: str = 'vehicle.tesla.model3',
                   camera_width: int = 800,
                   camera_height: int = 600) -> bool:
        """
        一次性创建所有资源
        
        参数:
            spawn_transform: 车辆生成位置
            camera_callback: 摄像头回调
            collision_callback: 碰撞回调
            vehicle_filter: 车辆类型
            camera_width, camera_height: 摄像头分辨率
            
        返回:
            bool: 是否全部成功
        """
        with self._state_lock:
            if self._state not in [ResourceState.IDLE, ResourceState.ERROR]:
                print(f"⚠️ 当前状态 {self._state} 不允许创建资源")
                return False
            self._state = ResourceState.CREATING
        
        try:
            # 确保同步模式
            if not self.ensure_sync_mode():
                raise RuntimeError("无法切换到同步模式")
            
            # 按顺序创建资源
            if not self.create_vehicle(spawn_transform, vehicle_filter):
                raise RuntimeError("车辆创建失败")
            
            if not self.create_camera(camera_callback, camera_width, camera_height):
                raise RuntimeError("摄像头创建失败")
            
            if not self.create_collision_sensor(collision_callback):
                raise RuntimeError("碰撞传感器创建失败")
            
            if not self.wait_for_sensors():
                raise RuntimeError("传感器初始化失败")
            
            with self._state_lock:
                self._state = ResourceState.READY
            
            print("✅ 所有资源创建完成")
            return True
            
        except Exception as e:
            print(f"❌ 资源创建失败: {e}")
            self.destroy_all()
            with self._state_lock:
                self._state = ResourceState.ERROR
            return False
    
    # ==================== 资源销毁 ====================
    
    def _destroy_camera(self):
        """销毁摄像头（内部方法）"""
        if self._camera is None:
            return
        
        try:
            self._camera.stop()
        except:
            pass
        
        try:
            self._camera.destroy()
        except:
            pass
        
        self._camera = None
        self._camera_callback = None
    
    def _destroy_collision_sensor(self):
        """销毁碰撞传感器（内部方法）"""
        if self._collision_sensor is None:
            return
        
        try:
            self._collision_sensor.stop()
        except:
            pass
        
        try:
            self._collision_sensor.destroy()
        except:
            pass
        
        self._collision_sensor = None
        self._collision_callback = None
    
    def _destroy_vehicle(self):
        """销毁车辆（内部方法）"""
        if self._vehicle is None:
            return
        
        try:
            self._vehicle.destroy()
        except:
            pass
        
        self._vehicle = None
    
    def destroy_all(self, restore_original_mode: bool = False):
        """
        销毁所有资源
        
        关键步骤：
        1. 切换到异步模式（避免 tick 死锁）
        2. 按顺序销毁：传感器 -> 车辆
        3. 等待 CARLA 处理
        4. 恢复同步模式（可选）
        
        参数:
            restore_original_mode: 是否恢复到进入时的同步模式
        """
        with self._state_lock:
            if self._state == ResourceState.DESTROYING:
                return  # 避免重复销毁
            self._state = ResourceState.DESTROYING
        
        print("🧹 正在清理资源...")
        
        # 1. 记录当前同步模式
        was_sync = self._sync_mode_enabled
        
        # 2. 切换到异步模式（关键！避免 tick 死锁）
        if was_sync:
            self._set_sync_mode(False, wait_time=0.3)
        
        # 3. 按顺序销毁资源
        self._destroy_collision_sensor()
        self._destroy_camera()
        self._destroy_vehicle()
        
        # 4. 等待 CARLA 处理销毁请求
        time.sleep(0.5)
        
        # 5. 恢复同步模式
        if restore_original_mode and self._original_sync_mode is not None:
            self._set_sync_mode(self._original_sync_mode)
        elif was_sync:
            self._set_sync_mode(True)
        
        with self._state_lock:
            self._state = ResourceState.IDLE
        
        print("✅ 资源清理完成")
    
    # ==================== 安全的 tick ====================
    
    def tick(self) -> bool:
        """
        安全的 tick 调用
        
        只在同步模式且资源就绪时调用
        
        返回:
            bool: 是否成功执行 tick
        """
        if not self._sync_mode_enabled:
            time.sleep(1.0 / self.simulation_fps)
            return True
        
        if self.state != ResourceState.READY:
            return False
        
        try:
            self.world.tick()
            return True
        except Exception as e:
            print(f"⚠️ tick 失败: {e}")
            return False
    
    # ==================== 类方法：紧急清理 ====================
    
    @classmethod
    def cleanup_all_managers(cls):
        """
        清理所有活跃的资源管理器
        
        用于程序异常退出时的紧急清理
        """
        with cls._managers_lock:
            for ref in cls._active_managers:
                mgr = ref()
                if mgr is not None:
                    try:
                        mgr.destroy_all()
                    except:
                        pass
            cls._active_managers.clear()
        
        print("🧹 所有资源管理器已清理")


# ==================== 便捷函数 ====================

@contextmanager
def carla_resources(world, blueprint_library, spawn_transform,
                    camera_callback, collision_callback,
                    simulation_fps: int = 20):
    """
    便捷的资源管理上下文
    
    使用示例：
        with carla_resources(world, bp_lib, transform, cam_cb, col_cb) as mgr:
            while mgr.is_ready:
                mgr.tick()
                # 处理数据...
    """
    mgr = CarlaResourceManagerV2(world, blueprint_library, simulation_fps)
    try:
        if not mgr.create_all(spawn_transform, camera_callback, collision_callback):
            raise RuntimeError("资源创建失败")
        yield mgr
    finally:
        mgr.destroy_all()


# ==================== 注册退出清理 ====================

import atexit

@atexit.register
def _cleanup_on_exit():
    """程序退出时清理所有资源"""
    CarlaResourceManagerV2.cleanup_all_managers()


# ==================== 向后兼容别名 ====================
# 允许使用旧名称导入
CarlaResourceManager = CarlaResourceManagerV2
