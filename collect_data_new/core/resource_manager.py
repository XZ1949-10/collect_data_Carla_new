#!/usr/bin/env python
# coding=utf-8
"""
CARLA 资源管理器

负责管理 CARLA 中的车辆、传感器等资源的生命周期。
使用 Context Manager 模式保证资源正确释放。

⚠️ 废弃警告 (Deprecated):
    此模块已被废弃，建议使用 sync_mode_manager.py 中的 ResourceLifecycleHelper。
    
    原因：
    1. CarlaResourceManager 内部有独立的同步模式管理，可能与外部 SyncModeManager 冲突
    2. ResourceLifecycleHelper 与 SyncModeManager 配套使用，有更完善的错误恢复机制
    3. ResourceLifecycleHelper 的 safe_tick() 支持自动恢复
    
    迁移指南：
        # 旧代码
        with CarlaResourceManager(world, bp_lib) as mgr:
            mgr.create_all(spawn_transform, camera_cb, collision_cb)
        
        # 新代码
        sync_mgr = SyncModeManager(world)
        helper = ResourceLifecycleHelper(sync_mgr)
        vehicle = helper.spawn_vehicle_safe(bp, transform)
        camera = helper.create_sensor_safe(bp, transform, vehicle, callback)
        # ... 使用资源 ...
        helper.destroy_all_safe([camera], vehicle)
"""

import time
import threading
import weakref
import atexit
import warnings
from enum import Enum, auto
from contextlib import contextmanager
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

if TYPE_CHECKING:
    from .sync_mode_manager import SyncModeManager

# 导入统一的 actor 工具
from .actor_utils import is_actor_alive, safe_destroy_actor, safe_destroy_sensor


class ResourceState(Enum):
    """资源状态枚举"""
    IDLE = auto()
    CREATING = auto()
    READY = auto()
    DESTROYING = auto()
    ERROR = auto()


class CarlaResourceManager:
    """
    CARLA 资源管理器
    
    特性：
    - 状态机管理资源生命周期
    - Context Manager 支持 (with 语句)
    - 自动清理机制
    - 线程安全
    
    使用示例：
        with CarlaResourceManager(world, bp_lib) as mgr:
            mgr.create_all(spawn_transform, camera_cb, collision_cb)
            # 使用资源...
        # 自动清理
    """
    
    _active_managers: List[weakref.ref] = []
    _managers_lock = threading.Lock()
    
    def __init__(self, world, blueprint_library, simulation_fps: int = 20,
                 sync_manager: 'SyncModeManager' = None):
        """
        初始化资源管理器
        
        ⚠️ 废弃警告：建议使用 ResourceLifecycleHelper 替代此类。
        """
        warnings.warn(
            "CarlaResourceManager 已废弃，建议使用 sync_mode_manager.ResourceLifecycleHelper。"
            "详见 resource_manager.py 文件头部的迁移指南。",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA 模块不可用")
        
        self.world = world
        self.blueprint_library = blueprint_library
        self.simulation_fps = simulation_fps
        
        self._vehicle: Optional[carla.Actor] = None
        self._camera: Optional[carla.Actor] = None
        self._collision_sensor: Optional[carla.Actor] = None
        
        self._state = ResourceState.IDLE
        self._state_lock = threading.Lock()
        
        self._sync_mode_enabled = False
        self._original_sync_mode = None
        
        self._camera_callback: Optional[Callable] = None
        self._collision_callback: Optional[Callable] = None
        
        self._destroy_timeout = 5.0
        self._create_timeout = 10.0
        self._sensor_init_ticks = 10
        
        # 可选的同步模式管理器
        self._sync_manager = sync_manager
        
        with CarlaResourceManager._managers_lock:
            CarlaResourceManager._active_managers.append(weakref.ref(self))
    
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
    
    def __enter__(self):
        try:
            settings = self.world.get_settings()
            self._original_sync_mode = settings.synchronous_mode
        except:
            self._original_sync_mode = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy_all(restore_original_mode=True)
        with CarlaResourceManager._managers_lock:
            CarlaResourceManager._active_managers = [
                ref for ref in CarlaResourceManager._active_managers 
                if ref() is not None and ref() is not self
            ]
        return False
    
    def _set_sync_mode(self, enabled: bool, wait_time: float = 1.0) -> bool:
        """设置同步模式（优先使用 SyncModeManager）"""
        # 如果有同步模式管理器，使用它
        if self._sync_manager is not None:
            if enabled:
                result = self._sync_manager.enable_sync_mode(wait_time)
            else:
                result = self._sync_manager.enable_async_mode(wait_time)
            self._sync_mode_enabled = enabled if result else self._sync_mode_enabled
            return result
        
        # 降级方案：直接操作
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode == enabled:
                self._sync_mode_enabled = enabled
                return True
            
            settings.synchronous_mode = enabled
            settings.fixed_delta_seconds = 1.0 / self.simulation_fps if enabled else None
            self.world.apply_settings(settings)
            time.sleep(wait_time)
            
            self._sync_mode_enabled = enabled
            return True
        except Exception as e:
            print(f"⚠️ 同步模式切换失败: {e}")
            return False
    
    def ensure_sync_mode(self) -> bool:
        return self._set_sync_mode(True)
    
    def ensure_async_mode(self) -> bool:
        return self._set_sync_mode(False)
    
    def create_vehicle(self, spawn_transform: carla.Transform,
                       vehicle_filter: str = 'vehicle.tesla.model3') -> bool:
        """创建车辆"""
        if self._vehicle is not None:
            self._destroy_vehicle()
        
        try:
            vehicle_bp = self.blueprint_library.filter(vehicle_filter)[0]
            self._vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if self._vehicle is None:
                print("❌ 车辆生成失败：位置可能被占用")
                return False
            
            self._stabilize_vehicle()
            print(f"✅ 车辆创建成功 (ID: {self._vehicle.id})")
            return True
        except Exception as e:
            print(f"❌ 创建车辆异常: {e}")
            return False
    
    def _stabilize_vehicle(self, ticks: int = 10):
        """等待车辆物理稳定"""
        # 优先使用 SyncModeManager
        if self._sync_manager is not None:
            self._sync_manager.stabilize_tick(ticks)
            return
        
        try:
            settings = self.world.get_settings()
            is_sync = settings.synchronous_mode
        except:
            is_sync = self._sync_mode_enabled
        
        if is_sync:
            for _ in range(ticks):
                try:
                    self.world.tick(2.0)
                    time.sleep(0.05)
                except:
                    break
        else:
            time.sleep(1.0)
    
    def create_camera(self, callback: Callable,
                      width: int = 800, height: int = 600, fov: int = 90,
                      location: Tuple[float, float, float] = (2.0, 0, 1.4),
                      rotation: Tuple[float, float, float] = (0, -15, 0)) -> bool:
        """创建摄像头"""
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
        """创建碰撞传感器"""
        if self._vehicle is None:
            print("❌ 无法创建碰撞传感器：车辆不存在")
            return False
        
        if self._collision_sensor is not None:
            self._destroy_collision_sensor()
        
        try:
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            self._collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self._vehicle
            )
            
            self._collision_callback = callback
            self._collision_sensor.listen(callback)
            
            print(f"✅ 碰撞传感器创建成功 (ID: {self._collision_sensor.id})")
            return True
        except Exception as e:
            print(f"❌ 创建碰撞传感器异常: {e}")
            return False
    
    def wait_for_sensors(self, timeout: float = 10.0) -> bool:
        """等待传感器就绪"""
        # 优先使用 SyncModeManager
        if self._sync_manager is not None:
            success_count = self._sync_manager.stabilize_tick(self._sensor_init_ticks)
            return success_count >= self._sensor_init_ticks // 2
        
        try:
            settings = self.world.get_settings()
            is_sync = settings.synchronous_mode
        except:
            is_sync = self._sync_mode_enabled
        
        if not is_sync:
            time.sleep(1.0)
            return True
        
        start_time = time.time()
        try:
            for _ in range(self._sensor_init_ticks):
                if time.time() - start_time > timeout:
                    return False
                self.world.tick(2.0)
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
        """一次性创建所有资源"""
        with self._state_lock:
            if self._state not in [ResourceState.IDLE, ResourceState.ERROR]:
                return False
            self._state = ResourceState.CREATING
        
        try:
            if not self.ensure_sync_mode():
                raise RuntimeError("无法切换到同步模式")
            
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
    
    def _destroy_camera(self):
        if self._camera is None:
            return
        # 使用统一的安全销毁工具
        safe_destroy_sensor(self._camera, silent=True)
        self._camera = None
        self._camera_callback = None
    
    def _destroy_collision_sensor(self):
        if self._collision_sensor is None:
            return
        # 使用统一的安全销毁工具
        safe_destroy_sensor(self._collision_sensor, silent=True)
        self._collision_sensor = None
        self._collision_callback = None
    
    def _destroy_vehicle(self):
        if self._vehicle is None:
            return
        # 使用统一的安全销毁工具
        safe_destroy_actor(self._vehicle, silent=True)
        self._vehicle = None
    
    def destroy_all(self, restore_original_mode: bool = False):
        """销毁所有资源"""
        with self._state_lock:
            if self._state == ResourceState.DESTROYING:
                return
            self._state = ResourceState.DESTROYING
        
        print("🧹 正在清理资源...")
        
        was_sync = self._sync_mode_enabled
        if was_sync:
            self._set_sync_mode(False, wait_time=1.0)
        
        self._destroy_collision_sensor()
        time.sleep(0.3)
        
        self._destroy_camera()
        time.sleep(0.3)
        
        self._destroy_vehicle()
        time.sleep(0.3)
        
        time.sleep(1.0)
        
        if restore_original_mode and self._original_sync_mode is not None:
            self._set_sync_mode(self._original_sync_mode, wait_time=1.0)
        elif was_sync:
            self._set_sync_mode(True, wait_time=1.0)
        
        with self._state_lock:
            self._state = ResourceState.IDLE
        
        print("✅ 资源清理完成")
    
    def tick(self) -> bool:
        """安全的 tick 调用"""
        # 优先使用 SyncModeManager
        if self._sync_manager is not None:
            return self._sync_manager.safe_tick()
        
        if not self._sync_mode_enabled:
            time.sleep(1.0 / self.simulation_fps)
            return True
        
        if self.state != ResourceState.READY:
            return False
        
        try:
            self.world.tick(2.0)
            return True
        except Exception as e:
            print(f"⚠️ tick 失败: {e}")
            return False
    
    @classmethod
    def cleanup_all_managers(cls):
        """清理所有活跃的资源管理器"""
        with cls._managers_lock:
            for ref in cls._active_managers:
                mgr = ref()
                if mgr is not None:
                    try:
                        mgr.destroy_all()
                    except:
                        pass
            cls._active_managers.clear()


@contextmanager
def carla_resources(world, blueprint_library, spawn_transform,
                    camera_callback, collision_callback,
                    simulation_fps: int = 20):
    """便捷的资源管理上下文"""
    mgr = CarlaResourceManager(world, blueprint_library, simulation_fps)
    try:
        if not mgr.create_all(spawn_transform, camera_callback, collision_callback):
            raise RuntimeError("资源创建失败")
        yield mgr
    finally:
        mgr.destroy_all()


@atexit.register
def _cleanup_on_exit():
    """程序退出时清理所有资源"""
    CarlaResourceManager.cleanup_all_managers()
