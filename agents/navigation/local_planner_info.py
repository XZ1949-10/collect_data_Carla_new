# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
本模块包含局部路径规划器，用于执行低级路点跟随
核心功能：管理路点队列，提供目标路点信息供外部控制器使用
"""

from enum import IntEnum
from collections import deque
import random

import carla
from agents.tools.misc import draw_waypoints, get_speed


class RoadOption(IntEnum):
    """
    RoadOption 枚举类：表示从一个车道段移动到另一个车道段时可能的拓扑配置
    
    用于描述车辆在路径上的行驶动作类型
    """
    VOID = -1            # 无效/未定义
    LEFT = 1             # 左转
    RIGHT = 2            # 右转
    STRAIGHT = 3         # 直行
    LANEFOLLOW = 4       # 车道跟随（保持当前车道）
    CHANGELANELEFT = 5   # 向左变道
    CHANGELANERIGHT = 6  # 向右变道


class LocalPlanner:
    """
    LocalPlanner（局部路径规划器）- 外部控制版本
    
    实现了跟随动态生成的路点轨迹的基本行为
    
    核心功能：
    1. 管理路点队列（waypoints queue）
    2. 提供目标路点信息供外部控制器使用
    3. 在交叉路口有多条路径可选时，如果没有指定全局规划，则随机选择
    
    工作模式：
    - 自动模式：自动生成随机路点
    - 全局规划模式：跟随预设的全局路径
    
    注意：
    本版本不包含 PID 控制器，需要外部提供控制指令
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None):
        """
        初始化局部路径规划器
        
        参数说明：
        :param vehicle: 要应用规划逻辑的车辆actor
        :param opt_dict: 参数字典，可选参数包括：
            target_speed: 期望巡航速度 (Km/h)
            sampling_radius: 路径中路点之间的距离 (米)
            offset: 路线路点与车道中心的偏移距离 (米，正值向右)
            base_min_distance: 基础最小距离（用于清理路点）
            distance_ratio: 距离比率（用于速度相关的距离调整）
        :param map_inst: carla.Map实例，避免昂贵的获取地图调用
        """
        # ============ 基础设置 ============
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        
        # 获取地图实例
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("警告: 忽略给定的地图，因为它不是 'carla.Map' 类型")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        # ============ 核心组件 ============
        self.target_waypoint = None          # 当前目标路点
        self.target_road_option = None       # 当前道路选项（左转/右转/直行等）

        # ============ 路点队列 ============
        self._waypoints_queue = deque(maxlen=10000)  # 路点队列：存储 (waypoint, RoadOption) 对
        self._min_waypoint_queue_length = 100         # 最小队列长度：保持足够的前瞻距离
        self._stop_waypoint_creation = False          # 是否停止自动创建路点的标志

        # ============ 默认参数 ============
        self._target_speed = 20.0            # 目标速度 (Km/h) - 供外部参考
        self._sampling_radius = 2.0          # 采样半径 (米)：相邻路点间距
        self._offset = 0                     # 车道偏移量（0=中心，正值向右）
        self._base_min_distance = 3.0       # 基础最小距离（用于清理路点）
        self._distance_ratio = 0.5           # 距离比率（用于速度相关的距离调整）
        self._follow_speed_limits = False    # 是否跟随速度限制

        # ============ 参数覆盖 ============
        # 使用opt_dict中提供的参数覆盖默认参数
        if opt_dict:
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = opt_dict['sampling_radius']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']
            if 'base_min_distance' in opt_dict:
                self._base_min_distance = opt_dict['base_min_distance']
            if 'distance_ratio' in opt_dict:
                self._distance_ratio = opt_dict['distance_ratio']
            if 'follow_speed_limits' in opt_dict:
                self._follow_speed_limits = opt_dict['follow_speed_limits']

        # ============ 初始化路点 ============
        self._init_waypoints()

    def reset_vehicle(self):
        """重置自车（ego-vehicle）"""
        self._vehicle = None

    def _init_waypoints(self):
        """
        初始化路点队列
        
        功能：
        1. 获取车辆当前位置的路点
        2. 将初始路点添加到队列
        """
        # 计算车辆当前位置的路点
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # 设置为目标路点，动作为车道跟随
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        # 添加到路点队列
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed):
        """
        修改目标速度（仅供外部参考使用）
        
        :param speed: 新的目标速度 (Km/h)
        """
        if self._follow_speed_limits:
            print("警告: 最大速度当前设置为跟随速度限制。"
                  "使用 'follow_speed_limits' 来停用此功能")
        self._target_speed = speed

    def follow_speed_limits(self, value=True):
        """
        激活一个标志，使最大速度根据速度限制动态变化
        
        :param value: 布尔值，True=启用，False=禁用
        """
        self._follow_speed_limits = value

    def _compute_next_waypoints(self, k=1):
        """
        向轨迹队列添加新的路点（自动生成模式）
        
        功能：
        1. 从队列最后一个路点开始
        2. 获取下一个可能的路点（可能有多个选项，如交叉口）
        3. 如果只有一个选项：选择它，动作为LANEFOLLOW
        4. 如果有多个选项：随机选择一个方向
        5. 将选择的路点添加到队列
        
        :param k: 要计算的路点数量
        """
        # 检查队列是否会溢出
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            # 获取队列中最后一个路点
            last_waypoint = self._waypoints_queue[-1][0]
            # 获取下一个路点选项（距离为采样半径）
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                # 没有下一个路点了（可能到达地图边界）
                break
            elif len(next_waypoints) == 1:
                # 只有一个选项 ==> 车道跟随
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # 有多个选项（如交叉口）==> 随机选择
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(road_option)]

            # 将选择的路点添加到队列
            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan, stop_waypoint_creation=True, clean_queue=True):
        """
        向局部规划器添加新的全局规划路径
        
        功能：
        1. 可选择清空现有队列
        2. 如果新路径长度超过队列容量，则扩展队列
        3. 将全局路径中的所有路点添加到队列
        4. 设置是否停止自动创建路点
        
        :param current_plan: 全局路径，格式为 [(carla.Waypoint, RoadOption), ...]
        :param stop_waypoint_creation: 是否停止自动创建随机路点
        :param clean_queue: 是否清空现有队列（True=替换，False=追加）
        """
        # 是否清空现有路径
        if clean_queue:
            self._waypoints_queue.clear()

        # 如果新计划长度超过队列最大长度，重新创建更大的队列
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            # 复制现有路点到新队列
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        # 将全局路径添加到队列
        for elem in current_plan:
            self._waypoints_queue.append(elem)

        # 设置是否停止自动路点创建
        self._stop_waypoint_creation = stop_waypoint_creation

    def set_offset(self, offset):
        """
        设置车辆的车道偏移量
        
        :param offset: 偏移量（米），正值向右，负值向左，0为车道中心
        """
        self._offset = offset

    def run_step(self, debug=False):
        """
        执行一步局部规划，更新路点队列并返回目标路点信息
        
        核心工作流程：
        ┌─────────────────────────────────────────┐
        │ 1. 检查是否需要跟随速度限制             │
        ├─────────────────────────────────────────┤
        │ 2. 补充路点队列（如果路点太少）         │
        ├─────────────────────────────────────────┤
        │ 3. 清理已通过的路点                     │
        │    - 计算动态距离阈值                   │
        │    - 移除距离小于阈值的路点             │
        ├─────────────────────────────────────────┤
        │ 4. 获取目标路点信息                     │
        │    - 如果队列为空：返回None             │
        │    - 否则：返回目标路点和相关信息       │
        └─────────────────────────────────────────┘
        
        :param debug: 是否激活路点调试可视化
        :return: 目标路点信息字典，包含：
            - 'target_waypoint': 目标路点 (carla.Waypoint 或 None)
            - 'target_road_option': 道路选项 (RoadOption)
            - 'target_speed': 建议目标速度 (km/h)
            - 'queue_length': 剩余路点数量
            - 'is_empty': 队列是否为空 (bool)
        
        使用示例：
        ```python
        # 获取目标路点信息
        target_info = local_planner.run_step()
        
        if not target_info['is_empty']:
            # 使用外部控制器计算控制指令
            throttle, brake, steer = external_controller.compute(
                target_waypoint=target_info['target_waypoint'],
                target_speed=target_info['target_speed'],
                vehicle=vehicle
            )
            
            # 应用控制
            control = carla.VehicleControl()
            control.throttle = throttle
            control.brake = brake
            control.steer = steer
            vehicle.apply_control(control)
        ```
        """
        # ========== 步骤1: 检查速度限制 ==========
        if self._follow_speed_limits:
            # 如果启用了跟随速度限制，使用当前道路的速度限制
            self._target_speed = self._vehicle.get_speed_limit()

        # ========== 步骤2: 补充路点队列 ==========
        # 如果没有停止自动创建且队列路点太少，则补充路点
        if not self._stop_waypoint_creation and len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # ========== 步骤3: 清理已通过的路点 ==========
        # 获取车辆当前位置和速度
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6  # 转换为 m/s
        
        # 计算动态最小距离阈值（速度越快，阈值越大）
        # 公式: min_distance = base + ratio × speed
        # 例如: 3.0 + 0.5 × 16.7(m/s) = 11.35米 (60km/h时)
        self._min_distance = self._base_min_distance + self._distance_ratio * vehicle_speed

        # 统计需要移除的路点数量
        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            # 特殊处理：保留最后一个路点（即使很近也不移除）
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # 最后一个路点的阈值设为1米
            else:
                min_distance = self._min_distance

            # 如果车辆到路点的距离小于阈值，标记为需要移除
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break  # 只移除连续的已通过路点

        # 从队列左侧移除已通过的路点
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # ========== 步骤4: 获取目标路点信息 ==========
        if len(self._waypoints_queue) == 0:
            # 情况1: 队列为空
            target_info = {
                'target_waypoint': None,
                'target_road_option': RoadOption.VOID,
                'target_speed': 0.0,
                'queue_length': 0,
                'is_empty': True
            }
            self.target_waypoint = None
            self.target_road_option = RoadOption.VOID
        else:
            # 情况2: 队列有路点
            # 获取队列第一个路点作为目标
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            
            target_info = {
                'target_waypoint': self.target_waypoint,
                'target_road_option': self.target_road_option,
                'target_speed': self._target_speed,
                'queue_length': len(self._waypoints_queue),
                'is_empty': False
            }

        # ========== 步骤5: 调试可视化（可选）==========
        if debug and self.target_waypoint is not None:
            # 在CARLA中绘制目标路点（用于调试）
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        # 返回目标路点信息
        return target_info

    def apply_control(self, throttle, brake, steer):
        """
        接收外部控制值并应用到车辆
        
        这是一个便捷方法，用于接收外部计算的控制值并直接应用到车辆
        
        :param throttle: 油门值 [0.0, 1.0]
        :param brake: 刹车值 [0.0, 1.0]
        :param steer: 转向值 [-1.0, +1.0]
        
        使用示例：
        ```python
        # 外部控制器计算控制值
        throttle, brake, steer = external_controller.compute(...)
        
        # 应用到车辆
        local_planner.apply_control(throttle, brake, steer)
        ```
        """
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steer)
        control.hand_brake = False
        control.manual_gear_shift = False
        
        self._vehicle.apply_control(control)

    def get_target_waypoint_info(self):
        """
        获取当前目标路点的详细信息（不更新路点队列）
        
        与 run_step() 的区别：
        - run_step(): 更新路点队列并返回目标信息
        - get_target_waypoint_info(): 仅返回当前目标信息，不更新队列
        
        :return: 目标路点信息字典
        """
        if self.target_waypoint is None:
            return {
                'target_waypoint': None,
                'target_road_option': RoadOption.VOID,
                'target_speed': 0.0,
                'target_location': None,
                'target_rotation': None,
                'queue_length': len(self._waypoints_queue),
                'is_empty': True
            }
        else:
            return {
                'target_waypoint': self.target_waypoint,
                'target_road_option': self.target_road_option,
                'target_speed': self._target_speed,
                'target_location': self.target_waypoint.transform.location,
                'target_rotation': self.target_waypoint.transform.rotation,
                'queue_length': len(self._waypoints_queue),
                'is_empty': False
            }

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        获取前方指定步数位置的路点和方向
        
        用途：预判前方道路情况（如即将进入的路口）
        
        :param steps: 前瞻步数（默认3步）
        :return: (waypoint, road_option) 或 (None, RoadOption.VOID)
        """
        if len(self._waypoints_queue) > steps:
            # 队列足够长，返回第steps个路点
            return self._waypoints_queue[steps]
        else:
            # 队列不够长，返回最后一个路点
            try:
                wpt, direction = self._waypoints_queue[-1]
                return wpt, direction
            except IndexError:
                # 队列为空，返回None
                return None, RoadOption.VOID

    def get_plan(self):
        """
        获取当前的局部规划路径
        
        :return: 路点队列 deque[(waypoint, RoadOption), ...]
        """
        return self._waypoints_queue

    def done(self):
        """
        检查规划器是否已完成（即是否到达目的地）
        
        :return: 布尔值，True表示队列为空（到达目的地）
        """
        return len(self._waypoints_queue) == 0


# ============================================================
# 辅助函数（模块级函数）
# ============================================================

def _retrieve_options(list_waypoints, current_waypoint):
    """
    计算当前活动路点与多个候选路点之间的连接类型
    
    功能：
    在交叉路口有多个可选方向时，判断每个方向是左转、右转还是直行
    
    工作原理：
    1. 对每个候选路点，获取其后续路点（前瞻3米）
    2. 计算当前路点到后续路点的角度变化
    3. 根据角度判断转向类型
    
    :param list_waypoints: 候选目标路点列表
    :param current_waypoint: 当前活动路点
    :return: RoadOption枚举列表，表示每个候选路点对应的连接类型
    
    示例：
    输入: [wp1, wp2, wp3], current_wp
    输出: [RoadOption.LEFT, RoadOption.STRAIGHT, RoadOption.RIGHT]
    """
    options = []
    for next_waypoint in list_waypoints:
        # 这里需要获取后续路点，因为有时我们连接到
        # 交叉口的起点，此时角度变化很小
        # 通过前瞻3米，可以更准确地判断转向类型
        next_next_waypoint = next_waypoint.next(3.0)[0]
        
        # 计算连接类型（左转/右转/直行）
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    计算活动路点与目标路点之间的拓扑连接类型
    
    判断逻辑：
    1. 计算两个路点的航向角（yaw）差值
    2. 根据角度差判断转向类型：
       - 角度差 < 35° 或 > 145° ==> 直行 (STRAIGHT)
       - 角度差 > 90°            ==> 左转 (LEFT)
       - 其他                    ==> 右转 (RIGHT)
    
    :param current_waypoint: 当前活动路点
    :param next_waypoint: 目标路点
    :param threshold: 判断阈值（度），默认35度
    :return: RoadOption枚举值 (STRAIGHT/LEFT/RIGHT)
    
    角度计算示例：
    当前朝向: 0° (北)
    目标朝向: 45° (东北)
    角度差: 45°
    判断: 45° > 35° 且 < 90° ==> RIGHT (右转)
    
    当前朝向: 0° (北)
    目标朝向: 10° (略偏东北)
    角度差: 10°
    判断: 10° < 35° ==> STRAIGHT (直行)
    
    当前朝向: 0° (北)
    目标朝向: 135° (东南)
    角度差: 135°
    判断: 135° > 90° ==> LEFT (左转)
    """
    # 获取目标路点的航向角并标准化到 [0, 360)
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    # 获取当前路点的航向角并标准化到 [0, 360)
    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    # 计算角度差并标准化到 [0, 180)
    diff_angle = (n - c) % 180.0
    
    # 根据角度差判断转向类型
    if diff_angle < threshold or diff_angle > (180 - threshold):
        # 角度差很小（接近0°）或很大（接近180°）==> 直行
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        # 角度差 > 90° ==> 左转
        return RoadOption.LEFT
    else:
        # 其他情况 ==> 右转
        return RoadOption.RIGHT
