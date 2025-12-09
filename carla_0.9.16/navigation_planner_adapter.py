#!/usr/bin/env python
# coding=utf-8
'''
作者: AI Assistant
日期: 2025-11-25
更新: 2025-12-04
说明: NavigationPlanner 适配器
      使用 BasicAgent + LocalPlanner 获取导航命令
      与数据收集时的命令获取方式保持一致
      
更新内容（2025-12-04）：
      - 移植 command_based_data_collection.py 中的命令获取逻辑
      - 添加距离过滤：只有距离转弯点<15米才触发转弯命令
      - 添加命令持久化：转弯命令保持直到转弯完成
      - 添加多条件重置：方向盘回正+不在交叉口+队列全是LANEFOLLOW
'''

import random
import numpy as np
import carla
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from agents.navigation.basic_agent import BasicAgent


class NavigationPlannerAdapter:
    """
    NavigationPlanner 适配器类
    
    使用 BasicAgent + LocalPlanner 来获取导航命令，
    与数据收集时的命令获取方式保持一致，确保推理时命令与训练数据一致。
    
    关键改进（2025-12-04）：
    - 完全移植 command_based_data_collection.py 中的 _get_navigation_command() 逻辑
    - 添加距离过滤：只有距离转弯点<15米才触发转弯命令
    - 添加命令持久化：转弯命令保持直到转弯完成
    - 添加多条件重置：方向盘回正+不在交叉口+队列全是LANEFOLLOW
    """
    
    def __init__(self, world, sampling_resolution=2.0, target_speed=20.0):
        """
        初始化导航规划器
        
        参数:
            world: carla.World 实例
            sampling_resolution: 路径采样分辨率（米）
            target_speed: 目标速度（km/h），用于 BasicAgent
        """
        self._world = world
        self._map = world.get_map()
        self._sampling_resolution = sampling_resolution
        self._target_speed = target_speed
        
        # 创建全局路径规划器（用于路线信息计算）
        self._global_planner = GlobalRoutePlanner(self._map, sampling_resolution)
        
        # BasicAgent（用于获取导航命令，与数据收集一致）
        self._agent = None  # 在 set_destination 时初始化
        self._vehicle = None
        
        # 路线相关（用于进度计算）
        self._route = []  # 路线：[(waypoint, road_option), ...]
        self._current_waypoint_index = 0
        self._destination = None
        
        # 命令映射：RoadOption -> 训练数据命令编码
        # 与 command_based_data_collection.py 完全一致
        self._road_option_to_command = {
            RoadOption.LEFT: 3,           # 左转
            RoadOption.RIGHT: 4,          # 右转
            RoadOption.STRAIGHT: 5,       # 直行
            RoadOption.LANEFOLLOW: 2,     # 跟车/车道保持
            RoadOption.CHANGELANELEFT: 2, # 变道左 -> 跟车
            RoadOption.CHANGELANERIGHT: 2,# 变道右 -> 跟车
            RoadOption.VOID: 2            # 未定义 -> 跟车
        }
        
        # ========== 转弯命令持久化状态 ==========
        self._last_turn_command = None   # 上一次检测到的转弯命令
        self._turn_command_frames = 0    # 转弯命令持续的帧数
        self._max_turn_frames = 200      # 转弯命令最大持续帧数（约10秒@20fps）
        
        # ========== 转弯命令重置阈值（可调整以控制持久时间） ==========
        self._steering_threshold = 0.02       # 方向盘回正阈值（越小越严格，需要完全回正）
        self._reset_frames_outside_junction = 60   # 交叉口外重置帧数（约3秒）
        self._reset_frames_inside_junction = 120   # 交叉口内重置帧数（约6秒）
        
        print(f"NavigationPlannerAdapter 初始化完成 (采样分辨率: {sampling_resolution}m)")
        print(f"  ✅ 使用与数据收集完全一致的命令获取逻辑")
        print(f"  ✅ 距离过滤 + 命令持久化 + 多条件重置")
    
    def set_destination(self, vehicle, destination):
        """
        设置目的地并规划路线
        
        使用 BasicAgent 来管理路线和获取导航命令，
        与数据收集时的方式保持一致。
        
        参数:
            vehicle: carla.Vehicle 实例
            destination: carla.Location 目的地位置
            
        返回:
            bool: 是否成功规划路线
        """
        try:
            self._vehicle = vehicle
            start_location = vehicle.get_location()
            
            # 使用全局规划器规划路线（用于进度计算）
            self._route = self._global_planner.trace_route(start_location, destination)
            
            if not self._route or len(self._route) == 0:
                print("⚠️ 无法规划路线：路线为空")
                return False
            
            self._destination = destination
            self._current_waypoint_index = 0
            
            # 创建 BasicAgent（与数据收集时的配置一致）
            opt_dict = {
                'target_speed': self._target_speed,
                'sampling_resolution': 1.0,  # 与数据收集一致
                'lateral_control_dict': {
                    'K_P': 1.5,
                    'K_I': 0.0,
                    'K_D': 0.05,
                    'dt': 0.05
                },
                'longitudinal_control_dict': {
                    'K_P': 1.0,
                    'K_I': 0.05,
                    'K_D': 0.0,
                    'dt': 0.05
                },
                'max_steering': 0.8,
                'max_throttle': 0.75,
                'max_brake': 0.5,
                'base_min_distance': 2.0,
                'distance_ratio': 0.3
            }
            
            self._agent = BasicAgent(
                vehicle,
                target_speed=self._target_speed,
                opt_dict=opt_dict,
                map_inst=self._map
            )
            
            # 设置目的地（BasicAgent 会自动规划路线并管理 LocalPlanner）
            self._agent.set_destination(destination, start_location=start_location)
            
            # 计算路线总长度
            total_distance = self._calculate_route_length()
            
            print(f"✓ 路线规划成功：{len(self._route)} 个路点，总长度 {total_distance:.1f}m")
            print(f"  ✅ BasicAgent 已创建并设置目的地")
            return True
            
        except Exception as e:
            print(f"⚠️ 路线规划失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_random_destination(self, vehicle):
        """
        设置随机目的地
        
        参数:
            vehicle: carla.Vehicle 实例
            
        返回:
            bool: 是否成功规划路线
        """
        spawn_points = self._map.get_spawn_points()
        
        if len(spawn_points) == 0:
            print("⚠️ 地图上没有可用的生成点")
            return False
        
        # 随机选择一个目的地（避免选择当前位置附近）
        current_location = vehicle.get_location()
        
        # 过滤掉距离太近的点（小于50米）
        valid_destinations = [
            sp.location for sp in spawn_points
            if current_location.distance(sp.location) > 50.0
        ]
        
        if not valid_destinations:
            # 如果所有点都太近，就使用所有点
            valid_destinations = [sp.location for sp in spawn_points]
        
        destination = random.choice(valid_destinations)
        
        return self.set_destination(vehicle, destination)
    
    def get_navigation_command(self, vehicle):
        """
        获取当前导航命令
        
        【重要】完全移植自 command_based_data_collection.py 的 _get_navigation_command()
        
        改进策略（与数据收集完全一致）：
        1. 缩小搜索范围：只搜索前5个路点（约10米），避免过早检测到转弯
        2. 基于距离的命令触发：只有当距离路口足够近（<15米）时才返回转弯命令
        3. 转弯命令持久化：当检测到转弯命令时保存，直到转弯完成
        4. 多条件重置：方向盘回正+不在交叉口+队列全是LANEFOLLOW
        
        参数:
            vehicle: carla.Vehicle 实例
            
        返回:
            int: 命令编码 (2=跟车, 3=左转, 4=右转, 5=直行)
        """
        # 如果没有 BasicAgent，返回默认命令
        if self._agent is None:
            return 2  # 默认返回跟车命令
        
        try:
            local_planner = self._agent.get_local_planner()
            if local_planner is None:
                return 2
            
            waypoints_queue = local_planner.get_plan()
            if waypoints_queue is None or len(waypoints_queue) == 0:
                return 2
            
            # ========== 步骤1：缩小搜索范围，避免过早检测转弯 ==========
            # 只搜索前5个路点（约10米，因为sampling_radius=2.0）
            search_range = min(5, len(waypoints_queue))
            
            found_turn_command = None
            turn_waypoint_index = -1
            
            for i in range(search_range):
                _, direction = waypoints_queue[i]
                
                # 如果找到转弯/直行命令，记录位置
                if direction in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
                    found_turn_command = direction
                    turn_waypoint_index = i
                    break
                
                # 如果是变道命令或LANEFOLLOW，继续查找
                if direction in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT, RoadOption.LANEFOLLOW]:
                    continue
            
            # ========== 步骤2：基于距离的命令触发 ==========
            if found_turn_command is not None and turn_waypoint_index >= 0:
                # 计算到转弯路点的距离
                turn_waypoint = waypoints_queue[turn_waypoint_index][0]
                vehicle_location = vehicle.get_location()
                distance_to_turn = vehicle_location.distance(turn_waypoint.transform.location)
                
                # 只有距离小于15米时才触发转弯命令
                if distance_to_turn < 15.0:
                    self._last_turn_command = self._road_option_to_command.get(found_turn_command, 2)
                    self._turn_command_frames = 0
                    return self._last_turn_command
                else:
                    # 距离太远，返回Follow
                    return 2
            
            # ========== 步骤3：检查是否已离开交叉口（命令持久化） ==========
            if self._last_turn_command is not None and self._last_turn_command != 2:
                # 检查队列是否全是 LANEFOLLOW
                check_range = min(5, len(waypoints_queue))
                all_lane_follow = all(
                    waypoints_queue[i][1] == RoadOption.LANEFOLLOW 
                    for i in range(check_range)
                )
                
                # 检查车辆是否在交叉口内
                current_waypoint = self._map.get_waypoint(vehicle.get_location())
                is_in_junction = current_waypoint.is_junction if current_waypoint else False
                
                # 获取方向盘角度
                steering = abs(vehicle.get_control().steer)
                
                # 增加帧计数
                self._turn_command_frames += 1
                
                # 判断是否应该重置命令
                should_reset = False
                
                # 条件1：超过最大帧数，强制重置
                if self._turn_command_frames >= self._max_turn_frames:
                    should_reset = True
                
                # 条件2：队列全是 LANEFOLLOW + 不在交叉口内 + 方向盘回正
                elif all_lane_follow and not is_in_junction and steering < self._steering_threshold:
                    should_reset = True
                
                # 条件3：队列全是 LANEFOLLOW + 不在交叉口内 + 已持续足够帧数
                elif all_lane_follow and not is_in_junction and self._turn_command_frames > self._reset_frames_outside_junction:
                    should_reset = True
                
                # 条件4：队列全是 LANEFOLLOW + 已持续足够帧数（即使在交叉口内也重置）
                elif all_lane_follow and self._turn_command_frames > self._reset_frames_inside_junction:
                    should_reset = True
                
                if should_reset:
                    # 重置转弯/直行命令
                    self._last_turn_command = None
                    self._turn_command_frames = 0
                    return 2  # 返回 Follow
                else:
                    # 还在转弯/直行中，继续返回之前的命令
                    return self._last_turn_command
            
            # ========== 步骤4：降级处理 ==========
            incoming_wp, incoming_direction = local_planner.get_incoming_waypoint_and_direction(steps=3)
            
            if incoming_direction is not None and incoming_direction != RoadOption.VOID:
                road_option = incoming_direction
            else:
                road_option = local_planner.target_road_option
                if road_option is None:
                    road_option = RoadOption.LANEFOLLOW
            
            command = self._road_option_to_command.get(road_option, 2)
            return command
            
        except Exception as e:
            print(f"⚠️ 获取导航命令失败: {e}")
            return 2  # 默认返回跟车命令
    
    def run_step(self):
        """
        执行一步导航（更新 LocalPlanner 状态）
        
        【重要】必须调用此方法来更新 LocalPlanner 的状态，
        否则 target_road_option 不会更新
        
        返回:
            carla.VehicleControl: 控制命令（可选择是否使用）
        """
        if self._agent is None:
            return None
        
        try:
            # 调用 BasicAgent.run_step() 来更新 LocalPlanner 状态
            control = self._agent.run_step()
            return control
        except Exception as e:
            print(f"⚠️ run_step 失败: {e}")
            return None
    
    def get_route_info(self, vehicle):
        """
        获取路线信息
        
        参数:
            vehicle: carla.Vehicle 实例
            
        返回:
            dict: 包含路线信息的字典
        """
        if not self._route or len(self._route) == 0:
            return {
                'current_command': 2,
                'progress': 0.0,
                'remaining_distance': 0.0,
                'total_distance': 0.0
            }
        
        # 更新当前路点索引
        self._update_current_waypoint(vehicle)
        
        # 计算进度
        current_location = vehicle.get_location()
        
        # 计算已行驶距离
        traveled_distance = 0.0
        for i in range(self._current_waypoint_index):
            if i + 1 < len(self._route):
                wp1 = self._route[i][0].transform.location
                wp2 = self._route[i + 1][0].transform.location
                traveled_distance += wp1.distance(wp2)
        
        # 加上到当前路点的距离
        if self._current_waypoint_index < len(self._route):
            current_waypoint = self._route[self._current_waypoint_index][0]
            traveled_distance += current_location.distance(current_waypoint.transform.location)
        
        # 计算剩余距离
        remaining_distance = 0.0
        for i in range(self._current_waypoint_index, len(self._route) - 1):
            wp1 = self._route[i][0].transform.location
            wp2 = self._route[i + 1][0].transform.location
            remaining_distance += wp1.distance(wp2)
        
        # 总距离
        total_distance = traveled_distance + remaining_distance
        
        # 进度百分比
        progress = (traveled_distance / total_distance * 100.0) if total_distance > 0 else 0.0
        
        # 当前命令
        current_command = self.get_navigation_command(vehicle)
        
        return {
            'current_command': current_command,
            'progress': progress,
            'remaining_distance': remaining_distance,
            'total_distance': total_distance
        }
    
    def is_route_completed(self, vehicle, threshold=5.0):
        """
        检查是否到达目的地
        
        使用 BasicAgent.done() 方法，与数据收集一致
        
        参数:
            vehicle: carla.Vehicle 实例
            threshold: 距离阈值（米）
            
        返回:
            bool: 是否到达目的地
        """
        # 优先使用 BasicAgent 的 done() 方法
        if self._agent is not None:
            try:
                return self._agent.done()
            except Exception:
                pass
        
        # 回退方案：检查距离
        if not self._route or len(self._route) == 0:
            return True
        
        if self._destination is None:
            return True
        
        current_location = vehicle.get_location()
        distance_to_destination = current_location.distance(self._destination)
        
        return distance_to_destination < threshold
    
    def _update_current_waypoint(self, vehicle):
        """
        更新当前路点索引（找到最近的路点）
        
        用于进度计算，不影响命令获取
        
        参数:
            vehicle: carla.Vehicle 实例
        """
        if not self._route or len(self._route) == 0:
            return
        
        current_location = vehicle.get_location()
        
        # 从当前索引开始向前查找最近的路点
        min_distance = float('inf')
        best_index = self._current_waypoint_index
        
        # 搜索范围：当前索引 到 当前索引+20（避免全局搜索）
        search_start = self._current_waypoint_index
        search_end = min(self._current_waypoint_index + 20, len(self._route))
        
        for i in range(search_start, search_end):
            waypoint = self._route[i][0]
            distance = current_location.distance(waypoint.transform.location)
            
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        self._current_waypoint_index = best_index
    
    def _calculate_route_length(self):
        """
        计算路线总长度
        
        返回:
            float: 路线长度（米）
        """
        if not self._route or len(self._route) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self._route) - 1):
            wp1 = self._route[i][0].transform.location
            wp2 = self._route[i + 1][0].transform.location
            total_length += wp1.distance(wp2)
        
        return total_length
