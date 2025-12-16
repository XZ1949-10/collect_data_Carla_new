#!/usr/bin/env python
# coding=utf-8
'''
红绿灯场景诊断脚本
专门用于分析模型在红绿灯场景的行为

使用方法:
    python diagnose_traffic_light.py --model-path ./model/your_model.pth

功能:
1. 实时 Grad-CAM 可视化 - 查看模型关注的区域
2. 刹车预测分析 - 统计刹车行为
3. 分支输出对比 - 分析所有分支的差异
4. 保存诊断帧 - 用于离线分析
'''

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch

# 设置编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import carla

from carla_config import *
from carla_sensors import SensorManager
from carla_model_loader import ModelLoader
from carla_image_processor import ImageProcessor
from carla_vehicle_controller import VehicleController
from carla_vehicle_spawner import VehicleSpawner
from navigation_planner_adapter import NavigationPlannerAdapter
from carla_interpretability import InterpretabilityVisualizer, GradCAM, BrakeAnalyzer


class TrafficLightDiagnoser:
    """红绿灯场景诊断器"""
    
    def __init__(self, model_path, host='localhost', port=2000, town='Town01', 
                 gpu_id=0, save_dir='./diagnose_output'):
        self.host = host
        self.port = port
        self.town = town
        self.save_dir = save_dir
        
        # 设备
        self.device = torch.device(
            f'cuda:{gpu_id}' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu'
        )
        
        # 模块
        self.model_loader = ModelLoader(model_path, self.device)
        self.image_processor = ImageProcessor(self.device, enable_crop=True)
        self.vehicle_controller = VehicleController()
        
        # CARLA 对象
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensor_manager = None
        self.navigation_planner = None
        self.vehicle_spawner = None
        
        # 模型
        self.model = None
        
        # 可解释性工具
        self.interp_viz = None
        self.grad_cam = None
        self.brake_analyzer = BrakeAnalyzer(history_size=200)
        
        # 状态
        self.frame_count = 0
        self.all_branch_predictions = None
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"诊断器初始化完成 - 设备: {self.device}")
        print(f"诊断结果将保存到: {save_dir}")
    
    def load_model(self, net_structure=2):
        """加载模型"""
        self.model = self.model_loader.load()
        self.model.eval()
        
        # 初始化可解释性工具
        self.grad_cam = GradCAM(self.model)
        self.interp_viz = InterpretabilityVisualizer(
            self.model, self.device, self.save_dir
        )
        
        print("✅ 模型和可解释性工具加载完成")
    
    def connect_carla(self):
        """连接 CARLA"""
        print(f"正在连接到 CARLA 服务器 {self.host}:{self.port}...")
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        
        print(f"正在加载地图 {self.town}...")
        self.world = self.client.load_world(self.town)
        
        # 同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SYNC_MODE_DELTA_SECONDS
        self.world.apply_settings(settings)
        
        self.vehicle_spawner = VehicleSpawner(self.world)
        self.navigation_planner = NavigationPlannerAdapter(
            self.world, sampling_resolution=ROUTE_SAMPLING_RESOLUTION
        )
        
        print("✅ CARLA 连接成功")
    
    def spawn_vehicle(self, spawn_index=None, dest_index=None):
        """生成车辆"""
        self.vehicle = self.vehicle_spawner.spawn('vehicle.tesla.model3', spawn_index)
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        
        for _ in range(3):
            self.world.tick()
        
        # 设置目的地
        spawn_points = self.world.get_map().get_spawn_points()
        if dest_index is not None and 0 <= dest_index < len(spawn_points):
            destination = spawn_points[dest_index].location
            self.navigation_planner.set_destination(self.vehicle, destination)
        else:
            self.navigation_planner.set_random_destination(self.vehicle)
        
        self.sensor_manager.setup_camera()
        print("✅ 车辆生成完成")
    
    def predict(self, img_tensor, speed_tensor, current_command):
        """模型预测（保存所有分支输出）"""
        with torch.no_grad():
            pred_control, pred_speed, log_var_control, log_var_speed = \
                self.model(img_tensor, speed_tensor)
        
        pred_control = pred_control.cpu().numpy()[0]
        self.all_branch_predictions = pred_control.copy()
        
        branch_idx = current_command - 2
        start_idx = branch_idx * 3
        
        steer = float(pred_control[start_idx])
        throttle = float(pred_control[start_idx + 1])
        brake = float(pred_control[start_idx + 2])
        
        # Clip
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'pred_speed': pred_speed.cpu().numpy()[0][0] * MAX_SPEED_KMH,
            'pred_speed_normalized': pred_speed.cpu().numpy()[0][0],
        }
    
    def run_diagnosis(self, duration=120, save_interval=10):
        """
        运行诊断
        
        参数:
            duration: 运行时长（秒）
            save_interval: 保存间隔（帧）
        """
        print(f"\n{'='*70}")
        print("开始红绿灯场景诊断")
        print(f"{'='*70}")
        print(f"运行时长: {duration}秒")
        print(f"保存间隔: 每{save_interval}帧")
        print("按 'q' 退出, 's' 手动保存当前帧, 'p' 打印统计")
        print(f"{'='*70}\n")
        
        # 等待摄像头
        print("等待摄像头数据...")
        while not self.sensor_manager.has_image():
            self.world.tick()
            time.sleep(0.01)
        print("摄像头就绪！\n")
        
        start_time = time.time()
        self.frame_count = 0
        
        # 创建窗口
        cv2.namedWindow('Traffic Light Diagnosis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Light Diagnosis', 1200, 700)
        
        try:
            while True:
                # 检查超时
                if duration > 0 and time.time() - start_time > duration:
                    print(f"\n已运行 {duration} 秒，停止诊断")
                    break
                
                self.world.tick()
                
                if not self.sensor_manager.has_image():
                    continue
                
                # 更新导航
                self.navigation_planner.run_step()
                current_command = self.navigation_planner.get_navigation_command(self.vehicle)
                
                # 获取数据
                current_image = self.sensor_manager.get_latest_image()
                current_speed = self.vehicle_controller.get_speed_normalized(
                    self.vehicle, SPEED_NORMALIZATION_MPS
                )
                
                # 预处理
                img_tensor = self.image_processor.preprocess(current_image)
                speed_tensor = torch.FloatTensor([[current_speed]]).to(self.device)
                
                # 预测
                control_result = self.predict(img_tensor, speed_tensor, current_command)
                control_result['speed_normalized'] = current_speed
                
                # 获取处理后的图像（模型输入）
                model_input_image = self.image_processor.get_processed_image(current_image)
                
                # 可解释性分析
                analysis_results = self.interp_viz.analyze_frame(
                    img_tensor, speed_tensor, model_input_image,
                    control_result, current_command
                )
                
                # 创建仪表板（使用render_dashboard方法）
                dashboard = self.interp_viz.render_dashboard(
                    model_input_image, analysis_results, 
                    control_result, current_command
                )
                
                # 添加分支对比
                dashboard = self._add_branch_comparison(dashboard, current_command)
                
                # 添加红绿灯检测状态
                dashboard = self._add_traffic_light_status(dashboard)
                
                # 显示
                cv2.imshow('Traffic Light Diagnosis', dashboard)
                
                # 应用控制（可选：不应用控制，只观察）
                self.vehicle_controller.apply_control(
                    self.vehicle,
                    control_result['steer'],
                    control_result['throttle'],
                    control_result['brake']
                )
                
                self.frame_count += 1
                
                # 定期保存
                if self.frame_count % save_interval == 0:
                    self._save_diagnosis_frame(dashboard, model_input_image, 
                                               analysis_results, control_result)
                
                # 打印状态
                if self.frame_count % 20 == 0:
                    self._print_status(current_speed, control_result, current_command)
                
                # 键盘处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户退出")
                    break
                elif key == ord('s'):
                    self._save_diagnosis_frame(dashboard, model_input_image,
                                               analysis_results, control_result, manual=True)
                elif key == ord('p'):
                    self._print_statistics()
                
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            cv2.destroyAllWindows()
            self._print_final_report()
    
    def _add_branch_comparison(self, dashboard, current_command):
        """添加分支对比到仪表板"""
        if self.all_branch_predictions is None:
            return dashboard
        
        x_start = 450
        y_start = 400
        
        cv2.putText(dashboard, "All Branches Comparison:", (x_start, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        branch_names = ['Follow', 'Left', 'Right', 'Straight']
        
        for i, name in enumerate(branch_names):
            y = y_start + 20 + i * 20
            start_idx = i * 3
            
            steer = self.all_branch_predictions[start_idx]
            throttle = self.all_branch_predictions[start_idx + 1]
            brake = self.all_branch_predictions[start_idx + 2]
            
            # 高亮当前分支
            color = (0, 255, 255) if i == current_command - 2 else (150, 150, 150)
            marker = ">>>" if i == current_command - 2 else "   "
            
            text = f"{marker} {name:8s}: S={steer:+.2f} T={throttle:.2f} B={brake:.2f}"
            cv2.putText(dashboard, text, (x_start, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        return dashboard
    
    def _add_traffic_light_status(self, dashboard):
        """添加红绿灯检测状态"""
        # 检测附近的红绿灯
        if self.vehicle is None:
            return dashboard
        
        vehicle_loc = self.vehicle.get_location()
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        
        nearest_tl = None
        nearest_dist = float('inf')
        
        for tl in traffic_lights:
            dist = vehicle_loc.distance(tl.get_location())
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_tl = tl
        
        # 显示红绿灯状态
        x, y = 640, 200
        cv2.putText(dashboard, "Traffic Light:", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        if nearest_tl and nearest_dist < 50:
            state = nearest_tl.get_state()
            state_name = str(state).split('.')[-1]
            
            # 颜色
            if 'Red' in state_name:
                color = (0, 0, 255)
            elif 'Yellow' in state_name:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            
            cv2.putText(dashboard, f"{state_name} ({nearest_dist:.1f}m)", (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 如果是红灯但没刹车，警告
            if 'Red' in state_name and nearest_dist < 30:
                if self.all_branch_predictions is not None:
                    # 检查所有分支的刹车值
                    max_brake = max(self.all_branch_predictions[2], 
                                    self.all_branch_predictions[5],
                                    self.all_branch_predictions[8],
                                    self.all_branch_predictions[11])
                    if max_brake < 0.3:
                        cv2.putText(dashboard, "WARNING: Red light but low brake!", 
                                    (x, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(dashboard, "None nearby", (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return dashboard
    
    def _save_diagnosis_frame(self, dashboard, model_input, results, control, manual=False):
        """保存诊断帧"""
        prefix = "manual" if manual else "auto"
        timestamp = time.strftime("%H%M%S")
        
        # 保存仪表板
        filename = f"{prefix}_{self.frame_count:06d}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, dashboard)
        
        # 保存 Grad-CAM（使用brake_cam键，并生成叠加图像）
        brake_cam = results.get('brake_cam')
        if brake_cam is not None:
            # 生成热力图叠加图像
            heatmap = cv2.applyColorMap(np.uint8(255 * brake_cam), cv2.COLORMAP_JET)
            # 将model_input转换为BGR格式
            if len(model_input.shape) == 3 and model_input.shape[2] == 3:
                model_input_bgr = cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR)
            else:
                model_input_bgr = model_input
            # 调整热力图尺寸以匹配输入图像
            heatmap_resized = cv2.resize(heatmap, (model_input_bgr.shape[1], model_input_bgr.shape[0]))
            # 叠加
            cam_overlay = cv2.addWeighted(model_input_bgr, 0.4, heatmap_resized, 0.6, 0)
            
            cam_filename = f"gradcam_{self.frame_count:06d}.png"
            cam_filepath = os.path.join(self.save_dir, cam_filename)
            cv2.imwrite(cam_filepath, cam_overlay)
        
        if manual:
            print(f"✅ 手动保存: {filepath}")
    
    def _print_status(self, speed, control, command):
        """打印状态"""
        cmd_names = {2: 'Follow', 3: 'Left', 4: 'Right', 5: 'Straight'}
        actual_speed = speed * SPEED_NORMALIZATION_MPS
        
        print(f"[Frame {self.frame_count:5d}] "
              f"Cmd: {cmd_names.get(command, '?'):8s} | "
              f"Spd: {actual_speed:5.1f} km/h | "
              f"Str: {control['steer']:+.3f} | "
              f"Thr: {control['throttle']:.3f} | "
              f"Brk: {control['brake']:.3f}")
    
    def _print_statistics(self):
        """打印统计信息"""
        stats = self.brake_analyzer.get_statistics()
        
        print(f"\n{'='*50}")
        print("刹车行为统计")
        print(f"{'='*50}")
        print(f"总帧数: {stats.get('total_frames', 0)}")
        print(f"刹车帧占比 (>0.1): {stats.get('brake_ratio', 0)*100:.1f}%")
        print(f"急刹车帧占比 (>0.5): {stats.get('hard_brake_ratio', 0)*100:.1f}%")
        print(f"平均刹车值: {stats.get('avg_brake', 0):.3f}")
        print(f"最大刹车值: {stats.get('max_brake', 0):.3f}")
        print(f"{'='*50}\n")
    
    def _print_final_report(self):
        """打印最终报告"""
        print(f"\n{'='*70}")
        print("诊断报告")
        print(f"{'='*70}")
        
        stats = self.brake_analyzer.get_statistics()
        
        print(f"\n📊 刹车行为分析:")
        print(f"   • 总帧数: {stats.get('total_frames', 0)}")
        print(f"   • 刹车帧占比: {stats.get('brake_ratio', 0)*100:.1f}%")
        print(f"   • 急刹车帧占比: {stats.get('hard_brake_ratio', 0)*100:.1f}%")
        print(f"   • 平均刹车值: {stats.get('avg_brake', 0):.3f}")
        
        # 诊断建议
        print(f"\n🔍 诊断建议:")
        
        brake_ratio = stats.get('brake_ratio', 0)
        if brake_ratio < 0.05:
            print("   ⚠️ 刹车帧占比过低 (<5%)，模型可能没有学会刹车行为")
            print("   建议: 检查训练数据中刹车样本的比例和质量")
        elif brake_ratio < 0.15:
            print("   ⚠️ 刹车帧占比偏低，模型刹车行为可能不够积极")
        else:
            print("   ✅ 刹车帧占比正常")
        
        avg_brake = stats.get('avg_brake', 0)
        if avg_brake < 0.1:
            print("   ⚠️ 平均刹车值过低，模型刹车力度不足")
        
        print(f"\n📁 诊断结果已保存到: {self.save_dir}")
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        # 清理可解释性模块（释放钩子和内存）
        if hasattr(self, 'grad_cam') and self.grad_cam is not None:
            self.grad_cam.cleanup()
            print("  - GradCAM 已清理")
        
        if hasattr(self, 'interp_viz') and self.interp_viz is not None:
            self.interp_viz.cleanup()
            print("  - 可解释性可视化器已清理")
        
        if self.sensor_manager:
            self.sensor_manager.cleanup()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        print("清理完成！")


def main():
    parser = argparse.ArgumentParser(description='红绿灯场景诊断工具')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--net-structure', type=int, default=2,
                        help='网络结构类型')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA 服务器地址')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA 服务器端口')
    parser.add_argument('--town', type=str, default='Town01',
                        help='地图名称')
    parser.add_argument('--spawn-index', type=int, default=None,
                        help='起点索引')
    parser.add_argument('--dest-index', type=int, default=None,
                        help='终点索引')
    parser.add_argument('--duration', type=int, default=120,
                        help='运行时长（秒）')
    parser.add_argument('--save-dir', type=str, default='./diagnose_output',
                        help='保存目录')
    parser.add_argument('--save-interval', type=int, default=30,
                        help='自动保存间隔（帧）')
    
    args = parser.parse_args()
    
    # 处理模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(script_dir, args.model_path)
    
    # 创建诊断器
    diagnoser = TrafficLightDiagnoser(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        town=args.town,
        gpu_id=args.gpu,
        save_dir=args.save_dir
    )
    
    try:
        diagnoser.load_model(args.net_structure)
        diagnoser.connect_carla()
        diagnoser.spawn_vehicle(args.spawn_index, args.dest_index)
        
        time.sleep(1.0)
        
        diagnoser.run_diagnosis(
            duration=args.duration,
            save_interval=args.save_interval
        )
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        diagnoser.cleanup()


if __name__ == '__main__':
    main()
