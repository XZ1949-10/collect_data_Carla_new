#!/usr/bin/env python
# coding=utf-8
"""
模型可解释性与可视化模块（学术严谨版 v2.0）

功能：
1. Grad-CAM / Grad-CAM++ 热力图 - 定性可视化网络关注区域
2. 多层级 Grad-CAM - 支持选择不同卷积层以获得不同分辨率
3. 遮挡敏感性分析 (Occlusion Sensitivity) - 定量测量区域重要性
4. 积分梯度 (Integrated Gradients) - 严谨的归因方法
5. 删除/插入曲线 (Deletion/Insertion) - 评估解释质量的标准指标
6. 定量指标汇总 - 可用于学术论文的数值指标

参考文献：
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
- Grad-CAM++: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations", WACV 2018
- Integrated Gradients: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
- RISE: Petsiuk et al., "RISE: Randomized Input Sampling for Explanation", BMVC 2018

更新 v2.0:
- 仪表板尺寸改为 1920x1080
- 支持多层级 Grad-CAM（可选择中间层以获得更高分辨率）
- 添加 Grad-CAM++ 支持
- 改进可视化布局和信息展示
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import os
from datetime import datetime
import json


# ============================================================================
# 第一部分：基础分析工具
# ============================================================================

class GradCAM:
    """
    Grad-CAM / Grad-CAM++ 可视化
    
    参考: 
    - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
    - Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations", WACV 2018
    
    改进：
    - 支持选择不同的卷积层（更早的层有更高分辨率）
    - 支持 Grad-CAM++ 模式（对多目标场景更准确）
    - 支持上下文管理器，确保钩子在异常情况下也能被清理
    """

    def __init__(self, model, target_layer_index=-1, use_gradcam_pp=True):
        """
        参数:
            model: PyTorch模型
            target_layer_index: 目标卷积层索引
                -1: 最后一层（默认，分辨率最低）
                -3: 倒数第3层（推荐，平衡分辨率和语义）
                -5: 倒数第5层（更高分辨率）
            use_gradcam_pp: 是否使用 Grad-CAM++ 模式（默认True）
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.use_gradcam_pp = use_gradcam_pp
        self.target_layer_index = target_layer_index
        self._is_cleaned = False  # 标记是否已清理
        
        # 获取所有卷积层
        self.conv_layers = self._get_all_conv_layers(model.carla_net.conv_block)
        
        # 选择目标层
        self.target_layer = self.conv_layers[target_layer_index]
        
        # 注册钩子
        self.hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self.hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )
        
        # 打印层信息
        layer_info = f"第{len(self.conv_layers) + target_layer_index + 1}层" if target_layer_index < 0 else f"第{target_layer_index + 1}层"
        print(f"✅ Grad-CAM{'++' if use_gradcam_pp else ''} 初始化: 目标层={layer_info}/{len(self.conv_layers)}层")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保钩子被清理"""
        self.cleanup()
        return False  # 不抑制异常
    
    def __del__(self):
        """析构函数，作为最后的清理保障"""
        if not self._is_cleaned:
            self.cleanup()
    
    def _get_all_conv_layers(self, module):
        """获取所有卷积层"""
        conv_layers = []
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)
        return conv_layers
    
    def _find_last_conv_layer(self, module):
        """向后兼容的方法"""
        return self.conv_layers[-1] if self.conv_layers else module
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach().clone()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()
    
    def get_layer_info(self):
        """获取当前目标层的信息"""
        if self.activations is not None:
            return {
                'layer_index': self.target_layer_index,
                'total_layers': len(self.conv_layers),
                'feature_map_size': tuple(self.activations.shape[2:]),
                'channels': self.activations.shape[1]
            }
        return None
    
    def generate(self, img_tensor, speed_tensor, target_branch=0, target_output='brake', smooth=True):
        """
        生成 Grad-CAM / Grad-CAM++ 热力图
        
        参数:
            img_tensor: 输入图像张量 (1, 3, 88, 200)
            speed_tensor: 速度张量 (1, 1)
            target_branch: 目标分支索引 (0=Follow, 1=Left, 2=Right, 3=Straight)
            target_output: 目标输出 ('steer', 'throttle', 'brake')
            smooth: 是否平滑处理
            
        返回:
            numpy数组: 热力图 (88, 200)，值范围 [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()
        
        img_tensor = img_tensor.clone().detach().requires_grad_(True)
        
        try:
            pred_control, pred_speed, _, _ = self.model(img_tensor, speed_tensor)
            
            output_idx = {'steer': 0, 'throttle': 1, 'brake': 2}[target_output]
            target_idx = target_branch * 3 + output_idx
            target = pred_control[0, target_idx]
            
            target.backward(retain_graph=False)
            
            if self.gradients is None or self.activations is None:
                return None
            
            if self.use_gradcam_pp:
                # Grad-CAM++ 权重计算
                # alpha = grad^2 / (2*grad^2 + sum(A * grad^3))
                grad_2 = self.gradients ** 2
                grad_3 = grad_2 * self.gradients
                sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True)
                
                alpha_numer = grad_2
                alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
                alpha = alpha_numer / alpha_denom
                
                weights = torch.sum(alpha * F.relu(self.gradients), dim=(2, 3), keepdim=True)
            else:
                # 标准 Grad-CAM 权重
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            
            # 上采样到原图尺寸
            cam = cv2.resize(cam, (200, 88), interpolation=cv2.INTER_LINEAR)
            
            # 归一化
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
            
            if smooth:
                cam = cv2.GaussianBlur(cam, (5, 5), 0)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam
        except Exception as e:
            print(f"⚠️ Grad-CAM 生成失败: {e}")
            return None
    
    def visualize(self, original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """将热力图叠加到原图"""
        if cam is None:
            return cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
        if cam.shape != (88, 200):
            cam = cv2.resize(cam, (200, 88))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_image
        
        if heatmap.shape[:2] != original_bgr.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
        
        return cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
    
    def cleanup(self):
        """清理钩子资源"""
        if self._is_cleaned:
            return  # 避免重复清理
        
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass  # 钩子可能已被移除
        self.hooks = []
        self._is_cleaned = True


class MultiLayerGradCAM:
    """
    多层级 Grad-CAM
    
    同时在多个卷积层生成热力图，提供不同分辨率的可视化。
    支持上下文管理器，确保所有子 GradCAM 实例的钩子被正确清理。
    """
    
    def __init__(self, model, layer_indices=[-1, -3, -5], use_gradcam_pp=True):
        """
        参数:
            model: PyTorch模型
            layer_indices: 要分析的层索引列表
            use_gradcam_pp: 是否使用 Grad-CAM++（默认True）
        """
        self.model = model
        self.layer_indices = layer_indices
        self.use_gradcam_pp = use_gradcam_pp
        self._is_cleaned = False  # 标记是否已清理
        
        # 为每个层创建 GradCAM 实例
        self.grad_cams = {}
        for idx in layer_indices:
            try:
                self.grad_cams[idx] = GradCAM(model, target_layer_index=idx, use_gradcam_pp=use_gradcam_pp)
            except Exception as e:
                print(f"⚠️ 无法为层 {idx} 创建 GradCAM: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保所有钩子被清理"""
        self.cleanup()
        return False  # 不抑制异常
    
    def __del__(self):
        """析构函数，作为最后的清理保障"""
        if not self._is_cleaned:
            self.cleanup()
    
    def generate_all(self, img_tensor, speed_tensor, target_branch=0, target_output='brake'):
        """
        在所有层生成热力图
        
        返回:
            dict: {layer_index: cam_array}
        """
        results = {}
        for idx, grad_cam in self.grad_cams.items():
            cam = grad_cam.generate(img_tensor.clone(), speed_tensor, target_branch, target_output)
            if cam is not None:
                results[idx] = cam
        return results
    
    def get_fused_cam(self, cams_dict, weights=None):
        """
        融合多层热力图
        
        参数:
            cams_dict: generate_all() 的返回值
            weights: 各层权重，默认等权重
        
        返回:
            融合后的热力图
        """
        if not cams_dict:
            return None
        
        cams = list(cams_dict.values())
        if weights is None:
            weights = [1.0 / len(cams)] * len(cams)
        
        fused = np.zeros_like(cams[0])
        for cam, w in zip(cams, weights):
            fused += w * cam
        
        # 归一化
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        return fused
    
    def cleanup(self):
        """清理所有子 GradCAM 实例的钩子"""
        if self._is_cleaned:
            return  # 避免重复清理
        
        for grad_cam in self.grad_cams.values():
            try:
                grad_cam.cleanup()
            except Exception:
                pass  # 忽略清理过程中的异常
        self.grad_cams = {}
        self._is_cleaned = True



class OcclusionSensitivity:
    """
    遮挡敏感性分析
    
    通过系统性遮挡图像区域，测量模型输出变化，生成重要性图。
    这是一种定量方法，可以计算具体的敏感性数值。
    
    参考: Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks", ECCV 2014
    """
    
    def __init__(self, model, device, patch_size=2, stride=2):
        """
        参数:
            model: PyTorch模型
            device: 计算设备
            patch_size: 遮挡块大小 (默认2x2像素)
            stride: 滑动步长 (默认2，无重叠)
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
    
    def analyze(self, img_tensor, speed_tensor, target_branch=0, target_output='brake'):
        """
        执行遮挡敏感性分析
        
        返回:
            dict: {
                'sensitivity_map': 敏感性图 (H, W)，值越大表示该区域越重要,
                'max_sensitivity': 最大敏感性值,
                'mean_sensitivity': 平均敏感性值,
                'sensitivity_std': 敏感性标准差,
                'top_regions': 最敏感的区域坐标列表
            }
        """
        self.model.eval()
        H, W = 88, 200
        
        # 获取基准输出
        with torch.no_grad():
            pred_control, _, _, _ = self.model(img_tensor, speed_tensor)
            output_idx = {'steer': 0, 'throttle': 1, 'brake': 2}[target_output]
            target_idx = target_branch * 3 + output_idx
            baseline_output = pred_control[0, target_idx].item()
        
        # 计算遮挡后的输出变化
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                # 创建遮挡图像
                occluded = img_tensor.clone()
                occluded[:, :, y:y+self.patch_size, x:x+self.patch_size] = 0
                
                with torch.no_grad():
                    pred_occluded, _, _, _ = self.model(occluded, speed_tensor)
                    occluded_output = pred_occluded[0, target_idx].item()
                
                # 计算输出变化（绝对值）
                change = abs(baseline_output - occluded_output)
                
                # 累加到敏感性图
                sensitivity_map[y:y+self.patch_size, x:x+self.patch_size] += change
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # 平均化
        count_map[count_map == 0] = 1
        sensitivity_map = sensitivity_map / count_map
        
        # 归一化到 [0, 1]
        s_min, s_max = sensitivity_map.min(), sensitivity_map.max()
        if s_max - s_min > 1e-8:
            sensitivity_map_norm = (sensitivity_map - s_min) / (s_max - s_min)
        else:
            sensitivity_map_norm = np.zeros_like(sensitivity_map)
        
        # 找到最敏感的区域
        flat_idx = np.argsort(sensitivity_map.flatten())[::-1][:5]
        top_regions = [(idx // W, idx % W) for idx in flat_idx]
        
        return {
            'sensitivity_map': sensitivity_map_norm,
            'sensitivity_map_raw': sensitivity_map,
            'max_sensitivity': float(s_max),
            'mean_sensitivity': float(np.mean(sensitivity_map)),
            'sensitivity_std': float(np.std(sensitivity_map)),
            'top_regions': top_regions,
            'baseline_output': baseline_output
        }


class IntegratedGradients:
    """
    积分梯度 (Integrated Gradients)
    
    一种满足公理化要求的归因方法，比Grad-CAM更严谨。
    
    参考: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
    """
    
    def __init__(self, model, device, steps=50):
        self.model = model
        self.device = device
        self.steps = steps
    
    def compute(self, img_tensor, speed_tensor, target_branch=0, target_output='brake', 
                baseline=None):
        """计算积分梯度"""
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(img_tensor)
        
        # 计算基线和输入的输出
        with torch.no_grad():
            pred_baseline, _, _, _ = self.model(baseline, speed_tensor)
            pred_input, _, _, _ = self.model(img_tensor, speed_tensor)
            
            output_idx = {'steer': 0, 'throttle': 1, 'brake': 2}[target_output]
            target_idx = target_branch * 3 + output_idx
            
            baseline_output = pred_baseline[0, target_idx].item()
            input_output = pred_input[0, target_idx].item()
            output_diff = input_output - baseline_output
        
        # 计算积分梯度
        scaled_inputs = [baseline + (float(i) / self.steps) * (img_tensor - baseline) 
                         for i in range(self.steps + 1)]
        
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            pred, _, _, _ = self.model(scaled_input, speed_tensor)
            target = pred[0, target_idx]
            
            self.model.zero_grad()
            target.backward(retain_graph=False)
            
            if scaled_input.grad is not None:
                gradients.append(scaled_input.grad.clone())
        
        if not gradients:
            return None
        
        # 梯形积分
        avg_gradients = torch.zeros_like(gradients[0])
        for i in range(len(gradients) - 1):
            avg_gradients += (gradients[i] + gradients[i + 1]) / 2
        avg_gradients /= self.steps
        
        # 计算归因
        integrated_gradients = (img_tensor - baseline) * avg_gradients
        
        # 转换为2D归因图
        attribution_map = integrated_gradients.squeeze().cpu().numpy()
        attribution_map = np.sum(np.abs(attribution_map), axis=0)
        
        # 归一化
        a_min, a_max = attribution_map.min(), attribution_map.max()
        if a_max - a_min > 1e-8:
            attribution_map_norm = (attribution_map - a_min) / (a_max - a_min)
        else:
            attribution_map_norm = np.zeros_like(attribution_map)
        
        # 计算完整性误差
        attribution_sum = float(integrated_gradients.sum().item())
        completeness_error = abs(attribution_sum - output_diff)
        
        return {
            'attribution_map': attribution_map_norm,
            'attribution_map_raw': attribution_map,
            'attribution_sum': attribution_sum,
            'output_diff': output_diff,
            'completeness_error': completeness_error,
            'baseline_output': baseline_output,
            'input_output': input_output
        }


class DeletionInsertion:
    """
    删除/插入曲线 (Deletion/Insertion Curves)
    
    评估解释质量的标准指标。
    参考: Petsiuk et al., "RISE: Randomized Input Sampling for Explanation", BMVC 2018
    """
    
    def __init__(self, model, device, steps=20):
        self.model = model
        self.device = device
        self.steps = steps
    
    def compute(self, img_tensor, speed_tensor, saliency_map, 
                target_branch=0, target_output='brake'):
        """计算删除和插入曲线"""
        self.model.eval()
        H, W = 88, 200
        
        if saliency_map.shape != (H, W):
            saliency_map = cv2.resize(saliency_map, (W, H))
        
        flat_saliency = saliency_map.flatten()
        sorted_indices = np.argsort(flat_saliency)[::-1]
        
        output_idx = {'steer': 0, 'throttle': 1, 'brake': 2}[target_output]
        target_idx = target_branch * 3 + output_idx
        
        with torch.no_grad():
            pred_orig, _, _, _ = self.model(img_tensor, speed_tensor)
            original_output = pred_orig[0, target_idx].item()
        
        baseline = torch.zeros_like(img_tensor)
        with torch.no_grad():
            pred_base, _, _, _ = self.model(baseline, speed_tensor)
            baseline_output = pred_base[0, target_idx].item()
        
        # 删除曲线
        deletion_curve = [original_output]
        deleted_img = img_tensor.clone()
        pixels_per_step = len(sorted_indices) // self.steps
        
        for step in range(self.steps):
            start_idx = step * pixels_per_step
            end_idx = (step + 1) * pixels_per_step
            indices_to_delete = sorted_indices[start_idx:end_idx]
            
            for idx in indices_to_delete:
                y, x = idx // W, idx % W
                deleted_img[:, :, y, x] = 0
            
            with torch.no_grad():
                pred, _, _, _ = self.model(deleted_img, speed_tensor)
                deletion_curve.append(pred[0, target_idx].item())
        
        # 插入曲线
        insertion_curve = [baseline_output]
        inserted_img = baseline.clone()
        
        for step in range(self.steps):
            start_idx = step * pixels_per_step
            end_idx = (step + 1) * pixels_per_step
            indices_to_insert = sorted_indices[start_idx:end_idx]
            
            for idx in indices_to_insert:
                y, x = idx // W, idx % W
                inserted_img[:, :, y, x] = img_tensor[:, :, y, x]
            
            with torch.no_grad():
                pred, _, _, _ = self.model(inserted_img, speed_tensor)
                insertion_curve.append(pred[0, target_idx].item())
        
        deletion_curve = np.array(deletion_curve)
        insertion_curve = np.array(insertion_curve)
        
        x = np.linspace(0, 1, len(deletion_curve))
        deletion_auc = float(np.trapz(deletion_curve, x))
        insertion_auc = float(np.trapz(insertion_curve, x))
        
        if abs(original_output) > 1e-8:
            deletion_auc_norm = deletion_auc / abs(original_output)
            insertion_auc_norm = insertion_auc / abs(original_output)
        else:
            deletion_auc_norm = deletion_auc
            insertion_auc_norm = insertion_auc
        
        return {
            'deletion_curve': deletion_curve,
            'insertion_curve': insertion_curve,
            'deletion_auc': deletion_auc,
            'insertion_auc': insertion_auc,
            'deletion_auc_normalized': deletion_auc_norm,
            'insertion_auc_normalized': insertion_auc_norm,
            'combined_score': insertion_auc - deletion_auc,
            'original_output': original_output,
            'baseline_output': baseline_output
        }



# ============================================================================
# 第二部分：统计分析器
# ============================================================================

class BrakeAnalyzer:
    """控制输出分析器（支持全部历史记录）"""
    
    def __init__(self, history_size=None):
        """
        参数:
            history_size: 历史记录最大帧数，None表示记录所有帧（无限制）
        """
        # 如果history_size为None，使用普通列表存储所有帧
        if history_size is None:
            self.brake_history = []
            self.throttle_history = []
            self.steer_history = []  # 改为存储steer而不是speed
            self._use_deque = False
        else:
            self.brake_history = deque(maxlen=history_size)
            self.throttle_history = deque(maxlen=history_size)
            self.steer_history = deque(maxlen=history_size)  # 改为存储steer
            self._use_deque = True
        
        self.total_frames = 0
        self.brake_frames = 0
        self.hard_brake_frames = 0
    
    def update(self, brake, throttle, steer):
        """更新控制历史记录"""
        self.brake_history.append(brake)
        self.throttle_history.append(throttle)
        # steer范围是-1到1，转换为0到1用于显示
        self.steer_history.append((steer + 1.0) / 2.0)
        
        self.total_frames += 1
        if brake > 0.1:
            self.brake_frames += 1
        if brake > 0.5:
            self.hard_brake_frames += 1
    
    def get_statistics(self):
        if self.total_frames == 0:
            return {
                'total_frames': 0, 'brake_ratio': 0, 'hard_brake_ratio': 0,
                'avg_brake': 0, 'max_brake': 0, 'brake_std': 0,
            }
        
        brakes = np.array(self.brake_history) if self.brake_history else np.array([0])
        
        return {
            'total_frames': self.total_frames,
            'brake_ratio': self.brake_frames / self.total_frames,
            'hard_brake_ratio': self.hard_brake_frames / self.total_frames,
            'avg_brake': float(np.mean(brakes)),
            'max_brake': float(np.max(brakes)),
            'brake_std': float(np.std(brakes)),
        }
    
    def plot_history(self, width=600, height=200):
        """绘制历史曲线图（显示所有帧数据）
        
        注意：Steer 原始范围是 -1~+1，已转换为 0~1 用于统一显示
              0.5 = 直行，<0.5 = 左转，>0.5 = 右转
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (28, 28, 30)
        
        if len(self.brake_history) < 2:
            cv2.putText(img, "Waiting for data...", (width//2 - 100, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
            return img
        
        margin_left, margin_right = 60, 20
        margin_top, margin_bottom = 35, 25
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        
        # 网格线和Y轴刻度（左侧：Brake/Throttle 0~1，右侧：Steer -1~+1）
        for i in range(0, 6):
            y = margin_top + int(plot_height * i / 5)
            cv2.line(img, (margin_left, y), (width - margin_right, y), (50, 50, 55), 1)
            # 左侧刻度：Brake/Throttle (0~1)
            val_left = 1.0 - i * 0.2
            cv2.putText(img, f"{val_left:.1f}", (5, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
            # 右侧刻度：Steer (-1~+1)
            val_right = 1.0 - i * 0.4  # 1.0, 0.6, 0.2, -0.2, -0.6, -1.0
            cv2.putText(img, f"{val_right:+.1f}", (width - margin_right + 5, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 220, 220), 1, cv2.LINE_AA)
        
        # 绘制 Steer=0 的中线（对应 y=0.5 的位置）
        center_y = margin_top + int(plot_height * 0.5)
        cv2.line(img, (margin_left, center_y), (width - margin_right, center_y), 
                 (80, 160, 160), 1, cv2.LINE_AA)
        
        def draw_curve(data, color, thickness=2):
            if len(data) < 2:
                return
            points = []
            data_len = len(data)
            for i, val in enumerate(data):
                x = margin_left + int(i * plot_width / data_len)
                y = margin_top + int((1.0 - val) * plot_height)
                y = max(margin_top, min(height - margin_bottom, y))
                points.append((x, y))
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], color, thickness, cv2.LINE_AA)
        
        draw_curve(self.brake_history, (100, 100, 255), 2)  # 红色 - Brake
        draw_curve(self.throttle_history, (100, 230, 100), 2)  # 绿色 - Throttle
        draw_curve(self.steer_history, (100, 220, 220), 2)  # 黄色 - Steer
        
        # 图例（说明 Steer 的中线含义）
        legend_y = 18
        cv2.circle(img, (margin_left + 15, legend_y), 6, (100, 100, 255), -1)
        cv2.putText(img, "Brake", (margin_left + 28, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(img, (margin_left + 100, legend_y), 6, (100, 230, 100), -1)
        cv2.putText(img, "Throttle", (margin_left + 113, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(img, (margin_left + 200, legend_y), 6, (100, 220, 220), -1)
        cv2.putText(img, "Steer", (margin_left + 213, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        # 添加 Steer 中线说明
        cv2.putText(img, "(0=straight)", (margin_left + 265, legend_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 220, 220), 1, cv2.LINE_AA)
        
        # 显示总帧数
        total_text = f"Total: {len(self.brake_history)} frames"
        cv2.putText(img, total_text, (width - 150, height - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 130), 1, cv2.LINE_AA)
        
        return img


class QuantitativeMetricsCollector:
    """定量指标收集器"""
    
    def __init__(self):
        self.metrics_history = []
        self.summary_stats = {}
    
    def add_frame_metrics(self, frame_id, metrics_dict):
        metrics_dict['frame_id'] = frame_id
        metrics_dict['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics_dict)
    
    def compute_summary(self):
        if not self.metrics_history:
            return {}
        
        occlusion_sensitivities = [m.get('occlusion_mean_sensitivity', 0) 
                                   for m in self.metrics_history if 'occlusion_mean_sensitivity' in m]
        ig_completeness_errors = [m.get('ig_completeness_error', 0) 
                                  for m in self.metrics_history if 'ig_completeness_error' in m]
        deletion_aucs = [m.get('deletion_auc', 0) 
                         for m in self.metrics_history if 'deletion_auc' in m]
        insertion_aucs = [m.get('insertion_auc', 0) 
                          for m in self.metrics_history if 'insertion_auc' in m]
        
        self.summary_stats = {
            'total_frames_analyzed': len(self.metrics_history),
            'occlusion_sensitivity': {
                'mean': float(np.mean(occlusion_sensitivities)) if occlusion_sensitivities else 0,
                'std': float(np.std(occlusion_sensitivities)) if occlusion_sensitivities else 0,
                'max': float(np.max(occlusion_sensitivities)) if occlusion_sensitivities else 0,
            },
            'integrated_gradients': {
                'mean_completeness_error': float(np.mean(ig_completeness_errors)) if ig_completeness_errors else 0,
                'std_completeness_error': float(np.std(ig_completeness_errors)) if ig_completeness_errors else 0,
            },
            'deletion_insertion': {
                'mean_deletion_auc': float(np.mean(deletion_aucs)) if deletion_aucs else 0,
                'mean_insertion_auc': float(np.mean(insertion_aucs)) if insertion_aucs else 0,
                'mean_combined_score': float(np.mean(insertion_aucs) - np.mean(deletion_aucs)) if deletion_aucs and insertion_aucs else 0,
            }
        }
        
        return self.summary_stats
    
    def export_to_json(self, filepath):
        self.compute_summary()
        export_data = {
            'summary': self.summary_stats,
            'frame_metrics': self.metrics_history,
            'export_time': datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 指标已导出到: {filepath}")
    
    def export_to_latex_table(self):
        self.compute_summary()
        s = self.summary_stats
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Model Interpretability Quantitative Metrics}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Mean} & \textbf{Std} \\
\hline
Occlusion Sensitivity & %.4f & %.4f \\
IG Completeness Error & %.4f & %.4f \\
Deletion AUC $\downarrow$ & %.4f & - \\
Insertion AUC $\uparrow$ & %.4f & - \\
Combined Score $\uparrow$ & %.4f & - \\
\hline
\end{tabular}
\end{table}
""" % (
            s.get('occlusion_sensitivity', {}).get('mean', 0),
            s.get('occlusion_sensitivity', {}).get('std', 0),
            s.get('integrated_gradients', {}).get('mean_completeness_error', 0),
            s.get('integrated_gradients', {}).get('std_completeness_error', 0),
            s.get('deletion_insertion', {}).get('mean_deletion_auc', 0),
            s.get('deletion_insertion', {}).get('mean_insertion_auc', 0),
            s.get('deletion_insertion', {}).get('mean_combined_score', 0),
        )
        return latex



# ============================================================================
# 第三部分：综合分析器
# ============================================================================

class ComprehensiveAnalyzer:
    """
    综合可解释性分析器
    
    整合所有分析方法，提供统一接口。
    """
    
    def __init__(self, model, device, 
                 enable_occlusion=True,
                 enable_integrated_gradients=True,
                 enable_deletion_insertion=True,
                 occlusion_patch_size=2,
                 occlusion_stride=2,
                 ig_steps=30,
                 di_steps=15,
                 grad_cam_layer_index=-3,
                 use_gradcam_pp=True,
                 enable_multi_layer=True,
                 multi_layer_indices=None,
                 history_max_frames=None):
        """
        参数:
            grad_cam_layer_index: Grad-CAM 目标层索引（-3推荐，更高分辨率）
            use_gradcam_pp: 是否使用 Grad-CAM++
            enable_multi_layer: 是否启用多层级分析
            multi_layer_indices: 多层级分析使用的层索引列表
            history_max_frames: 历史记录最大帧数，None表示记录所有帧
        """
        self.model = model
        self.device = device
        
        # 初始化 Grad-CAM（使用改进的参数）
        self.grad_cam = GradCAM(model, target_layer_index=grad_cam_layer_index, 
                                use_gradcam_pp=use_gradcam_pp)
        
        # 多层级 Grad-CAM
        self.enable_multi_layer = enable_multi_layer
        if multi_layer_indices is None:
            multi_layer_indices = [-1, -3, -5]  # 默认值
        if enable_multi_layer:
            self.multi_layer_cam = MultiLayerGradCAM(model, layer_indices=multi_layer_indices, 
                                                     use_gradcam_pp=use_gradcam_pp)
        else:
            self.multi_layer_cam = None
        
        # 使用配置的历史记录帧数（None表示记录所有帧）
        self.brake_analyzer = BrakeAnalyzer(history_size=history_max_frames)
        self.metrics_collector = QuantitativeMetricsCollector()
        
        self.enable_occlusion = enable_occlusion
        self.enable_ig = enable_integrated_gradients
        self.enable_di = enable_deletion_insertion
        
        if enable_occlusion:
            self.occlusion = OcclusionSensitivity(model, device, occlusion_patch_size, occlusion_stride)
        else:
            self.occlusion = None
            
        if enable_integrated_gradients:
            self.ig = IntegratedGradients(model, device, ig_steps)
        else:
            self.ig = None
            
        if enable_deletion_insertion:
            self.di = DeletionInsertion(model, device, di_steps)
        else:
            self.di = None
        
        self.frame_count = 0
        self.analysis_interval = 1  # 实时分析（每帧都执行完整分析）
        self._is_cleaned = False  # 清理标记
    
    def analyze_frame(self, img_tensor, speed_tensor, original_image,
                      control_result, current_command, full_analysis=False):
        """分析单帧"""
        self.frame_count += 1
        branch_idx = current_command - 2
        
        results = {
            'frame_id': self.frame_count,
            'current_branch': branch_idx,
            'branch_name': ['Follow', 'Left', 'Right', 'Straight'][branch_idx],
            'control': control_result.copy()
        }
        
        # 1. Grad-CAM（每帧都执行）
        results['brake_cam'] = self.grad_cam.generate(
            img_tensor.clone(), speed_tensor,
            target_branch=branch_idx, target_output='brake'
        )
        results['throttle_cam'] = self.grad_cam.generate(
            img_tensor.clone(), speed_tensor,
            target_branch=branch_idx, target_output='throttle'
        )
        results['steer_cam'] = self.grad_cam.generate(
            img_tensor.clone(), speed_tensor,
            target_branch=branch_idx, target_output='steer'
        )
        
        # 获取层信息
        results['grad_cam_layer_info'] = self.grad_cam.get_layer_info()
        
        # 2. 多层级 Grad-CAM（如果启用）- 为Brake、Throttle、Steer分别生成
        if self.multi_layer_cam is not None:
            # Brake多层热力图
            results['multi_layer_cams'] = self.multi_layer_cam.generate_all(
                img_tensor.clone(), speed_tensor,
                target_branch=branch_idx, target_output='brake'
            )
            results['fused_cam'] = self.multi_layer_cam.get_fused_cam(results['multi_layer_cams'])
            
            # Throttle多层热力图
            results['throttle_multi_layer_cams'] = self.multi_layer_cam.generate_all(
                img_tensor.clone(), speed_tensor,
                target_branch=branch_idx, target_output='throttle'
            )
            results['throttle_fused_cam'] = self.multi_layer_cam.get_fused_cam(results['throttle_multi_layer_cams'])
            
            # Steer多层热力图
            results['steer_multi_layer_cams'] = self.multi_layer_cam.generate_all(
                img_tensor.clone(), speed_tensor,
                target_branch=branch_idx, target_output='steer'
            )
            results['steer_fused_cam'] = self.multi_layer_cam.get_fused_cam(results['steer_multi_layer_cams'])
        
        # 3. 更新控制分析器（Brake, Throttle, Steer）
        self.brake_analyzer.update(
            control_result['brake'],
            control_result['throttle'],
            control_result['steer']
        )
        results['brake_stats'] = self.brake_analyzer.get_statistics()
        
        # 4. 定量分析（按间隔执行）
        do_full = full_analysis or (self.frame_count % self.analysis_interval == 0)
        
        if do_full:
            metrics = {}
            
            if self.occlusion is not None:
                occ_result = self.occlusion.analyze(
                    img_tensor, speed_tensor,
                    target_branch=branch_idx, target_output='brake'
                )
                results['occlusion'] = occ_result
                metrics['occlusion_mean_sensitivity'] = occ_result['mean_sensitivity']
                metrics['occlusion_max_sensitivity'] = occ_result['max_sensitivity']
            
            if self.ig is not None:
                ig_result = self.ig.compute(
                    img_tensor, speed_tensor,
                    target_branch=branch_idx, target_output='brake'
                )
                if ig_result:
                    results['integrated_gradients'] = ig_result
                    metrics['ig_completeness_error'] = ig_result['completeness_error']
                    metrics['ig_output_diff'] = ig_result['output_diff']
            
            if self.di is not None and results['brake_cam'] is not None:
                di_result = self.di.compute(
                    img_tensor, speed_tensor, results['brake_cam'],
                    target_branch=branch_idx, target_output='brake'
                )
                results['deletion_insertion'] = di_result
                metrics['deletion_auc'] = di_result['deletion_auc']
                metrics['insertion_auc'] = di_result['insertion_auc']
                metrics['combined_score'] = di_result['combined_score']
            
            if metrics:
                self.metrics_collector.add_frame_metrics(self.frame_count, metrics)
            
            results['quantitative_metrics'] = metrics
            results['is_full_analysis'] = True
        else:
            results['is_full_analysis'] = False
        
        return results
    
    def get_metrics_summary(self):
        return self.metrics_collector.compute_summary()
    
    def export_metrics(self, filepath):
        self.metrics_collector.export_to_json(filepath)
    
    def cleanup(self):
        """清理所有分析器资源"""
        if hasattr(self, '_is_cleaned') and self._is_cleaned:
            return  # 避免重复清理
        
        if hasattr(self, 'grad_cam') and self.grad_cam is not None:
            self.grad_cam.cleanup()
        
        if hasattr(self, 'multi_layer_cam') and self.multi_layer_cam is not None:
            self.multi_layer_cam.cleanup()
        
        self._is_cleaned = True


# ============================================================================
# 第四部分：仪表板渲染器（1920x1080 版本）
# ============================================================================

class InterpretabilityDashboard:
    """
    可解释性仪表板渲染器（2560x1440 2K版本）
    
    改进：
    - 分辨率 2560x1440 (2K)
    - 更大的热力图显示区域
    - 多层级热力图对比
    - 更详细的定量指标展示
    - 删除/插入曲线可视化
    - 优化布局避免重叠
    """
    
    # 仪表板尺寸 2560x1440 (2K分辨率)
    WIDTH = 2560
    HEIGHT = 1440
    
    # 布局常量
    HEADER_HEIGHT = 55
    FOOTER_HEIGHT = 45
    ROW_GAP = 8
    PANEL_GAP = 10
    MARGIN = 12
    
    # 颜色主题（深色主题，更现代）
    BG_COLOR = (18, 18, 22)
    PANEL_BG = (28, 28, 34)
    PANEL_BORDER = (50, 50, 60)
    TEXT_PRIMARY = (240, 240, 240)
    TEXT_SECONDARY = (150, 150, 160)
    ACCENT_BLUE = (255, 180, 100)
    ACCENT_GREEN = (100, 220, 100)
    ACCENT_RED = (100, 100, 255)
    ACCENT_YELLOW = (100, 220, 220)
    ACCENT_PURPLE = (220, 130, 220)
    
    # 分支颜色 (BGR格式)
    BRANCH_COLORS = {
        0: (80, 200, 200),   # Follow - 黄
        1: (200, 200, 80),   # Left - 青
        2: (200, 80, 200),   # Right - 紫
        3: (80, 200, 80),    # Straight - 绿
    }
    
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        self.frame_count = 0
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    
    def render(self, original_image, analysis_results, control_result,
               current_command, traffic_light_info=None, all_branch_predictions=None):
        """渲染完整仪表板（2560x1440）- 优化布局版本"""
        self.frame_count += 1
        branch_idx = current_command - 2
        
        # 创建画布 (2560x1440)
        dashboard = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        dashboard[:] = self.BG_COLOR
        
        # ============ 布局计算 ============
        # 可用高度 = 总高度 - Header - Footer
        content_start_y = self.HEADER_HEIGHT + 5
        content_end_y = self.HEIGHT - self.FOOTER_HEIGHT - 5
        content_height = content_end_y - content_start_y
        
        # 6行内容的高度分配 (总可用高度约1375px)
        # Row1: 主图像行 (较高)
        # Row2-4: 热力图行 (相同高度)
        # Row5: 控制面板行
        # Row6: 历史/曲线行
        row1_h = 150  # 主图像行
        row234_h = 130  # 热力图行 (每行)
        row5_h = 190  # 控制面板行
        # row6高度动态计算，确保不超出footer
        
        # 计算各行Y坐标
        self.row1_y = content_start_y
        self.row2_y = self.row1_y + row1_h + self.ROW_GAP
        self.row3_y = self.row2_y + row234_h + self.ROW_GAP
        self.row4_y = self.row3_y + row234_h + self.ROW_GAP
        self.row5_y = self.row4_y + row234_h + self.ROW_GAP
        self.row6_y = self.row5_y + row5_h + self.ROW_GAP
        
        # 动态计算row6高度，确保不超出footer
        row6_h = content_end_y - self.row6_y
        row6_h = max(row6_h, 200)  # 最小高度200px
        
        # 存储行高度供各绘制函数使用
        self.row1_h = row1_h
        self.row234_h = row234_h
        self.row5_h = row5_h
        self.row6_h = row6_h
        
        # ============ 绘制各区域 ============
        self._draw_header(dashboard, current_command, analysis_results.get('frame_id', 0))
        self._draw_main_image_row(dashboard, original_image, analysis_results, branch_idx)
        self._draw_brake_all_layers_row(dashboard, original_image, analysis_results, branch_idx)
        self._draw_throttle_all_layers_row(dashboard, original_image, analysis_results, branch_idx)
        self._draw_steer_all_layers_row(dashboard, original_image, analysis_results, branch_idx)
        self._draw_control_panel(dashboard, control_result, branch_idx)
        self._draw_quantitative_panel(dashboard, analysis_results)
        self._draw_branch_panel(dashboard, all_branch_predictions, branch_idx)
        self._draw_traffic_light(dashboard, traffic_light_info, control_result)
        self._draw_history_panel(dashboard, analysis_results)
        self._draw_curves_panel(dashboard, analysis_results)
        self._draw_metrics_summary_panel(dashboard, analysis_results)
        self._draw_footer(dashboard, analysis_results)
        
        return dashboard
    
    def _draw_panel(self, img, x, y, w, h, title="", title_color=None, title_size=0.5):
        """绘制面板（带圆角效果）"""
        if title_color is None:
            title_color = self.TEXT_PRIMARY
        
        # 面板背景
        cv2.rectangle(img, (x, y), (x+w, y+h), self.PANEL_BG, -1)
        # 边框
        cv2.rectangle(img, (x, y), (x+w, y+h), self.PANEL_BORDER, 1)
        
        if title:
            # 标题背景条
            cv2.rectangle(img, (x+1, y+1), (x+w-1, y+22), (38, 38, 45), -1)
            cv2.line(img, (x, y+22), (x+w, y+22), self.PANEL_BORDER, 1)
            cv2.putText(img, title, (x+8, y+16), 
                        cv2.FONT_HERSHEY_SIMPLEX, title_size, title_color, 1, cv2.LINE_AA)

    def _draw_header(self, img, current_command, frame_id):
        """绘制标题栏"""
        cv2.rectangle(img, (0, 0), (self.WIDTH, self.HEADER_HEIGHT), (28, 28, 34), -1)
        cv2.line(img, (0, self.HEADER_HEIGHT), (self.WIDTH, self.HEADER_HEIGHT), self.PANEL_BORDER, 2)
        
        # 主标题
        cv2.putText(img, "Model Interpretability Dashboard v3.0", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.TEXT_PRIMARY, 2, cv2.LINE_AA)
        
        # 副标题
        cv2.putText(img, "All Conv Layers Grad-CAM++ | Academic Analysis", (480, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        
        # 当前分支标签
        branch_idx = current_command - 2
        branch_names = ['Follow', 'Left', 'Right', 'Straight']
        branch_color = self.BRANCH_COLORS.get(branch_idx, (150, 150, 150))
        
        # 分支标签框
        label_x = 2100
        cv2.rectangle(img, (label_x, 10), (label_x + 220, 45), branch_color, -1)
        cv2.rectangle(img, (label_x, 10), (label_x + 220, 45), self.TEXT_PRIMARY, 2)
        cv2.putText(img, f"Branch: {branch_names[branch_idx]}", (label_x + 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 25), 2, cv2.LINE_AA)
        
        # 帧号
        cv2.putText(img, f"Frame: {frame_id}", (2380, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
    
    def _draw_main_image_row(self, img, original_image, results, branch_idx):
        """绘制第一行（原图 + 3个Grad-CAM热力图 + Occlusion + IG + Layer Info）"""
        y_start = self.row1_y
        panel_h = self.row1_h
        
        # 7个面板: 原图 + Brake CAM + Throttle CAM + Steer CAM + Occlusion + IG + Layer Info
        # 计算图像尺寸 (保持88:200比例)
        img_h = panel_h - 30
        img_w = int(img_h * 200 / 88)
        panel_w = img_w + 12
        gap = 8
        
        orig_resized = cv2.resize(original_image, (img_w, img_h))
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
        
        # 面板1: 原始输入
        x1 = self.MARGIN
        self._draw_panel(img, x1, y_start, panel_w, panel_h, "Input (88x200)", title_size=0.45)
        img[y_start+25:y_start+25+img_h, x1+6:x1+6+img_w] = orig_bgr
        
        # 面板2: Brake Grad-CAM++
        x2 = x1 + panel_w + gap
        self._draw_panel(img, x2, y_start, panel_w, panel_h, 
                         f"Brake CAM++ (B{branch_idx})", self.ACCENT_RED, 0.45)
        brake_cam = results.get('brake_cam')
        if brake_cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * brake_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
            img[y_start+25:y_start+25+img_h, x2+6:x2+6+img_w] = overlay
        
        # 面板3: Throttle Grad-CAM++
        x3 = x2 + panel_w + gap
        self._draw_panel(img, x3, y_start, panel_w, panel_h,
                         f"Throttle CAM++ (B{branch_idx})", self.ACCENT_GREEN, 0.45)
        throttle_cam = results.get('throttle_cam')
        if throttle_cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * throttle_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
            img[y_start+25:y_start+25+img_h, x3+6:x3+6+img_w] = overlay
        
        # 面板4: Steer Grad-CAM++
        x4 = x3 + panel_w + gap
        self._draw_panel(img, x4, y_start, panel_w, panel_h,
                         f"Steer CAM++ (B{branch_idx})", self.ACCENT_YELLOW, 0.45)
        steer_cam = results.get('steer_cam')
        if steer_cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * steer_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
            img[y_start+25:y_start+25+img_h, x4+6:x4+6+img_w] = overlay
        
        # 面板5: Occlusion Sensitivity
        x5 = x4 + panel_w + gap
        self._draw_panel(img, x5, y_start, panel_w, panel_h,
                         "Occlusion Sensitivity", self.ACCENT_BLUE, 0.45)
        occlusion = results.get('occlusion')
        if occlusion is not None:
            occ_map = occlusion.get('sensitivity_map')
            if occ_map is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * occ_map), cv2.COLORMAP_HOT)
                heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
                overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
                img[y_start+25:y_start+25+img_h, x5+6:x5+6+img_w] = overlay
        else:
            cv2.putText(img, "Computing...", (x5 + 60, y_start + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 110), 1, cv2.LINE_AA)
        
        # 面板6: Integrated Gradients
        x6 = x5 + panel_w + gap
        self._draw_panel(img, x6, y_start, panel_w, panel_h,
                         "Integrated Gradients", self.ACCENT_PURPLE, 0.45)
        ig_result = results.get('integrated_gradients')
        if ig_result is not None:
            ig_map = ig_result.get('attribution_map')
            if ig_map is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * ig_map), cv2.COLORMAP_VIRIDIS)
                heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
                overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
                img[y_start+25:y_start+25+img_h, x6+6:x6+6+img_w] = overlay
        else:
            cv2.putText(img, "Computing...", (x6 + 60, y_start + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 110), 1, cv2.LINE_AA)
        
        # 面板7: Layer Resolution Info
        x7 = x6 + panel_w + gap
        info_w = self.WIDTH - x7 - self.MARGIN
        self._draw_panel(img, x7, y_start, info_w, panel_h, "Layer Resolution Info", title_size=0.45)
        
        layer_info = results.get('grad_cam_layer_info')
        if layer_info:
            text_y = y_start + 42
            cv2.putText(img, f"Target Layer: {layer_info['layer_index']}/{layer_info['total_layers']}", 
                        (x7 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, f"Feature Map: {layer_info['feature_map_size']}", 
                        (x7 + 10, text_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, f"Channels: {layer_info['channels']}", 
                        (x7 + 10, text_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, "Grad-CAM++ (improved)", 
                        (x7 + 10, text_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ACCENT_GREEN, 1, cv2.LINE_AA)
    
    def _draw_brake_all_layers_row(self, img, original_image, results, branch_idx):
        """绘制第二行（Brake 所有卷积层热力图）"""
        y_start = self.row2_y
        panel_h = self.row234_h
        
        # 10个面板: 8层 + Fused + 空白 (适应2560宽度)
        # 计算每个面板宽度
        num_panels = 10
        gap = 6
        total_gap = gap * (num_panels - 1)
        panel_w = (self.WIDTH - 2 * self.MARGIN - total_gap) // num_panels
        img_h = panel_h - 28
        img_w = int(img_h * 200 / 88)
        if img_w > panel_w - 8:
            img_w = panel_w - 8
            img_h = int(img_w * 88 / 200)
        
        orig_resized = cv2.resize(original_image, (img_w, img_h))
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
        
        # 获取所有层的热力图
        multi_cams = results.get('multi_layer_cams', {})
        
        # 定义所有8个卷积层（从浅到深）
        all_layers = [
            (-8, 'Brake L1'), (-7, 'Brake L2'), (-6, 'Brake L3'), (-5, 'Brake L4'),
            (-4, 'Brake L5'), (-3, 'Brake L6'), (-2, 'Brake L7'), (-1, 'Brake L8')
        ]
        
        x_start = self.MARGIN
        img_offset_x = (panel_w - img_w) // 2
        img_offset_y = 25
        
        for i, (layer_idx, layer_name) in enumerate(all_layers):
            x = x_start + i * (panel_w + gap)
            self._draw_panel(img, x, y_start, panel_w, panel_h, layer_name, self.ACCENT_RED, 0.4)
            
            cam = multi_cams.get(layer_idx)
            if cam is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
                overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
                img[y_start+img_offset_y:y_start+img_offset_y+img_h, x+img_offset_x:x+img_offset_x+img_w] = overlay
            else:
                cv2.putText(img, "N/A", (x + panel_w//2 - 12, y_start + panel_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 110), 1, cv2.LINE_AA)
        
        # 融合热力图
        x_fused = x_start + len(all_layers) * (panel_w + gap)
        self._draw_panel(img, x_fused, y_start, panel_w, panel_h, "Brake Fused", self.ACCENT_PURPLE, 0.4)
        fused_cam = results.get('fused_cam')
        if fused_cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * fused_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
            img[y_start+img_offset_y:y_start+img_offset_y+img_h, x_fused+img_offset_x:x_fused+img_offset_x+img_w] = overlay
    
    def _draw_multi_layer_row(self, img, original_image, results, branch_idx):
        """保留旧函数名以兼容，实际调用新函数"""
        self._draw_brake_all_layers_row(img, original_image, results, branch_idx)
        
    def _draw_throttle_all_layers_row(self, img, original_image, results, branch_idx):
        """绘制第三行（Throttle 所有卷积层热力图）"""
        y_start = self.row3_y
        panel_h = self.row234_h
        
        # 与Brake行相同的布局
        num_panels = 10
        gap = 6
        total_gap = gap * (num_panels - 1)
        panel_w = (self.WIDTH - 2 * self.MARGIN - total_gap) // num_panels
        img_h = panel_h - 28
        img_w = int(img_h * 200 / 88)
        if img_w > panel_w - 8:
            img_w = panel_w - 8
            img_h = int(img_w * 88 / 200)
        
        orig_resized = cv2.resize(original_image, (img_w, img_h))
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
        
        multi_cams = results.get('throttle_multi_layer_cams', {})
        
        all_layers = [
            (-8, 'Throt L1'), (-7, 'Throt L2'), (-6, 'Throt L3'), (-5, 'Throt L4'),
            (-4, 'Throt L5'), (-3, 'Throt L6'), (-2, 'Throt L7'), (-1, 'Throt L8')
        ]
        
        x_start = self.MARGIN
        img_offset_x = (panel_w - img_w) // 2
        img_offset_y = 25
        
        for i, (layer_idx, layer_name) in enumerate(all_layers):
            x = x_start + i * (panel_w + gap)
            self._draw_panel(img, x, y_start, panel_w, panel_h, layer_name, self.ACCENT_GREEN, 0.4)
            
            cam = multi_cams.get(layer_idx)
            if cam is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
                overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
                img[y_start+img_offset_y:y_start+img_offset_y+img_h, x+img_offset_x:x+img_offset_x+img_w] = overlay
            else:
                cv2.putText(img, "N/A", (x + panel_w//2 - 12, y_start + panel_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 110), 1, cv2.LINE_AA)
        
        # 融合热力图
        x_fused = x_start + len(all_layers) * (panel_w + gap)
        self._draw_panel(img, x_fused, y_start, panel_w, panel_h, "Throt Fused", self.ACCENT_PURPLE, 0.4)
        fused_cam = results.get('throttle_fused_cam')
        if fused_cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * fused_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
            img[y_start+img_offset_y:y_start+img_offset_y+img_h, x_fused+img_offset_x:x_fused+img_offset_x+img_w] = overlay
    
    def _draw_throttle_multi_layer_row(self, img, original_image, results, branch_idx):
        """保留旧函数名以兼容"""
        self._draw_throttle_all_layers_row(img, original_image, results, branch_idx)
    
    def _draw_steer_all_layers_row(self, img, original_image, results, branch_idx):
        """绘制第四行（Steer 所有卷积层热力图）"""
        y_start = self.row4_y
        panel_h = self.row234_h
        
        # 与Brake行相同的布局
        num_panels = 10
        gap = 6
        total_gap = gap * (num_panels - 1)
        panel_w = (self.WIDTH - 2 * self.MARGIN - total_gap) // num_panels
        img_h = panel_h - 28
        img_w = int(img_h * 200 / 88)
        if img_w > panel_w - 8:
            img_w = panel_w - 8
            img_h = int(img_w * 88 / 200)
        
        orig_resized = cv2.resize(original_image, (img_w, img_h))
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
        
        multi_cams = results.get('steer_multi_layer_cams', {})
        
        all_layers = [
            (-8, 'Steer L1'), (-7, 'Steer L2'), (-6, 'Steer L3'), (-5, 'Steer L4'),
            (-4, 'Steer L5'), (-3, 'Steer L6'), (-2, 'Steer L7'), (-1, 'Steer L8')
        ]
        
        x_start = self.MARGIN
        img_offset_x = (panel_w - img_w) // 2
        img_offset_y = 25
        
        for i, (layer_idx, layer_name) in enumerate(all_layers):
            x = x_start + i * (panel_w + gap)
            self._draw_panel(img, x, y_start, panel_w, panel_h, layer_name, self.ACCENT_YELLOW, 0.4)
            
            cam = multi_cams.get(layer_idx)
            if cam is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
                overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
                img[y_start+img_offset_y:y_start+img_offset_y+img_h, x+img_offset_x:x+img_offset_x+img_w] = overlay
            else:
                cv2.putText(img, "N/A", (x + panel_w//2 - 12, y_start + panel_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 110), 1, cv2.LINE_AA)
        
        # 融合热力图
        x_fused = x_start + len(all_layers) * (panel_w + gap)
        self._draw_panel(img, x_fused, y_start, panel_w, panel_h, "Steer Fused", self.ACCENT_PURPLE, 0.4)
        fused_cam = results.get('steer_fused_cam')
        if fused_cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * fused_cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            overlay = cv2.addWeighted(orig_bgr, 0.35, heatmap_resized, 0.65, 0)
            img[y_start+img_offset_y:y_start+img_offset_y+img_h, x_fused+img_offset_x:x_fused+img_offset_x+img_w] = overlay
    
    def _draw_steer_multi_layer_row(self, img, original_image, results, branch_idx):
        """保留旧函数名以兼容"""
        self._draw_steer_all_layers_row(img, original_image, results, branch_idx)
    
    def _draw_control_panel(self, img, control_result, branch_idx):
        """绘制控制输出面板（第五行）"""
        y = self.row5_y
        h = self.row5_h
        
        # 第五行分为4个面板: Control | Metrics | Branches | Traffic Light
        # 计算宽度分配 (总宽度约2506px)
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 3 * gap
        w1 = int(total_w * 0.20)  # Control Output (约501px)
        
        x = self.MARGIN
        w = w1
        
        self._draw_panel(img, x, y, w, h, "Control Output", title_size=0.45)
        
        # 控制条和标签都在面板内部显示
        # 标签放在控制条上方，值放在控制条右侧但在面板内
        bar_x = x + 15
        bar_w = w - 30  # 控制条宽度
        bar_h = 26
        bar_gap = 52  # 控制条之间的间距
        
        # Brake - 标签在上，值在条内右侧
        by = y + 45
        self._draw_control_bar_internal(img, bar_x, by, bar_w, bar_h, 
                             control_result['brake'], "Brake", self.ACCENT_RED)
        
        # Throttle
        ty = by + bar_gap
        self._draw_control_bar_internal(img, bar_x, ty, bar_w, bar_h,
                             control_result['throttle'], "Throttle", self.ACCENT_GREEN)
        
        # Steer
        sy = ty + bar_gap
        self._draw_steer_bar_internal(img, bar_x, sy, bar_w, bar_h,
                             control_result['steer'], "Steer", self.ACCENT_YELLOW)
        
        branch_color = self.BRANCH_COLORS.get(branch_idx, (150, 150, 150))
        cv2.putText(img, f"[Branch {branch_idx}]", (x + w - 95, y + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, branch_color, 1, cv2.LINE_AA)
    
    def _draw_control_bar_internal(self, img, x, y, w, h, value, label, color):
        """绘制数值条（标签和值都在面板内部）"""
        # 标签在控制条上方
        cv2.putText(img, f"{label}: {value:.3f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # 控制条
        cv2.rectangle(img, (x, y), (x+w, y+h), (45, 45, 50), -1)
        bar_w = int(w * min(1.0, max(0.0, value)))
        if bar_w > 0:
            cv2.rectangle(img, (x, y), (x+bar_w, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 80), 1)
    
    def _draw_steer_bar_internal(self, img, x, y, w, h, value, label, color):
        """绘制转向条（标签和值都在面板内部）"""
        # 标签在控制条上方
        cv2.putText(img, f"{label}: {value:+.3f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # 控制条
        cv2.rectangle(img, (x, y), (x+w, y+h), (45, 45, 50), -1)
        center = x + w // 2
        steer_x = center + int((w//2) * value)
        cv2.rectangle(img, (min(center, steer_x), y), (max(center, steer_x), y+h), color, -1)
        cv2.line(img, (center, y), (center, y+h), (90, 90, 100), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 80), 1)
    
    def _draw_value_bar(self, img, x, y, w, h, value, label, color):
        """绘制数值条"""
        cv2.rectangle(img, (x, y), (x+w, y+h), (45, 45, 50), -1)
        bar_w = int(w * min(1.0, max(0.0, value)))
        if bar_w > 0:
            cv2.rectangle(img, (x, y), (x+bar_w, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 80), 1)
        cv2.putText(img, f"{label}: {value:.3f}", (x+w+12, y+h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def _draw_steer_bar(self, img, x, y, w, h, value, label, color):
        """绘制转向条"""
        cv2.rectangle(img, (x, y), (x+w, y+h), (45, 45, 50), -1)
        center = x + w // 2
        steer_x = center + int((w//2) * value)
        cv2.rectangle(img, (min(center, steer_x), y), (max(center, steer_x), y+h), color, -1)
        cv2.line(img, (center, y), (center, y+h), (90, 90, 100), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 80), 1)
        cv2.putText(img, f"{label}: {value:+.3f}", (x+w+12, y+h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    
    def _draw_quantitative_panel(self, img, results):
        """绘制定量指标面板（第五行）"""
        y = self.row5_y
        h = self.row5_h
        
        # 计算位置 - 与control_panel保持一致的宽度比例
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 3 * gap
        w1 = int(total_w * 0.20)  # Control Output
        w2 = int(total_w * 0.28)  # Quantitative Metrics (约701px)
        
        x = self.MARGIN + w1 + gap
        w = w2
        
        self._draw_panel(img, x, y, w, h, "Quantitative Metrics (Academic)", self.ACCENT_BLUE, 0.45)
        
        metrics = results.get('quantitative_metrics', {})
        is_full = results.get('is_full_analysis', False)
        
        text_x = x + 15
        text_y = y + 40
        line_h = 32  # 增大行间距，避免重叠
        
        if is_full and metrics:
            # 遮挡敏感性
            occ_sens = metrics.get('occlusion_mean_sensitivity', 0)
            cv2.putText(img, f"Occlusion Sensitivity:", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, f"{occ_sens:.5f}", (text_x + 195, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            
            # 积分梯度完整性误差
            ig_err = metrics.get('ig_completeness_error', 0)
            cv2.putText(img, f"IG Completeness Error:", (text_x, text_y + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, f"{ig_err:.5f}", (text_x + 195, text_y + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            
            # 删除AUC
            del_auc = metrics.get('deletion_auc', 0)
            cv2.putText(img, f"Deletion AUC (lower=better):", (text_x, text_y + line_h*2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, f"{del_auc:.5f}", (text_x + 245, text_y + line_h*2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ACCENT_RED, 1, cv2.LINE_AA)
            
            # 插入AUC
            ins_auc = metrics.get('insertion_auc', 0)
            cv2.putText(img, f"Insertion AUC (higher=better):", (text_x, text_y + line_h*3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, f"{ins_auc:.5f}", (text_x + 245, text_y + line_h*3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ACCENT_GREEN, 1, cv2.LINE_AA)
            
            # 综合得分
            combined = metrics.get('combined_score', 0)
            cv2.putText(img, f"Combined Score:", (text_x, text_y + line_h*4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            score_color = self.ACCENT_GREEN if combined > 0 else self.ACCENT_RED
            cv2.putText(img, f"{combined:+.5f}", (text_x + 195, text_y + line_h*4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Real-time analysis enabled", (text_x, text_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, "(Computing...)", (text_x, text_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1, cv2.LINE_AA)

    
    def _draw_branch_panel(self, img, all_predictions, current_branch):
        """绘制分支对比面板（第五行）"""
        y = self.row5_y
        h = self.row5_h
        
        # 计算位置 - 与其他面板保持一致的宽度比例
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 3 * gap
        w1 = int(total_w * 0.20)  # Control Output
        w2 = int(total_w * 0.28)  # Quantitative Metrics
        w3 = int(total_w * 0.28)  # Branch Comparison (约701px)
        
        x = self.MARGIN + w1 + gap + w2 + gap
        w = w3
        
        self._draw_panel(img, x, y, w, h, "All Branches Comparison", title_size=0.45)
        
        if all_predictions is None:
            cv2.putText(img, "No branch data", (x + 20, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            return
        
        branch_names = ['Follow', 'Left', 'Right', 'Straight']
        text_x = x + 12
        text_y = y + 45
        line_h = 35
        
        for i, name in enumerate(branch_names):
            by = text_y + i * line_h
            # 确保不超出面板底部
            if by > y + h - 20:
                break
            s, t, b = all_predictions[i*3], all_predictions[i*3+1], all_predictions[i*3+2]
            is_current = (i == current_branch)
            
            if is_current:
                cv2.rectangle(img, (x + 5, by - 12), (x + w - 5, by + 16), (45, 45, 55), -1)
                marker = ">>"
                color = self.BRANCH_COLORS.get(i, self.TEXT_PRIMARY)
            else:
                marker = "  "
                color = self.TEXT_SECONDARY
            
            text = f"{marker} {name:8s} S:{s:+.2f}  T:{t:.2f}  B:{b:.2f}"
            cv2.putText(img, text, (text_x, by + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    
    def _draw_history_panel(self, img, results):
        """绘制历史曲线面板（第六行）"""
        y = self.row6_y
        h = self.row6_h
        
        # 第六行分为3个面板: History | Curves | Statistics
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 2 * gap
        w1 = int(total_w * 0.40)  # History (约1006px)
        
        x = self.MARGIN
        w = w1
        
        # 获取总帧数用于标题显示
        stats = results.get('brake_stats', {})
        total_frames = stats.get('total_frames', 0)
        self._draw_panel(img, x, y, w, h, f"Control History (All {total_frames} Frames)", title_size=0.45)
        
        # 历史曲线图由外部嵌入，这里只显示占位提示
        # 实际图表由 carla_inference.py 中的 brake_analyzer.plot_history() 嵌入
    
    def _draw_curves_panel(self, img, results):
        """绘制删除/插入曲线面板（第六行）"""
        y = self.row6_y
        h = self.row6_h
        
        # 计算位置
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 2 * gap
        w1 = int(total_w * 0.40)  # History
        w2 = int(total_w * 0.35)  # Curves (约880px)
        
        x = self.MARGIN + w1 + gap
        w = w2
        
        self._draw_panel(img, x, y, w, h, "Deletion/Insertion Curves", title_size=0.45)
        
        di_result = results.get('deletion_insertion')
        if di_result is not None:
            # 曲线图区域：标题高度25px，底部边距15px
            curve_y = y + 28
            curve_h = h - 45
            self._draw_di_curves(img, x + 12, curve_y, w - 24, curve_h, di_result)
        else:
            cv2.putText(img, "Computing curves...", (x + 20, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 110), 1, cv2.LINE_AA)
    
    def _draw_di_curves(self, img, x, y, w, h, di_result):
        """绘制删除/插入曲线（带尺度信息）"""
        # 绘制背景
        cv2.rectangle(img, (x, y), (x+w, y+h), (35, 35, 40), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (55, 55, 65), 1)
        
        deletion_curve = di_result.get('deletion_curve', [])
        insertion_curve = di_result.get('insertion_curve', [])
        
        if len(deletion_curve) < 2 or len(insertion_curve) < 2:
            cv2.putText(img, "Waiting for curves...", (x + 15, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 110), 1, cv2.LINE_AA)
            return
        
        all_vals = list(deletion_curve) + list(insertion_curve)
        v_min, v_max = min(all_vals), max(all_vals)
        if v_max - v_min < 1e-8:
            v_max = v_min + 1
        
        # 边距 - 优化以适应面板
        margin_left = 45
        margin_right = 15
        margin_top = 25
        margin_bottom = 22
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom
        
        # 确保绘图区域有效
        if plot_w <= 0 or plot_h <= 0:
            return
        
        plot_x = x + margin_left
        plot_y = y + margin_top
        
        def to_point(i, val, curve_len):
            if curve_len <= 1:
                return (plot_x, plot_y + plot_h // 2)
            px = plot_x + int(i * plot_w / (curve_len - 1))
            py = plot_y + int((1 - (val - v_min) / (v_max - v_min + 1e-8)) * plot_h)
            # 确保点在绘图区域内
            py = max(plot_y, min(plot_y + plot_h, py))
            return (px, py)
        
        # 绘制Y轴刻度和网格线
        num_y_ticks = 4
        for i in range(num_y_ticks + 1):
            tick_val = v_min + (v_max - v_min) * (num_y_ticks - i) / num_y_ticks
            tick_y = plot_y + int(i * plot_h / num_y_ticks)
            cv2.line(img, (plot_x, tick_y), (plot_x + plot_w, tick_y), (45, 45, 50), 1)
            cv2.putText(img, f"{tick_val:.2f}", (x + 3, tick_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (90, 90, 100), 1, cv2.LINE_AA)
        
        # 绘制X轴刻度
        x_labels = ['0%', '50%', '100%']
        for i, label in enumerate(x_labels):
            tick_x = plot_x + int(i * plot_w / 2)
            cv2.putText(img, label, (tick_x - 12, y + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (90, 90, 100), 1, cv2.LINE_AA)
        
        # 绘制删除曲线
        for i in range(len(deletion_curve) - 1):
            p1 = to_point(i, deletion_curve[i], len(deletion_curve))
            p2 = to_point(i + 1, deletion_curve[i + 1], len(deletion_curve))
            cv2.line(img, p1, p2, self.ACCENT_RED, 2, cv2.LINE_AA)
        
        # 绘制插入曲线
        for i in range(len(insertion_curve) - 1):
            p1 = to_point(i, insertion_curve[i], len(insertion_curve))
            p2 = to_point(i + 1, insertion_curve[i + 1], len(insertion_curve))
            cv2.line(img, p1, p2, self.ACCENT_GREEN, 2, cv2.LINE_AA)
        
        # 图例 - 放在绘图区域上方
        legend_y = y + 12
        cv2.putText(img, "Deletion", (plot_x, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.ACCENT_RED, 1, cv2.LINE_AA)
        cv2.putText(img, "Insertion", (plot_x + 80, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.ACCENT_GREEN, 1, cv2.LINE_AA)
        
        # AUC值 - 放在右上角
        del_auc = di_result.get('deletion_auc', 0)
        ins_auc = di_result.get('insertion_auc', 0)
        
        auc_x = x + w - 95
        cv2.putText(img, f"Del: {del_auc:.3f}", (auc_x, y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.ACCENT_RED, 1, cv2.LINE_AA)
        cv2.putText(img, f"Ins: {ins_auc:.3f}", (auc_x, y + 24), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.ACCENT_GREEN, 1, cv2.LINE_AA)

    
    def _draw_metrics_summary_panel(self, img, results):
        """绘制指标汇总面板（第六行）- Brake Statistics"""
        y = self.row6_y
        h = self.row6_h
        
        # 计算位置
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 2 * gap
        w1 = int(total_w * 0.40)  # History
        w2 = int(total_w * 0.35)  # Curves
        w3 = total_w - w1 - w2   # Statistics (约630px)
        
        x = self.MARGIN + w1 + gap + w2 + gap
        w = w3
        
        self._draw_panel(img, x, y, w, h, "Brake Statistics", title_size=0.45)
        
        stats = results.get('brake_stats', {})
        text_x = x + 15
        text_y = y + 45
        # 动态计算行高，确保所有内容都能显示
        num_items = 6
        available_h = h - 55  # 减去标题和边距
        line_h = min(28, available_h // num_items)
        
        items = [
            ("Total Frames:", f"{stats.get('total_frames', 0)}"),
            ("Brake > 0.1:", f"{stats.get('brake_ratio', 0)*100:.1f}%"),
            ("Hard Brake > 0.5:", f"{stats.get('hard_brake_ratio', 0)*100:.1f}%"),
            ("Avg Brake:", f"{stats.get('avg_brake', 0):.4f}"),
            ("Max Brake:", f"{stats.get('max_brake', 0):.4f}"),
            ("Brake Std:", f"{stats.get('brake_std', 0):.4f}"),
        ]
        
        # 计算标签和值的位置
        label_w = 140  # 标签宽度
        
        for i, (label, value) in enumerate(items):
            item_y = text_y + i * line_h
            # 确保不超出面板底部
            if item_y > y + h - 15:
                break
            cv2.putText(img, label, (text_x, item_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(img, value, (text_x + label_w, item_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
    
    def _draw_traffic_light(self, img, tl_info, control_result):
        """绘制红绿灯状态（第五行右侧）"""
        y = self.row5_y
        h = self.row5_h
        
        # 计算位置 (第五行最右边的面板) - 与其他面板保持一致
        gap = self.PANEL_GAP
        total_w = self.WIDTH - 2 * self.MARGIN - 3 * gap
        w1 = int(total_w * 0.20)  # Control Output
        w2 = int(total_w * 0.28)  # Quantitative Metrics
        w3 = int(total_w * 0.28)  # Branch Comparison
        w4 = total_w - w1 - w2 - w3  # Traffic Light (约24%，约601px)
        
        x = self.MARGIN + w1 + gap + w2 + gap + w3 + gap
        w = w4
        
        self._draw_panel(img, x, y, w, h, "Traffic Light Status", title_size=0.5)
        
        if tl_info is None:
            cv2.putText(img, "No traffic light nearby", (x + 20, y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
            return
        
        state = tl_info.get('state', 'Unknown')
        distance = tl_info.get('distance', -1)
        
        cx, cy = x + 70, y + 110
        radius = 38
        
        colors = {
            'Red': (60, 60, 255),
            'Yellow': (60, 220, 220),
            'Green': (60, 200, 60),
            'Unknown': (100, 100, 100)
        }
        color = colors.get(state, colors['Unknown'])
        
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius, self.TEXT_PRIMARY, 2)
        
        cv2.putText(img, state, (cx + 55, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        
        if distance >= 0:
            cv2.putText(img, f"Distance: {distance:.1f}m", (cx + 55, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        
        if state == 'Red' and control_result['brake'] < 0.3:
            cv2.putText(img, "WARNING: LOW BRAKE!", (x + 15, y + h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.ACCENT_RED, 2, cv2.LINE_AA)
    
    def _draw_footer(self, img, results):
        """绘制底部信息栏"""
        y = self.HEIGHT - self.FOOTER_HEIGHT
        cv2.rectangle(img, (0, y), (self.WIDTH, self.HEIGHT), (25, 25, 30), -1)
        cv2.line(img, (0, y), (self.WIDTH, y), self.PANEL_BORDER, 1)
        
        methods = "Methods: Grad-CAM++ (Multi-Layer) | Occlusion Sensitivity | Integrated Gradients | Deletion/Insertion Curves"
        cv2.putText(img, methods, (20, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_SECONDARY, 1, cv2.LINE_AA)
        
        refs = "References: Selvaraju 2017, Chattopadhay 2018, Sundararajan 2017, Petsiuk 2018"
        cv2.putText(img, refs, (20, y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 100), 1, cv2.LINE_AA)
        
        is_full = results.get('is_full_analysis', False)
        status = "FULL ANALYSIS" if is_full else "Quick Analysis"
        status_color = self.ACCENT_GREEN if is_full else self.TEXT_SECONDARY
        cv2.putText(img, f"[{status}]", (self.WIDTH - 160, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
    
    def save_frame(self, dashboard, prefix="dashboard"):
        """保存帧"""
        if self.save_dir:
            filename = f"{prefix}_{self.frame_count:06d}.png"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, dashboard)
            return filepath
        return None



# ============================================================================
# 第五部分：主接口类
# ============================================================================

class InterpretabilityVisualizer:
    """
    可解释性可视化器（主接口）v3.0
    
    改进：
    - 支持 Grad-CAM++ 和所有卷积层分析
    - 2560x1440 仪表板 (2K分辨率)
    - 更好的分辨率和可视化效果
    - 支持选择计算设备（GPU/CPU）
    - 支持配置第一行和第二行热力图的卷积层
    """
    
    def __init__(self, model, device, save_dir=None,
                 enable_occlusion=True,
                 enable_integrated_gradients=True,
                 enable_deletion_insertion=True,
                 grad_cam_layer_index=-3,
                 use_gradcam_pp=True,
                 enable_multi_layer=True,
                 multi_layer_indices=None,
                 history_max_frames=None,
                 ig_steps=30):
        """
        参数:
            model: PyTorch模型
            device: 计算设备 (可以是 'cpu' 或 'cuda:x')
            save_dir: 保存目录
            grad_cam_layer_index: 第一行Grad-CAM热力图使用的目标层索引
            use_gradcam_pp: 使用 Grad-CAM++
            enable_multi_layer: 启用多层级分析
            multi_layer_indices: 第二行多层级分析使用的层索引列表
            history_max_frames: 历史记录最大帧数，None表示记录所有帧
            ig_steps: 积分梯度的积分步数
        """
        self.device = device
        self.save_dir = save_dir
        self._owns_model_copy = False  # 标记是否拥有模型副本
        
        # 如果指定的设备与模型当前设备不同，需要复制模型到目标设备
        model_device = next(model.parameters()).device
        if str(device) != str(model_device):
            import copy
            print(f"📋 复制模型到 {device} 用于可解释性分析...")
            print(f"   ⚠️ 注意: 这会增加内存使用（模型将存在两份副本）")
            self.model = copy.deepcopy(model).to(device)
            self.model.eval()
            self._owns_model_copy = True  # 标记我们拥有这个副本
            print(f"   原模型设备: {model_device}, 分析模型设备: {device}")
        else:
            self.model = model
        
        # 默认多层索引
        if multi_layer_indices is None:
            multi_layer_indices = [-1, -3, -5]
        
        # 初始化分析器
        self.analyzer = ComprehensiveAnalyzer(
            self.model, device,
            enable_occlusion=enable_occlusion,
            enable_integrated_gradients=enable_integrated_gradients,
            enable_deletion_insertion=enable_deletion_insertion,
            grad_cam_layer_index=grad_cam_layer_index,
            use_gradcam_pp=use_gradcam_pp,
            enable_multi_layer=enable_multi_layer,
            multi_layer_indices=multi_layer_indices,
            history_max_frames=history_max_frames,
            ig_steps=ig_steps
        )
        
        # 初始化仪表板（1920x1080）
        self.dashboard = InterpretabilityDashboard(save_dir)
        
        # 兼容旧接口
        self.grad_cam = self.analyzer.grad_cam
        self.brake_analyzer = self.analyzer.brake_analyzer
        
        self.frame_count = 0
        self._last_results = None
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"✅ 可解释性可视化器 v3.0 已初始化")
            print(f"   分辨率: 2560x1440 (2K)")
            print(f"   保存目录: {save_dir}")
            print(f"   Grad-CAM++: {'启用' if use_gradcam_pp else '禁用'}")
            print(f"   多层级分析: {'启用' if enable_multi_layer else '禁用'}")
            print(f"   第一行热力图层索引: {grad_cam_layer_index}")
            print(f"   第二行多层索引: {multi_layer_indices}")
            print(f"   历史记录帧数: {'无限制' if history_max_frames is None else history_max_frames}")
    
    def analyze_frame(self, img_tensor, speed_tensor, original_image,
                      control_result, current_command, target_bbox=None):
        """分析单帧"""
        self.frame_count += 1
        
        results = self.analyzer.analyze_frame(
            img_tensor, speed_tensor, original_image,
            control_result, current_command
        )
        
        self._last_results = results
        return results
    
    def render_dashboard(self, original_image, analysis_results, control_result,
                         current_command, traffic_light_info=None, 
                         all_branch_predictions=None):
        """渲染仪表板（1920x1080）"""
        return self.dashboard.render(
            original_image, analysis_results, control_result,
            current_command, traffic_light_info, all_branch_predictions
        )
    
    def set_analysis_interval(self, interval):
        """设置完整分析间隔"""
        self.analyzer.analysis_interval = interval
    
    def get_metrics(self):
        """获取当前指标"""
        return self.analyzer.metrics_collector.metrics_history
    
    def get_metrics_summary(self):
        """获取指标汇总"""
        return self.analyzer.get_metrics_summary()
    
    def save_metrics(self, filepath=None):
        """保存指标到文件"""
        if filepath is None:
            if self.save_dir:
                filepath = os.path.join(self.save_dir, 'metrics.json')
            else:
                filepath = './interpretability_metrics.json'
        
        self.analyzer.export_metrics(filepath)
    
    def get_latex_table(self):
        """获取LaTeX格式的指标表格"""
        return self.analyzer.metrics_collector.export_to_latex_table()
    
    def cleanup(self):
        """清理资源（钩子、模型副本等）"""
        # 清理分析器（包括所有 GradCAM 钩子）
        if hasattr(self, 'analyzer') and self.analyzer is not None:
            self.analyzer.cleanup()
        
        # 如果我们拥有模型副本，释放它以节省内存
        if hasattr(self, '_owns_model_copy') and self._owns_model_copy:
            del self.model
            self.model = None
            self._owns_model_copy = False
            # 尝试触发垃圾回收
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# 便捷函数
# ============================================================================

def create_interpretability_visualizer(model, device, save_dir=None, full_analysis=True,
                                       grad_cam_layer_index=None, multi_layer_indices=None,
                                       ig_steps=None):
    """
    创建可解释性可视化器 v2.0
    
    参数:
        model: PyTorch模型
        device: 计算设备
        save_dir: 保存目录
        full_analysis: 是否启用所有分析方法
        grad_cam_layer_index: 第一行热力图使用的卷积层索引（命令行参数优先）
        multi_layer_indices: 第二行多层级热力图使用的卷积层索引列表（命令行参数优先）
        ig_steps: 积分梯度的积分步数（命令行参数优先）
    
    返回:
        InterpretabilityVisualizer 实例
    """
    # 尝试从配置文件导入默认参数
    try:
        from carla_config import (
            DASHBOARD_ROW1_LAYER_INDEX,
            DASHBOARD_ROW2_LAYER_INDICES,
            DASHBOARD_IG_STEPS,
            DASHBOARD_HISTORY_MAX_FRAMES
        )
        # 命令行参数优先，否则使用配置文件参数
        if grad_cam_layer_index is None:
            grad_cam_layer_index = DASHBOARD_ROW1_LAYER_INDEX
        if multi_layer_indices is None:
            multi_layer_indices = DASHBOARD_ROW2_LAYER_INDICES
        if ig_steps is None:
            ig_steps = DASHBOARD_IG_STEPS
        history_max_frames = DASHBOARD_HISTORY_MAX_FRAMES
        print(f"✅ 可解释性参数:")
        print(f"   第一行热力图层索引: {grad_cam_layer_index}")
        print(f"   第二行多层索引: {multi_layer_indices}")
        print(f"   积分梯度步数: {ig_steps}")
        print(f"   历史记录帧数: {'无限制' if history_max_frames is None else history_max_frames}")
    except ImportError:
        # 使用默认值
        if grad_cam_layer_index is None:
            grad_cam_layer_index = -3
        if multi_layer_indices is None:
            multi_layer_indices = [-1, -3, -5]
        if ig_steps is None:
            ig_steps = 30
        history_max_frames = None
        print("⚠️ 未找到配置文件，使用默认/命令行参数")
    
    return InterpretabilityVisualizer(
        model, device, save_dir,
        enable_occlusion=full_analysis,
        enable_integrated_gradients=full_analysis,
        enable_deletion_insertion=full_analysis,
        grad_cam_layer_index=grad_cam_layer_index,
        use_gradcam_pp=True,
        enable_multi_layer=True,
        multi_layer_indices=multi_layer_indices,
        history_max_frames=history_max_frames,
        ig_steps=ig_steps
    )


# ============================================================================
# 注意力IoU计算（可选功能）
# ============================================================================

class AttentionIoU:
    """
    注意力-目标IoU计算
    
    原理：计算高注意力区域与目标区域（如红绿灯）的交并比
    用途：量化模型是否关注正确的区域
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.last_metrics = {}
    
    def compute(self, saliency_map, target_bbox=None, target_mask=None):
        """
        计算注意力与目标的IoU
        
        参数:
            saliency_map: 显著性图 (H, W)
            target_bbox: 目标边界框 (x1, y1, x2, y2)
            target_mask: 目标掩码 (H, W)
        
        返回:
            IoU值
        """
        if saliency_map is None:
            return 0.0
        
        # 二值化显著性图
        attention_mask = (saliency_map > self.threshold).astype(np.float32)
        
        if target_mask is not None:
            target = target_mask.astype(np.float32)
        elif target_bbox is not None:
            x1, y1, x2, y2 = target_bbox
            target = np.zeros_like(saliency_map)
            target[y1:y2, x1:x2] = 1.0
        else:
            return 0.0
        
        # 计算IoU
        intersection = np.sum(attention_mask * target)
        union = np.sum(attention_mask) + np.sum(target) - intersection
        
        iou = intersection / (union + 1e-8)
        
        self.last_metrics = {
            'iou': float(iou),
            'attention_area': float(np.sum(attention_mask)),
            'target_area': float(np.sum(target)),
            'intersection': float(intersection)
        }
        
        return float(iou)
    
    def get_last_metrics(self):
        return self.last_metrics
