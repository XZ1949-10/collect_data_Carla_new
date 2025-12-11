#!/usr/bin/env python3
"""
CARLA-CIL 项目架构图生成器
风格：类似 Docker 鲸鱼的卡通风格，使用可爱的小汽车作为吉祥物
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Arc, Wedge
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_cute_car(ax, x, y, scale=1.0, color='#3498db'):
    """绘制可爱的卡通小汽车（类似 Docker 鲸鱼风格）"""
    s = scale
    
    # 车身主体（圆润的矩形）
    body = FancyBboxPatch((x - 2*s, y - 0.6*s), 4*s, 1.2*s,
                          boxstyle="round,pad=0,rounding_size=0.3",
                          facecolor=color, edgecolor='#2980b9', linewidth=2)
    ax.add_patch(body)
    
    # 车顶（半圆形）
    roof = FancyBboxPatch((x - 1.2*s, y + 0.5*s), 2.4*s, 1*s,
                          boxstyle="round,pad=0,rounding_size=0.4",
                          facecolor=color, edgecolor='#2980b9', linewidth=2)
    ax.add_patch(roof)
    
    # 车窗（浅蓝色）
    window = FancyBboxPatch((x - 1*s, y + 0.6*s), 2*s, 0.7*s,
                            boxstyle="round,pad=0,rounding_size=0.2",
                            facecolor='#87CEEB', edgecolor='#5DADE2', linewidth=1.5)
    ax.add_patch(window)
    
    # 车灯（前后）
    front_light = Circle((x + 1.8*s, y), 0.2*s, facecolor='#F1C40F', edgecolor='#F39C12', linewidth=1.5)
    back_light = Circle((x - 1.8*s, y), 0.15*s, facecolor='#E74C3C', edgecolor='#C0392B', linewidth=1.5)
    ax.add_patch(front_light)
    ax.add_patch(back_light)
    
    # 车轮
    wheel1 = Circle((x - 1.2*s, y - 0.7*s), 0.4*s, facecolor='#2C3E50', edgecolor='#1A252F', linewidth=2)
    wheel2 = Circle((x + 1.2*s, y - 0.7*s), 0.4*s, facecolor='#2C3E50', edgecolor='#1A252F', linewidth=2)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    
    # 轮毂
    hub1 = Circle((x - 1.2*s, y - 0.7*s), 0.15*s, facecolor='#BDC3C7', edgecolor='#95A5A6', linewidth=1)
    hub2 = Circle((x + 1.2*s, y - 0.7*s), 0.15*s, facecolor='#BDC3C7', edgecolor='#95A5A6', linewidth=1)
    ax.add_patch(hub1)
    ax.add_patch(hub2)
    
    # 可爱的眼睛（在车窗上）
    eye1 = Circle((x - 0.4*s, y + 0.95*s), 0.18*s, facecolor='white', edgecolor='#2C3E50', linewidth=1.5)
    eye2 = Circle((x + 0.4*s, y + 0.95*s), 0.18*s, facecolor='white', edgecolor='#2C3E50', linewidth=1.5)
    pupil1 = Circle((x - 0.35*s, y + 0.95*s), 0.08*s, facecolor='#2C3E50')
    pupil2 = Circle((x + 0.45*s, y + 0.95*s), 0.08*s, facecolor='#2C3E50')
    ax.add_patch(eye1)
    ax.add_patch(eye2)
    ax.add_patch(pupil1)
    ax.add_patch(pupil2)
    
    # 微笑
    smile = Arc((x, y + 0.5*s), 0.6*s, 0.3*s, angle=0, theta1=200, theta2=340,
                color='#2C3E50', linewidth=2)
    ax.add_patch(smile)

def draw_module_box(ax, x, y, width, height, title, items, color, icon='📦'):
    """绘制模块框"""
    # 主框体
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='#2C3E50',
                         linewidth=2, alpha=0.9)
    ax.add_patch(box)
    
    # 标题栏
    title_bar = FancyBboxPatch((x, y + height - 0.6), width, 0.6,
                               boxstyle="round,pad=0,rounding_size=0.1",
                               facecolor='#2C3E50', edgecolor='none', alpha=0.8)
    ax.add_patch(title_bar)
    
    # 标题文字
    ax.text(x + width/2, y + height - 0.3, f'{icon} {title}',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 内容项
    for i, item in enumerate(items):
        ax.text(x + 0.15, y + height - 1.0 - i*0.4, f'• {item}',
                ha='left', va='center', fontsize=8, color='#2C3E50')

def draw_arrow(ax, start, end, color='#7F8C8D', style='->'):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=2,
                               connectionstyle='arc3,rad=0.1'))

def draw_flow_arrow(ax, start, end, label='', color='#3498db'):
    """绘制带标签的流程箭头"""
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=3,
                               connectionstyle='arc3,rad=0'))
    if label:
        ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9))

def main():
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 背景
    bg = FancyBboxPatch((-0.5, -0.5), 17, 13,
                        boxstyle="round,pad=0,rounding_size=0.3",
                        facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=3)
    ax.add_patch(bg)
    
    # ========== 标题区域 ==========
    ax.text(8, 12, '🚗 CARLA-CIL 项目架构', ha='center', va='center',
            fontsize=20, fontweight='bold', color='#2C3E50')
    ax.text(8, 11.4, '基于条件模仿学习的端到端自动驾驶系统', ha='center', va='center',
            fontsize=12, color='#7F8C8D')
    
    # ========== 绘制可爱的小汽车吉祥物 ==========
    draw_cute_car(ax, 14.5, 11, scale=0.8, color='#3498db')
    
    # ========== 主要模块 ==========
    
    # 1. 数据收集模块 (collect_data_new)
    draw_module_box(ax, 0.5, 6.5, 4.5, 4,
                    'collect_data_new', 
                    ['auto_collector.py', 'command_based.py', 
                     'route_planner.py', 'npc_manager.py',
                     'noiser.py (噪声注入)', 'anomaly_detector.py'],
                    '#E8F8F5', '📦')
    
    # 2. 模型训练模块 (carla_train)
    draw_module_box(ax, 6, 6.5, 4.5, 4,
                    'carla_train',
                    ['main_ddp.py (分布式训练)', 'carla_net_ori.py (网络)',
                     'carla_loader_ddp.py', 'finetune.py',
                     'helper.py', 'test.py'],
                    '#FEF9E7', '🧠')
    
    # 3. 模型推理模块 (carla_0.9.16)
    draw_module_box(ax, 11.5, 6.5, 4.5, 4,
                    'carla_0.9.16',
                    ['carla_inference.py', 'carla_model_predictor.py',
                     'carla_sensors.py', 'carla_visualizer.py',
                     'vehicle_controller.py', 'navigation_adapter.py'],
                    '#F5EEF8', '🔮')
    
    # 4. 导航代理模块 (agents)
    draw_module_box(ax, 0.5, 1.5, 4.5, 3.5,
                    'agents/navigation',
                    ['global_route_planner.py', 'local_planner.py',
                     'basic_agent.py', 'controller.py',
                     'behavior_agent.py'],
                    '#EBF5FB', '🗺️')
    
    # 5. 核心子模块 (core)
    draw_module_box(ax, 6, 1.5, 4.5, 3.5,
                    'core 核心模块',
                    ['base_collector.py', 'resource_manager.py',
                     'sync_mode_manager.py', 'collision_recovery.py',
                     'weather_manager.py'],
                    '#FDEDEC', '⚙️')
    
    # 6. 工具模块 (utils)
    draw_module_box(ax, 11.5, 1.5, 4.5, 3.5,
                    'utils 工具模块',
                    ['visualization.py', 'data_utils.py',
                     'balance_selector.py', 'report_generator.py',
                     'carla_visualizer.py'],
                    '#E8F6F3', '🔧')
    
    # ========== 流程箭头 ==========
    # 数据收集 -> 训练
    draw_flow_arrow(ax, (5, 8.5), (6, 8.5), 'H5数据', '#27AE60')
    
    # 训练 -> 推理
    draw_flow_arrow(ax, (10.5, 8.5), (11.5, 8.5), '模型.pth', '#E74C3C')
    
    # 导航 -> 数据收集
    draw_flow_arrow(ax, (2.75, 5), (2.75, 6.5), '路径规划', '#3498db')
    
    # 核心 -> 数据收集
    draw_flow_arrow(ax, (8.25, 5), (5, 7), '基础功能', '#9B59B6')
    
    # 工具 -> 各模块
    ax.annotate('', xy=(13.75, 5), xytext=(13.75, 6.5),
                arrowprops=dict(arrowstyle='->', color='#1ABC9C', lw=2))
    
    # ========== 数据流说明框 ==========
    flow_box = FancyBboxPatch((0.5, -0.3), 15.5, 1.2,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#FDFEFE', edgecolor='#3498db',
                              linewidth=2, alpha=0.95)
    ax.add_patch(flow_box)
    
    ax.text(8.25, 0.3, '📊 数据流: CARLA仿真 → 数据收集(H5) → 模型训练(PyTorch DDP) → 实时推理 → 车辆控制',
            ha='center', va='center', fontsize=10, color='#2C3E50', fontweight='bold')
    
    # ========== 技术栈标签 ==========
    tech_labels = [
        ('CARLA 0.9.16', 1.5, 10.8, '#E74C3C'),
        ('PyTorch', 4, 10.8, '#EE4C2C'),
        ('Python 3.8+', 6.5, 10.8, '#3776AB'),
        ('NumPy', 9, 10.8, '#013243'),
        ('OpenCV', 11.5, 10.8, '#5C3EE8'),
    ]
    
    for label, x, y, color in tech_labels:
        badge = FancyBboxPatch((x - 0.6, y - 0.2), 1.8, 0.5,
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(badge)
        ax.text(x + 0.3, y + 0.05, label, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    
    # ========== CIL 网络架构简图 ==========
    net_box = FancyBboxPatch((6, 10.2), 4.5, 0.8,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#FFF5E6', edgecolor='#F39C12',
                             linewidth=2, alpha=0.95)
    ax.add_patch(net_box)
    ax.text(8.25, 10.6, '🧠 CIL网络: RGB图像 + 速度 → CNN → 4分支 → [转向,油门,刹车]',
            ha='center', va='center', fontsize=8, color='#2C3E50')
    
    # ========== 保存图片 ==========
    plt.tight_layout()
    plt.savefig('carla_cil_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('carla_cil_architecture.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('✅ 架构图已保存:')
    print('   - carla_cil_architecture.png')
    print('   - carla_cil_architecture.svg')
    plt.show()

if __name__ == '__main__':
    main()
