#!/usr/bin/env python
# coding=utf-8
"""
报告生成器

提供数据验证报告、删除报告的生成功能，支持可视化图表和多格式输出。
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from ..config import COMMAND_NAMES


class VerificationReport:
    """验证报告生成器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.report_data: Dict[str, Any] = {}
    
    def generate(self, 
                 total_frames: int,
                 command_stats: Dict[int, int],
                 speed_stats: List[float],
                 steer_stats: List[float],
                 throttle_stats: List[float],
                 brake_stats: List[float],
                 file_sizes: List[float],
                 corrupted_files: List[tuple],
                 warning_files: List[tuple],
                 incomplete_files: List[tuple],
                 total_files: int) -> Dict[str, Any]:
        """
        生成验证报告数据
        
        返回:
            Dict: 报告数据
        """
        self.report_data = {
            'verification_time': datetime.now().isoformat(),
            'data_path': self.data_path,
            'file_statistics': {
                'total_files': total_files,
                'corrupted_files': len(corrupted_files),
                'warning_files': len(warning_files),
                'incomplete_files': len(incomplete_files),
                'valid_files': total_files - len(corrupted_files) - len(incomplete_files),
                'average_file_size_mb': float(np.mean(file_sizes)) if file_sizes else 0,
                'total_data_size_mb': float(np.sum(file_sizes)) if file_sizes else 0
            },
            'frame_statistics': {
                'total_frames': int(total_frames),
                'average_frames_per_file': int(total_frames / max(total_files, 1))
            },
            'command_distribution': {
                COMMAND_NAMES.get(cmd, f'Unknown({cmd})'): int(count) 
                for cmd, count in command_stats.items()
            },
            'speed_statistics': {
                'mean': float(np.mean(speed_stats)) if speed_stats else 0,
                'min': float(np.min(speed_stats)) if speed_stats else 0,
                'max': float(np.max(speed_stats)) if speed_stats else 0,
                'median': float(np.median(speed_stats)) if speed_stats else 0
            },
            'control_statistics': {
                'steer': {
                    'min': float(np.min(steer_stats)) if steer_stats else 0,
                    'max': float(np.max(steer_stats)) if steer_stats else 0,
                    'mean': float(np.mean(steer_stats)) if steer_stats else 0
                },
                'throttle': {
                    'min': float(np.min(throttle_stats)) if throttle_stats else 0,
                    'max': float(np.max(throttle_stats)) if throttle_stats else 0,
                    'mean': float(np.mean(throttle_stats)) if throttle_stats else 0
                },
                'brake': {
                    'min': float(np.min(brake_stats)) if brake_stats else 0,
                    'max': float(np.max(brake_stats)) if brake_stats else 0,
                    'mean': float(np.mean(brake_stats)) if brake_stats else 0
                }
            },
            'quality_scores': self._calculate_quality_scores(
                command_stats, speed_stats, steer_stats, file_sizes, total_frames
            ),
            'corrupted_files': [f[0] if isinstance(f, tuple) else f for f in corrupted_files],
            'warning_files': [(f, w) for f, w in warning_files],
            'incomplete_files': [{'filename': f, 'frame_count': c} for f, c in incomplete_files]
        }
        
        return self.report_data
    
    def _calculate_quality_scores(self, command_stats: Dict, speed_stats: List,
                                   steer_stats: List, file_sizes: List,
                                   total_frames: int) -> Dict[str, float]:
        """计算数据质量评分"""
        scores = {}
        
        # 命令完整性：有多少种有效命令（2,3,4,5）
        valid_cmd_count = len([c for c in command_stats.keys() if c in {2, 3, 4, 5}])
        scores['command_completeness'] = min(100, valid_cmd_count / 4 * 100)
        
        # 速度合理性：速度在合理范围内的比例
        if speed_stats:
            invalid_speed_ratio = sum(1 for s in speed_stats if s > 100 or s < 0) / len(speed_stats)
            scores['speed_validity'] = min(100, (1 - invalid_speed_ratio) * 100)
        else:
            scores['speed_validity'] = 0
        
        # 数据量评分（10万帧为满分）
        scores['data_volume'] = min(100, total_frames / 100000 * 100)
        
        # 文件健康度
        if file_sizes:
            small_file_ratio = len([f for f in file_sizes if f < 0.1]) / len(file_sizes)
            scores['file_health'] = min(100, (1 - small_file_ratio) * 100)
        else:
            scores['file_health'] = 0
        
        # 方向盘平衡性
        if steer_stats:
            left_ratio = sum(1 for s in steer_stats if s < -0.1) / len(steer_stats)
            right_ratio = sum(1 for s in steer_stats if s > 0.1) / len(steer_stats)
            scores['steer_balance'] = min(100, (1 - abs(left_ratio - right_ratio)) * 100)
        else:
            scores['steer_balance'] = 0
        
        return scores
    
    def save_json(self, filename: str = 'verification_report.json') -> str:
        """保存JSON格式报告"""
        filepath = os.path.join(self.data_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=4, ensure_ascii=False)
        return filepath
    
    def print_summary(self):
        """打印报告摘要"""
        data = self.report_data
        
        print("\n" + "="*70)
        print("📊 验证报告")
        print("="*70)
        
        # 文件统计
        fs = data['file_statistics']
        print(f"\n📁 文件统计:")
        print(f"  • 总文件数: {fs['total_files']}")
        print(f"  • 有效文件: {fs['valid_files']}")
        print(f"  • 损坏文件: {fs['corrupted_files']}")
        print(f"  • 不完整文件: {fs['incomplete_files']}")
        print(f"  • 平均文件大小: {fs['average_file_size_mb']:.2f} MB")
        print(f"  • 总数据大小: {fs['total_data_size_mb']:.2f} MB ({fs['total_data_size_mb']/1024:.2f} GB)")
        
        # 帧统计
        print(f"\n🎬 帧统计:")
        print(f"  • 总帧数: {data['frame_statistics']['total_frames']:,}")
        
        # 命令分布
        print(f"\n🎯 命令分布:")
        total = data['frame_statistics']['total_frames']
        for cmd, count in data['command_distribution'].items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  • {cmd}: {count:,} 帧 ({pct:.1f}%)")
        
        # 速度统计
        ss = data['speed_statistics']
        print(f"\n🚗 速度统计:")
        print(f"  • 平均: {ss['mean']:.1f} km/h")
        print(f"  • 范围: {ss['min']:.1f} ~ {ss['max']:.1f} km/h")
        
        # 控制信号
        cs = data['control_statistics']
        print(f"\n🎮 控制信号:")
        print(f"  • 方向盘: {cs['steer']['min']:.3f} ~ {cs['steer']['max']:.3f}")
        print(f"  • 油门: {cs['throttle']['min']:.3f} ~ {cs['throttle']['max']:.3f}")
        print(f"  • 刹车: {cs['brake']['min']:.3f} ~ {cs['brake']['max']:.3f}")
        
        # 质量评分
        qs = data['quality_scores']
        print(f"\n📈 质量评分:")
        score_names = {
            'command_completeness': '命令完整性',
            'speed_validity': '速度合理性',
            'data_volume': '数据量',
            'file_health': '文件健康',
            'steer_balance': '转向平衡'
        }
        for key, name in score_names.items():
            score = qs.get(key, 0)
            indicator = '🟢' if score >= 80 else ('🟡' if score >= 60 else '🔴')
            print(f"  {indicator} {name}: {score:.1f}")
        
        print("\n" + "="*70)


class DeletionReport:
    """删除报告生成器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.deleted_files: List[tuple] = []
        self.delete_enabled: bool = False
    
    def add_file(self, filepath: str, reason: str):
        """添加待删除/已删除文件"""
        self.deleted_files.append((filepath, reason))
    
    def generate(self, delete_enabled: bool = False) -> Dict[str, Any]:
        """生成删除报告"""
        self.delete_enabled = delete_enabled
        
        # 按原因分类
        reason_categories = defaultdict(list)
        for filepath, reason in self.deleted_files:
            if '帧数不足' in reason:
                reason_categories['帧数不足'].append({'file': filepath, 'detail': reason})
            elif '图像过暗' in reason:
                reason_categories['图像过暗'].append({'file': filepath, 'detail': reason})
            elif '无效命令' in reason:
                reason_categories['无效命令值'].append({'file': filepath, 'detail': reason})
            elif '文件损坏' in reason or '验证失败' in reason:
                reason_categories['文件损坏'].append({'file': filepath, 'detail': reason})
            else:
                reason_categories['其他'].append({'file': filepath, 'detail': reason})
        
        return {
            'report_time': datetime.now().isoformat(),
            'delete_enabled': delete_enabled,
            'status': '已删除' if delete_enabled else '待删除（预览模式）',
            'total_invalid_files': len(self.deleted_files),
            'summary': {cat: len(files) for cat, files in reason_categories.items()},
            'details': dict(reason_categories)
        }
    
    def save(self) -> tuple:
        """保存删除报告（JSON和TXT格式）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_data = self.generate(self.delete_enabled)
        
        # JSON报告
        json_path = os.path.join(self.data_path, f'deletion_report_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        
        # TXT报告
        txt_path = os.path.join(self.data_path, f'deletion_report_{timestamp}.txt')
        self._write_txt_report(txt_path, report_data)
        
        return json_path, txt_path
    
    def _write_txt_report(self, filepath: str, data: Dict):
        """写入TXT格式报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("📋 数据文件删除报告\n")
            f.write("="*70 + "\n\n")
            f.write(f"报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"操作状态: {'✅ 已删除' if data['delete_enabled'] else '⚠️ 预览模式'}\n")
            f.write(f"不满足条件的文件总数: {data['total_invalid_files']}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("📊 按原因分类统计\n")
            f.write("-"*70 + "\n")
            for category, count in data['summary'].items():
                f.write(f"  • {category}: {count} 个文件\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("📝 详细列表\n")
            f.write("-"*70 + "\n\n")
            
            for category, files in data['details'].items():
                f.write(f"\n【{category}】({len(files)} 个文件)\n")
                f.write("-"*40 + "\n")
                for item in files[:20]:  # 限制每类最多显示20个
                    f.write(f"  文件: {os.path.basename(item['file'])}\n")
                    f.write(f"  原因: {item['detail']}\n\n")
                if len(files) > 20:
                    f.write(f"  ... 还有 {len(files) - 20} 个文件\n\n")
            
            f.write("="*70 + "\n")
            f.write("报告结束\n")
            f.write("="*70 + "\n")
    
    def print_summary(self):
        """打印删除报告摘要"""
        data = self.generate(self.delete_enabled)
        
        print(f"\n" + "="*70)
        print(f"🗑️  删除报告摘要")
        print("="*70)
        print(f"状态: {'✅ 已删除' if data['delete_enabled'] else '⚠️ 预览模式'}")
        print(f"不满足条件的文件总数: {data['total_invalid_files']}")
        print(f"\n按原因分类:")
        for category, count in data['summary'].items():
            print(f"  • {category}: {count} 个文件")
        print("="*70 + "\n")


class ChartGenerator:
    """图表生成器（可选依赖matplotlib）"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._matplotlib_available = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """检查matplotlib是否可用"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except ImportError:
            return False
    
    def generate_charts(self, report_data: Dict[str, Any]) -> Optional[str]:
        """
        生成可视化图表
        
        参数:
            report_data: 验证报告数据
            
        返回:
            str: 图表文件路径，matplotlib不可用时返回None
        """
        if not self._matplotlib_available:
            print("⚠️ matplotlib不可用，跳过图表生成")
            return None
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('数据收集统计报告', fontsize=16, fontweight='bold')
        
        # 1. 命令分布饼图
        self._plot_command_pie(axes[0, 0], report_data['command_distribution'])
        
        # 2. 速度分布（使用统计数据）
        self._plot_speed_info(axes[0, 1], report_data['speed_statistics'])
        
        # 3. 控制信号统计
        self._plot_control_stats(axes[1, 0], report_data['control_statistics'])
        
        # 4. 质量评分
        self._plot_quality_scores(axes[1, 1], report_data['quality_scores'])
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.data_path, 'verification_report.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _plot_command_pie(self, ax, command_dist: Dict):
        """绘制命令分布饼图"""
        if not command_dist:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title('命令分布')
            return
        
        labels = list(command_dist.keys())
        sizes = list(command_dist.values())
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
               colors=colors[:len(sizes)], startangle=90)
        ax.set_title('命令分布')
    
    def _plot_speed_info(self, ax, speed_stats: Dict):
        """绘制速度统计信息"""
        metrics = ['最小值', '平均值', '中位数', '最大值']
        values = [
            speed_stats.get('min', 0),
            speed_stats.get('mean', 0),
            speed_stats.get('median', speed_stats.get('mean', 0)),
            speed_stats.get('max', 0)
        ]
        
        bars = ax.bar(metrics, values, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_ylabel('速度 (km/h)')
        ax.set_title('速度统计')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_control_stats(self, ax, control_stats: Dict):
        """绘制控制信号统计"""
        controls = ['方向盘', '油门', '刹车']
        mins = [control_stats['steer']['min'], control_stats['throttle']['min'], control_stats['brake']['min']]
        maxs = [control_stats['steer']['max'], control_stats['throttle']['max'], control_stats['brake']['max']]
        means = [control_stats['steer']['mean'], control_stats['throttle']['mean'], control_stats['brake']['mean']]
        
        x = np.arange(len(controls))
        width = 0.25
        
        ax.bar(x - width, mins, width, label='最小值', color='lightblue')
        ax.bar(x, means, width, label='平均值', color='steelblue')
        ax.bar(x + width, maxs, width, label='最大值', color='darkblue')
        
        ax.set_xticks(x)
        ax.set_xticklabels(controls)
        ax.set_title('控制信号统计')
        ax.legend()
    
    def _plot_quality_scores(self, ax, quality_scores: Dict):
        """绘制质量评分条形图"""
        score_names = {
            'command_completeness': '命令完整性',
            'speed_validity': '速度合理性',
            'data_volume': '数据量',
            'file_health': '文件健康',
            'steer_balance': '转向平衡'
        }
        
        metrics = [score_names.get(k, k) for k in quality_scores.keys()]
        scores = list(quality_scores.values())
        colors = ['green' if s >= 80 else 'orange' if s >= 60 else 'red' for s in scores]
        
        bars = ax.barh(metrics, scores, color=colors, alpha=0.7)
        ax.set_xlabel('评分')
        ax.set_title('数据质量评分')
        ax.set_xlim(0, 100)
        
        for bar, score in zip(bars, scores):
            ax.text(score + 2, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}', va='center', fontsize=9)
