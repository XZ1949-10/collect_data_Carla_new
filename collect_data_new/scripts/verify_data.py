#!/usr/bin/env python
# coding=utf-8
"""
数据验证脚本

使用方法:
    python -m collect_data_new.scripts.verify_data --data-path ./carla_data
    python -m collect_data_new.scripts.verify_data --data-path ./carla_data --min-frames 200
    python -m collect_data_new.scripts.verify_data --data-path ./carla_data --delete-invalid
    python -m collect_data_new.scripts.verify_data --data-path ./carla_data --no-charts
"""

import argparse
import os
import sys
import h5py
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collect_data_new.utils import DataLoader
from collect_data_new.utils.report_generator import VerificationReport, DeletionReport, ChartGenerator
from collect_data_new.config import COMMAND_NAMES


class DataVerifier:
    """数据验证器"""
    
    # 有效命令值（与 COMMAND_NAMES 保持一致）
    # 注意：命令0（VOID）在某些情况下可能出现，但不是有效的导航命令
    # 2=Follow, 3=Left, 4=Right, 5=Straight
    VALID_COMMANDS = {2, 3, 4, 5}
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.loader = DataLoader(data_path)
        
        # 报告生成器
        self.verification_report = VerificationReport(data_path)
        self.deletion_report = DeletionReport(data_path)
        self.chart_generator = ChartGenerator(data_path)
    
    def verify_all(self, delete_invalid: bool = False, min_frames: int = 200,
                   generate_charts: bool = True):
        """
        验证所有数据文件
        
        参数:
            delete_invalid: 是否删除不满足条件的文件
            min_frames: 最小帧数要求
            generate_charts: 是否生成可视化图表
        """
        print("\n" + "="*70)
        print("🔍 数据验证工具")
        print("="*70)
        print(f"数据路径: {self.data_path}")
        print(f"最小帧数要求: {min_frames}")
        print(f"模式: {'🗑️ 自动删除无效文件' if delete_invalid else '👁️ 仅预览（不删除）'}\n")
        
        if not os.path.exists(self.data_path):
            print(f"❌ 数据路径不存在: {self.data_path}")
            return
        
        h5_files = self.loader.find_h5_files()
        
        if not h5_files:
            print("❌ 未找到任何H5文件")
            return
        
        print(f"✅ 找到 {len(h5_files)} 个文件\n")
        print("正在验证数据文件...\n")
        
        # 统计数据
        total_frames = 0
        command_stats = defaultdict(int)
        speed_stats = []
        steer_stats = []
        throttle_stats = []
        brake_stats = []
        file_sizes = []
        corrupted_files = []
        warning_files = []
        incomplete_files = []
        
        for idx, filepath in enumerate(h5_files):
            filename = os.path.basename(filepath)
            should_delete = False
            delete_reasons = []
            warnings = []
            
            try:
                with h5py.File(filepath, 'r') as f:
                    # 检查数据集是否存在
                    if 'rgb' not in f or 'targets' not in f:
                        raise ValueError("缺少必要的数据集 'rgb' 或 'targets'")
                    
                    rgb = f['rgb'][:]
                    targets = f['targets'][:]
                    
                    # 验证形状
                    assert rgb.shape[0] == targets.shape[0], "RGB和targets数量不匹配"
                    assert rgb.shape[1:] == (88, 200, 3), f"RGB形状错误: {rgb.shape}"
                    assert targets.shape[1] == 25, f"Targets形状错误: {targets.shape}"
                    
                    num_frames = rgb.shape[0]
                    commands = targets[:, 24]
                    speeds = targets[:, 10]
                    steers = targets[:, 0]
                    throttles = targets[:, 1]
                    brakes = targets[:, 2]
                    file_size = os.path.getsize(filepath) / 1024 / 1024
                    
                    # === 数据质量检查 ===
                    
                    # 图像亮度检查
                    if rgb.mean() < 5:
                        warnings.append("图像过暗")
                        delete_reasons.append("图像过暗(mean<5)")
                        should_delete = True
                    
                    # 速度异常检查
                    if np.max(speeds) > 150:
                        warnings.append(f"速度异常（最大{np.max(speeds):.1f} km/h）")
                    
                    # 方向盘值范围检查
                    if np.min(steers) < -1.1 or np.max(steers) > 1.1:
                        warnings.append(f"方向盘值异常（{np.min(steers):.2f} ~ {np.max(steers):.2f}）")
                    
                    # 油门/刹车值范围检查
                    if np.min(throttles) < -0.1 or np.max(throttles) > 1.1:
                        warnings.append(f"油门值异常（{np.min(throttles):.2f} ~ {np.max(throttles):.2f}）")
                    
                    if np.min(brakes) < -0.1 or np.max(brakes) > 1.1:
                        warnings.append(f"刹车值异常（{np.min(brakes):.2f} ~ {np.max(brakes):.2f}）")
                    
                    # 命令值检查
                    invalid_cmds = set(commands.astype(int)) - self.VALID_COMMANDS
                    if invalid_cmds:
                        warnings.append(f"无效命令值: {invalid_cmds}")
                        delete_reasons.append(f"无效命令值: {invalid_cmds}")
                        should_delete = True
                    
                    # 帧数检查
                    if num_frames < min_frames:
                        incomplete_files.append((filepath, num_frames))
                        delete_reasons.append(f"帧数不足({num_frames}<{min_frames})")
                        should_delete = True
                    
                    # 记录警告
                    if warnings:
                        warning_files.append((filename, warnings))
                        for w in warnings:
                            print(f"  ⚠️  {filename}: {w}")
                    
                    # 统计有效数据
                    if not should_delete:
                        total_frames += num_frames
                        for cmd in np.unique(commands):
                            command_stats[int(cmd)] += int(np.sum(commands == cmd))
                        speed_stats.extend(speeds.tolist())
                        steer_stats.extend(steers.tolist())
                        throttle_stats.extend(throttles.tolist())
                        brake_stats.extend(brakes.tolist())
                        file_sizes.append(file_size)
                
            except Exception as e:
                print(f"  ❌ {filename}: 验证失败 - {e}")
                corrupted_files.append((filepath, str(e)))
                should_delete = True
                delete_reasons.append(f"文件损坏: {e}")
            
            # 处理删除
            if should_delete:
                reason_str = "; ".join(delete_reasons)
                self.deletion_report.add_file(filepath, reason_str)
                
                if delete_invalid:
                    try:
                        os.remove(filepath)
                        print(f"  🗑️  已删除: {filename}")
                    except Exception as e:
                        print(f"  ❌ 删除失败 {filename}: {e}")
            
            # 进度显示
            if (idx + 1) % 10 == 0 or idx == len(h5_files) - 1:
                progress = (idx + 1) / len(h5_files) * 100
                print(f"  进度: {progress:.1f}% ({idx + 1}/{len(h5_files)})")
        
        # 生成验证报告
        report_data = self.verification_report.generate(
            total_frames, command_stats, speed_stats,
            steer_stats, throttle_stats, brake_stats,
            file_sizes, corrupted_files, warning_files,
            incomplete_files, len(h5_files)
        )
        
        # 打印报告
        self.verification_report.print_summary()
        
        # 保存JSON报告
        json_path = self.verification_report.save_json()
        print(f"✅ 验证报告已保存: {json_path}")
        
        # 生成可视化图表
        if generate_charts:
            chart_path = self.chart_generator.generate_charts(report_data)
            if chart_path:
                print(f"✅ 可视化报告已保存: {chart_path}")
        
        # 保存删除报告
        if self.deletion_report.deleted_files:
            self.deletion_report.delete_enabled = delete_invalid
            json_path, txt_path = self.deletion_report.save()
            print(f"✅ 删除报告已保存:")
            print(f"   JSON: {json_path}")
            print(f"   TXT:  {txt_path}")
            self.deletion_report.print_summary()


def main():
    parser = argparse.ArgumentParser(description='验证CARLA收集的数据')
    parser.add_argument('--data-path', required=True, help='数据目录路径')
    parser.add_argument('--delete-invalid', action='store_true', 
                        help='删除不满足条件的文件')
    parser.add_argument('--min-frames', type=int, default=100, 
                        help='最小帧数要求（默认200）')
    parser.add_argument('--no-charts', action='store_true',
                        help='不生成可视化图表')
    
    args = parser.parse_args()
    
    verifier = DataVerifier(args.data_path)
    verifier.verify_all(
        delete_invalid=args.delete_invalid, 
        min_frames=args.min_frames,
        generate_charts=not args.no_charts
    )


if __name__ == '__main__':
    main()
