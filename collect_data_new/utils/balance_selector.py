#!/usr/bin/env python
# coding=utf-8
"""
H5数据平衡选择工具

功能说明:
  1. 好数据判断：H5文件内出现2种或以上不同command（在200帧内）
  2. 平衡比例：follow:left:right:straight = 0.4:0.2:0.2:0.2
  3. 平衡规则：
     - command=2 出现 → follow场景累计+1
     - command=3 出现 → left场景累计+1
     - command=4 出现 → right场景累计+1
     - command=5 出现 → straight场景累计+1
  4. 流程：
     - 分析所有H5文件
     - 筛选好数据（>=2种场景）复制到 good_data 文件夹
     - 在好数据中按比例平衡选择，复制到 good_data/balanced 子文件夹

使用方法:
    # 交互式模式
    python -m collect_data_new.utils.balance_selector
    
    # 命令行模式
    python -m collect_data_new.utils.balance_selector --source E:/data1,E:/data2 --output E:/selected
    
    # 仅分析不复制
    python -m collect_data_new.utils.balance_selector --source E:/data1 --analyze-only
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import h5py

from ..config import COMMAND_NAMES


# ==================== 数据类 ====================

@dataclass
class FileAnalysis:
    """文件分析结果"""
    filepath: str
    filename: str
    source_dir: str = ''
    valid: bool = False
    total_frames: int = 0
    scenes_present: Set[str] = field(default_factory=set)
    num_scenes: int = 0
    is_good_data: bool = False
    command_counts: Dict[str, int] = field(default_factory=dict)
    good_data_path: str = ''
    error: str = ''


@dataclass
class SelectionStats:
    """选择统计"""
    total_analyzed: int = 0
    total_valid: int = 0
    total_good: int = 0
    total_balanced: int = 0
    scene_counts: Dict[str, int] = field(default_factory=dict)


# ==================== 场景分析器 ====================

class SceneAnalyzer:
    """场景分析器"""
    
    # 命令到场景名称的映射
    COMMAND_TO_SCENE = {
        2: 'follow',
        3: 'left',
        4: 'right',
        5: 'straight'
    }
    
    SCENE_NAMES = ['follow', 'left', 'right', 'straight']
    
    def analyze_file(self, filepath: str, max_frames: int = 200) -> FileAnalysis:
        """
        分析单个H5文件
        
        参数:
            filepath: H5文件路径
            max_frames: 分析的最大帧数（默认200帧）
            
        返回:
            FileAnalysis: 分析结果
        """
        result = FileAnalysis(
            filepath=filepath,
            filename=os.path.basename(filepath)
        )
        
        try:
            with h5py.File(filepath, 'r') as f:
                if 'rgb' not in f or 'targets' not in f:
                    return result
                
                targets = f['targets'][:]
                total_frames = targets.shape[0]
                
                if total_frames < 10:
                    return result
                
                result.valid = True
                result.total_frames = total_frames
                
                # 只分析前 max_frames 帧
                analyze_frames = min(max_frames, total_frames)
                commands = targets[:analyze_frames, 24]
                
                # 统计场景出现情况
                scenes_present = set()
                command_counts = {}
                
                for cmd in np.unique(commands):
                    cmd_int = int(cmd)
                    scene_name = self.COMMAND_TO_SCENE.get(cmd_int)
                    if scene_name:
                        scenes_present.add(scene_name)
                        command_counts[scene_name] = int(np.sum(commands == cmd))
                
                result.scenes_present = scenes_present
                result.num_scenes = len(scenes_present)
                result.is_good_data = len(scenes_present) >= 2
                result.command_counts = command_counts
                
        except Exception as e:
            result.error = str(e)
        
        return result


# ==================== 平衡数据选择器 ====================

class BalancedDataSelector:
    """
    平衡数据选择器
    
    工作流程：
    1. 扫描分析所有H5文件
    2. 筛选好数据（>=2种场景）复制到 output/good_data
    3. 从好数据中按比例平衡选择，复制到 output/good_data/balanced
    """
    
    # 默认目标比例
    DEFAULT_RATIOS = {
        'follow': 0.40,
        'left': 0.20,
        'right': 0.20,
        'straight': 0.20,
    }
    
    def __init__(self, source_dirs: List[str], output_dir: str,
                 target_ratios: Optional[Dict[str, float]] = None):
        """
        初始化选择器
        
        参数:
            source_dirs: 源数据目录列表
            output_dir: 输出目录
            target_ratios: 目标场景比例
        """
        self.source_dirs = source_dirs
        self.output_dir = output_dir
        self.target_ratios = target_ratios or self.DEFAULT_RATIOS.copy()
        
        self.analyzer = SceneAnalyzer()
        
        self.all_files: List[FileAnalysis] = []
        self.good_files: List[FileAnalysis] = []
        self.balanced_files: List[FileAnalysis] = []
        
        self.scene_file_counts: Dict[str, int] = defaultdict(int)
        self.stats = SelectionStats()
    
    def scan_and_analyze(self, verbose: bool = True) -> bool:
        """
        扫描并分析所有源目录中的H5文件
        
        参数:
            verbose: 是否显示详细进度
            
        返回:
            bool: 是否找到好数据
        """
        if verbose:
            print("\n" + "="*70)
            print("🔍 第一步：扫描并分析数据文件")
            print("="*70)
            print("\n分析规则：")
            print("  • 在200帧内检查command出现情况")
            print("  • command=2 → follow, command=3 → left")
            print("  • command=4 → right, command=5 → straight")
            print("  • 出现2种或以上不同command → 好数据")
        
        for source_dir in self.source_dirs:
            if verbose:
                print(f"\n📂 扫描目录: {source_dir}")
            
            if not os.path.exists(source_dir):
                if verbose:
                    print(f"  ⚠️ 目录不存在，跳过")
                continue
            
            h5_files = self._find_h5_files(source_dir)
            if verbose:
                print(f"  找到 {len(h5_files)} 个H5文件")
            
            for idx, filepath in enumerate(h5_files):
                if verbose and (idx + 1) % 50 == 0:
                    print(f"  分析进度: {idx + 1}/{len(h5_files)}")
                
                analysis = self.analyzer.analyze_file(filepath)
                analysis.source_dir = source_dir
                self.all_files.append(analysis)
        
        # 统计
        valid_files = [f for f in self.all_files if f.valid]
        self.good_files = [f for f in valid_files if f.is_good_data]
        
        # 计算场景累计
        self.scene_file_counts = defaultdict(int)
        for f in self.good_files:
            for scene in f.scenes_present:
                self.scene_file_counts[scene] += 1
        
        # 更新统计
        self.stats.total_analyzed = len(self.all_files)
        self.stats.total_valid = len(valid_files)
        self.stats.total_good = len(self.good_files)
        self.stats.scene_counts = dict(self.scene_file_counts)
        
        if verbose:
            bad_count = len(valid_files) - len(self.good_files)
            print(f"\n✅ 分析完成:")
            print(f"  • 总文件数: {len(self.all_files)}")
            print(f"  • 有效文件: {len(valid_files)}")
            print(f"  • 好数据（>=2种场景）: {len(self.good_files)}")
            print(f"  • 单场景数据: {bad_count}")
        
        return len(self.good_files) > 0
    
    def _find_h5_files(self, path: str) -> List[str]:
        """递归查找H5文件"""
        h5_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(root, f))
        return h5_files
    
    def print_analysis_report(self):
        """打印分析报告"""
        print("\n" + "="*70)
        print("📊 数据分析报告")
        print("="*70)
        
        total_good = len(self.good_files)
        print(f"\n好数据文件数: {total_good}")
        print(f"\n场景累计统计:")
        print(f"  目标比例: follow=0.4, left=0.2, right=0.2, straight=0.2")
        
        total_scene_count = sum(self.scene_file_counts.values())
        for scene in SceneAnalyzer.SCENE_NAMES:
            count = self.scene_file_counts.get(scene, 0)
            ratio = count / total_scene_count if total_scene_count > 0 else 0
            target = self.target_ratios.get(scene, 0)
            status = "✅" if abs(ratio - target) <= 0.1 else "⚠️"
            print(f"  {status} {scene:10s}: {count:5d} (当前: {ratio*100:5.1f}%, 目标: {target*100:.0f}%)")
        
        # 按场景数量分布
        scene_count_dist: Dict[int, int] = defaultdict(int)
        for f in self.good_files:
            scene_count_dist[f.num_scenes] += 1
        
        print(f"\n按包含场景数量分布:")
        for num in sorted(scene_count_dist.keys()):
            count = scene_count_dist[num]
            print(f"  包含 {num} 种场景: {count} 文件")
    
    def copy_good_data(self, verbose: bool = True) -> bool:
        """
        复制好数据到输出目录
        
        返回:
            bool: 是否成功
        """
        if verbose:
            print("\n" + "="*70)
            print("📦 第二步：复制好数据到输出目录")
            print("="*70)
        
        if not self.good_files:
            if verbose:
                print("❌ 没有好数据可复制")
            return False
        
        good_data_dir = os.path.join(self.output_dir, 'good_data')
        os.makedirs(good_data_dir, exist_ok=True)
        
        if verbose:
            print(f"\n输出目录: {good_data_dir}")
            print(f"待复制文件数: {len(self.good_files)}")
        
        copied_count = 0
        used_names: Set[str] = set()
        
        for idx, f in enumerate(self.good_files):
            src_path = f.filepath
            dst_filename = f.filename
            
            # 处理文件名重复
            if dst_filename in used_names:
                source_name = os.path.basename(f.source_dir)
                base, ext = os.path.splitext(dst_filename)
                dst_filename = f"{base}_{source_name}{ext}"
            
            used_names.add(dst_filename)
            dst_path = os.path.join(good_data_dir, dst_filename)
            f.good_data_path = dst_path
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                if verbose and copied_count % 100 == 0:
                    print(f"  进度: {copied_count}/{len(self.good_files)}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ 复制失败: {src_path} - {e}")
        
        if verbose:
            print(f"\n✅ 好数据复制完成: {copied_count} 个文件")
            print(f"   保存位置: {good_data_dir}")
        
        return True


    def select_balanced_data(self, verbose: bool = True):
        """
        从好数据中平衡选择数据
        
        平衡规则：
        - 每个H5文件包含某场景，则该场景计数+1
        - 按目标比例选择文件，使各场景的文件数接近目标比例
        """
        if verbose:
            print("\n" + "="*70)
            print("⚖️ 第三步：平衡选择数据")
            print("="*70)
        
        if not self.good_files:
            if verbose:
                print("❌ 没有好数据可选择")
            return
        
        # 按场景分组文件
        scene_files: Dict[str, List[FileAnalysis]] = defaultdict(list)
        for f in self.good_files:
            for scene in f.scenes_present:
                scene_files[scene].append(f)
        
        if verbose:
            print(f"\n各场景可用文件数:")
            for scene in SceneAnalyzer.SCENE_NAMES:
                count = len(scene_files.get(scene, []))
                print(f"  {scene:10s}: {count} 文件")
        
        # 计算目标：以最稀缺场景为基准
        min_available = float('inf')
        min_scene = None
        for scene, ratio in self.target_ratios.items():
            if ratio > 0:
                available = len(scene_files.get(scene, []))
                if available > 0:
                    needed_total = available / ratio
                    if needed_total < min_available:
                        min_available = needed_total
                        min_scene = scene
        
        if min_scene is None:
            if verbose:
                print("❌ 没有可用的场景数据")
            return
        
        # 计算各场景目标文件数
        total_target = int(min_available * 0.95)
        scene_targets = {scene: int(total_target * ratio) 
                        for scene, ratio in self.target_ratios.items()}
        
        if verbose:
            print(f"\n最稀缺场景: {min_scene}")
            print(f"目标总文件数: {total_target}")
            print(f"\n各场景目标文件数:")
            for scene, target in scene_targets.items():
                available = len(scene_files.get(scene, []))
                status = "✅" if available >= target else "⚠️"
                print(f"  {scene:10s}: {target:5d} (可用: {available}) {status}")
        
        # 贪心选择算法
        self.balanced_files = []
        selected_set: Set[str] = set()
        scene_selected: Dict[str, int] = defaultdict(int)
        
        # 按稀缺程度排序场景
        scene_priority = sorted(
            self.target_ratios.keys(),
            key=lambda s: len(scene_files.get(s, []))
        )
        
        if verbose:
            print(f"\n选择优先级（从稀缺到丰富）: {scene_priority}")
        
        # 多轮选择
        max_rounds = 10
        for round_num in range(max_rounds):
            made_progress = False
            
            for scene in scene_priority:
                target = scene_targets.get(scene, 0)
                if scene_selected[scene] >= target:
                    continue
                
                # 获取该场景的文件，按包含场景数排序
                available = [f for f in scene_files.get(scene, []) 
                            if f.filepath not in selected_set]
                available = sorted(available, key=lambda f: f.num_scenes, reverse=True)
                
                for f in available:
                    if scene_selected[scene] >= target:
                        break
                    
                    selected_set.add(f.filepath)
                    self.balanced_files.append(f)
                    made_progress = True
                    
                    for s in f.scenes_present:
                        scene_selected[s] += 1
            
            if not made_progress:
                break
        
        self.stats.total_balanced = len(self.balanced_files)
        
        if verbose:
            print(f"\n选择结果:")
            for scene in SceneAnalyzer.SCENE_NAMES:
                selected = scene_selected[scene]
                target = scene_targets[scene]
                diff = selected - target
                status = "✅" if abs(diff) <= target * 0.1 else ("📈" if diff > 0 else "📉")
                print(f"  {status} {scene:10s}: {selected:5d} / {target:5d} (差异: {diff:+d})")
            
            print(f"\n✅ 平衡选择完成: {len(self.balanced_files)} 个文件")
    
    def copy_balanced_data(self, verbose: bool = True):
        """复制平衡后的数据到子文件夹"""
        if verbose:
            print("\n" + "="*70)
            print("📦 第四步：复制平衡数据到子文件夹")
            print("="*70)
        
        if not self.balanced_files:
            if verbose:
                print("❌ 没有平衡数据可复制")
            return
        
        balanced_dir = os.path.join(self.output_dir, 'good_data', 'balanced')
        os.makedirs(balanced_dir, exist_ok=True)
        
        if verbose:
            print(f"\n输出目录: {balanced_dir}")
            print(f"待复制文件数: {len(self.balanced_files)}")
        
        copied_count = 0
        used_names: Set[str] = set()
        
        for f in self.balanced_files:
            # 优先从good_data目录复制
            if f.good_data_path and os.path.exists(f.good_data_path):
                src_path = f.good_data_path
            else:
                src_path = f.filepath
            
            dst_filename = os.path.basename(src_path)
            
            # 处理文件名重复
            if dst_filename in used_names:
                base, ext = os.path.splitext(dst_filename)
                counter = 1
                while f"{base}_{counter}{ext}" in used_names:
                    counter += 1
                dst_filename = f"{base}_{counter}{ext}"
            
            used_names.add(dst_filename)
            dst_path = os.path.join(balanced_dir, dst_filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                if verbose and copied_count % 100 == 0:
                    print(f"  进度: {copied_count}/{len(self.balanced_files)}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ 复制失败: {src_path} - {e}")
        
        if verbose:
            print(f"\n✅ 平衡数据复制完成: {copied_count} 个文件")
            print(f"   保存位置: {balanced_dir}")
        
        # 保存报告
        self._save_report(balanced_dir)
    
    def _save_report(self, output_dir: str):
        """保存选择报告"""
        scene_counts: Dict[str, int] = defaultdict(int)
        for f in self.balanced_files:
            for scene in f.scenes_present:
                scene_counts[scene] += 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'source_dirs': self.source_dirs,
            'output_dir': self.output_dir,
            'target_ratios': self.target_ratios,
            'statistics': {
                'total_analyzed': self.stats.total_analyzed,
                'total_good_data': self.stats.total_good,
                'total_balanced': self.stats.total_balanced,
            },
            'scene_distribution': dict(scene_counts),
            'selected_files': [
                {
                    'filename': f.filename,
                    'source': f.source_dir,
                    'frames': f.total_frames,
                    'scenes': list(f.scenes_present),
                    'num_scenes': f.num_scenes,
                }
                for f in self.balanced_files
            ]
        }
        
        report_path = os.path.join(output_dir, 'balance_report.json')
        with open(report_path, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        
        print(f"📄 报告已保存: {report_path}")
    
    def run_full_pipeline(self, analyze_only: bool = False, verbose: bool = True) -> bool:
        """
        运行完整的数据选择流程
        
        参数:
            analyze_only: 是否仅分析不复制
            verbose: 是否显示详细信息
            
        返回:
            bool: 是否成功
        """
        # 第一步：扫描和分析
        if not self.scan_and_analyze(verbose):
            return False
        
        # 显示分析报告
        if verbose:
            self.print_analysis_report()
        
        if analyze_only:
            if verbose:
                print("\n✅ 分析完成（仅分析模式）")
            return True
        
        # 第二步：复制好数据
        if not self.copy_good_data(verbose):
            return False
        
        # 第三步：平衡选择
        self.select_balanced_data(verbose)
        
        # 第四步：复制平衡数据
        self.copy_balanced_data(verbose)
        
        return True


# ==================== 交互式模式 ====================

def interactive_mode():
    """交互式模式"""
    print("\n" + "="*70)
    print("🎯 H5数据平衡选择工具")
    print("="*70)
    print("\n此工具执行以下步骤:")
    print("  1. 分析所有H5文件，找出包含2种或以上场景的好数据")
    print("  2. 将好数据复制到 output/good_data 目录")
    print("  3. 从好数据中按比例平衡选择")
    print("  4. 将平衡数据复制到 output/good_data/balanced 目录\n")
    
    # 输入源目录
    print("请输入源数据目录（多个目录用逗号分隔）:")
    print("例如: E:/carla_data1, E:/carla_data2")
    source_input = input("> ").strip()
    
    if not source_input:
        print("❌ 未输入源目录")
        return
    
    source_dirs = [d.strip() for d in source_input.split(',')]
    
    # 验证目录
    valid_dirs = []
    for d in source_dirs:
        if os.path.exists(d):
            valid_dirs.append(d)
            print(f"  ✅ {d}")
        else:
            print(f"  ❌ 目录不存在: {d}")
    
    if not valid_dirs:
        print("❌ 没有有效的源目录")
        return
    
    # 输入输出目录
    print("\n请输入输出目录:")
    output_dir = input("> ").strip()
    
    if not output_dir:
        output_dir = "./selected_data"
        print(f"  使用默认目录: {output_dir}")
    
    # 创建选择器并运行
    selector = BalancedDataSelector(valid_dirs, output_dir)
    
    # 第一步：扫描和分析
    if not selector.scan_and_analyze():
        print("❌ 没有找到好数据")
        return
    
    selector.print_analysis_report()
    
    # 确认继续
    print("\n是否继续复制好数据？")
    choice = input("继续? (y/n, 默认y): ").strip().lower()
    if choice == 'n':
        print("已取消")
        return
    
    # 第二步：复制好数据
    if not selector.copy_good_data():
        return
    
    # 第三步：平衡选择
    selector.select_balanced_data()
    
    # 确认复制平衡数据
    print("\n是否复制平衡后的数据到子文件夹？")
    choice = input("继续? (y/n, 默认y): ").strip().lower()
    if choice == 'n':
        print("已取消")
        return
    
    # 第四步：复制平衡数据
    selector.copy_balanced_data()
    
    print("\n" + "="*70)
    print("✅ 全部完成！")
    print("="*70)
    print(f"  好数据目录: {os.path.join(output_dir, 'good_data')}")
    print(f"  平衡数据目录: {os.path.join(output_dir, 'good_data', 'balanced')}")


# ==================== 命令行入口 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='H5数据平衡选择工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 交互式模式
  python -m collect_data_new.utils.balance_selector
  
  # 命令行模式
  python -m collect_data_new.utils.balance_selector --source E:/data1,E:/data2 --output E:/selected
  
  # 仅分析
  python -m collect_data_new.utils.balance_selector --source E:/data1 --analyze-only
        '''
    )
    
    parser.add_argument('--source', type=str, 
                       help='源数据目录，多个用逗号分隔')
    parser.add_argument('--output', type=str, default='./selected_data',
                       help='输出目录 (默认: ./selected_data)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析，不复制文件')
    
    # 自定义比例参数
    parser.add_argument('--follow', type=float, default=0.40,
                       help='Follow场景目标比例 (默认: 0.40)')
    parser.add_argument('--left', type=float, default=0.20,
                       help='Left场景目标比例 (默认: 0.20)')
    parser.add_argument('--right', type=float, default=0.20,
                       help='Right场景目标比例 (默认: 0.20)')
    parser.add_argument('--straight', type=float, default=0.20,
                       help='Straight场景目标比例 (默认: 0.20)')
    
    args = parser.parse_args()
    
    # 如果没有提供源目录，进入交互模式
    if not args.source:
        interactive_mode()
        return
    
    # 命令行模式
    source_dirs = [d.strip() for d in args.source.split(',')]
    
    # 验证目录
    valid_dirs = []
    for d in source_dirs:
        if os.path.exists(d):
            valid_dirs.append(d)
        else:
            print(f"⚠️ 目录不存在，跳过: {d}")
    
    if not valid_dirs:
        print("❌ 没有有效的源目录")
        return
    
    # 构建目标比例
    target_ratios = {
        'follow': args.follow,
        'left': args.left,
        'right': args.right,
        'straight': args.straight,
    }
    
    # 创建选择器并运行
    selector = BalancedDataSelector(valid_dirs, args.output, target_ratios)
    selector.run_full_pipeline(analyze_only=args.analyze_only)
    
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
