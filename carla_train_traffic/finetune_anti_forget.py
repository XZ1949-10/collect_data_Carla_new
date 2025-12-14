#!/usr/bin/env python
# coding=utf-8
'''
防遗忘微调脚本
支持三种防遗忘策略:
1. EWC (Elastic Weight Consolidation) - 弹性权重巩固
2. 混合数据训练 - 新旧数据按比例混合
3. 知识蒸馏 - 用旧模型输出作为软标签

使用方法:
    # 方式1: 仅使用新数据 + EWC防遗忘
    python finetune_anti_forget.py \
        --pretrained /path/to/best_model.pth \
        --new-train-dir /path/to/traffic_light/train \
        --new-eval-dir /path/to/traffic_light/val \
        --ewc-lambda 5000

    # 方式2: 混合新旧数据训练 (推荐)
    python finetune_anti_forget.py \
        --pretrained /path/to/best_model.pth \
        --old-train-dir /path/to/original/train \
        --old-eval-dir /path/to/original/val \
        --new-train-dir /path/to/traffic_light/train \
        --new-eval-dir /path/to/traffic_light/val \
        --mix-ratio 0.3 \
        --use-mixed-data

    # 方式3: 知识蒸馏
    python finetune_anti_forget.py \
        --pretrained /path/to/best_model.pth \
        --new-train-dir /path/to/traffic_light/train \
        --new-eval-dir /path/to/traffic_light/val \
        --use-distillation \
        --distill-alpha 0.5

    # 组合使用 (最强防遗忘)
    python finetune_anti_forget.py \
        --pretrained /path/to/best_model.pth \
        --old-train-dir /path/to/original/train \
        --old-eval-dir /path/to/original/val \
        --new-train-dir /path/to/traffic_light/train \
        --new-eval-dir /path/to/traffic_light/val \
        --use-mixed-data --mix-ratio 0.3 \
        --use-distillation --distill-alpha 0.3 \
        --ewc-lambda 1000

分布式训练:
    torchrun --nproc_per_node=6 finetune_anti_forget.py [参数...]
'''
import argparse
import os
import copy
import random
import time
import datetime
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from carla_net_ori import FinalNet
from carla_loader_dynamic import CarlaH5DataDDP
from carla_loader_mixed import MixedDataLoader
from helper import AverageMeter, save_checkpoint


parser = argparse.ArgumentParser(description='Anti-Forgetting Fine-tuning')

# 模型参数
parser.add_argument('--pretrained', required=True, type=str,
                    help='预训练模型路径')
parser.add_argument('--net-structure', default=1, type=int,
                    help='网络结构 1|2|3')

# 数据参数
parser.add_argument('--old-train-dir', default='', type=str,
                    help='旧数据训练集路径 (用于混合训练)')
parser.add_argument('--old-eval-dir', default='', type=str,
                    help='旧数据验证集路径')
parser.add_argument('--new-train-dir', required=True, type=str,
                    help='新数据(红绿灯)训练集路径')
parser.add_argument('--new-eval-dir', required=True, type=str,
                    help='新数据验证集路径')
parser.add_argument('--min-frames', default=10, type=int,
                    help='每个h5文件最小帧数')

# 混合数据参数
parser.add_argument('--use-mixed-data', action='store_true', default=False,
                    help='使用新旧数据混合训练')
parser.add_argument('--mix-ratio', default=0.5, type=float,
                    help='新数据占比 (0.3 = 新数据30%, 旧数据70%)')
parser.add_argument('--mix-mode', default='balanced', type=str,
                    choices=['concat', 'balanced'],
                    help='混合模式: concat=简单拼接, balanced=平衡采样')

# EWC参数
parser.add_argument('--ewc-lambda', default=0, type=float,
                    help='EWC正则化强度 (0=禁用, 推荐1000-10000)')
parser.add_argument('--ewc-samples', default=2000, type=int,
                    help='计算Fisher信息矩阵的样本数')

# 知识蒸馏参数
parser.add_argument('--use-distillation', action='store_true', default=False,
                    help='使用知识蒸馏')
parser.add_argument('--distill-alpha', default=0.5, type=float,
                    help='蒸馏损失权重 (0-1, 越大越保守)')
parser.add_argument('--distill-temperature', default=2.0, type=float,
                    help='蒸馏温度 (越高越软)')

# 训练参数
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='数据加载线程数')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='总batch size')
parser.add_argument('--epochs', default=30, type=int,
                    help='训练轮数')
parser.add_argument('--lr', default=5e-5, type=float,
                    help='学习率 (微调应该比预训练小)')
parser.add_argument('--speed-weight', default=0.5, type=float,
                    help='速度损失权重')
parser.add_argument('--branch-weight', default=1.5, type=float,
                    help='分支损失权重')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='权重衰减')

# 学习率调度
parser.add_argument('--lr-patience', default=3, type=int,
                    help='学习率调度耐心值')
parser.add_argument('--lr-factor', default=0.5, type=float,
                    help='学习率衰减因子')
parser.add_argument('--min-lr', default=1e-7, type=float,
                    help='最小学习率')

# 早停
parser.add_argument('--early-stop', action='store_true', default=True,
                    help='启用早停')
parser.add_argument('--patience', default=8, type=int,
                    help='早停耐心值')

# 其他
parser.add_argument('--id', default='finetune_traffic_light', type=str,
                    help='实验ID')
parser.add_argument('--print-freq', default=10, type=int,
                    help='打印频率')
parser.add_argument('--seed', default=42, type=int,
                    help='随机种子')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='使用混合精度')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='使用channels_last内存格式')


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class EWC:
    """
    Elastic Weight Consolidation (弹性权重巩固)
    
    核心思想: 在微调时，对重要参数施加约束，防止其偏离太远
    重要性通过Fisher信息矩阵估计
    """
    def __init__(self, model, dataloader, device, num_samples=2000):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader, num_samples)
    
    def _compute_fisher(self, dataloader, num_samples):
        """计算Fisher信息矩阵 (对角近似)"""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.model.eval()
        samples_seen = 0
        
        for img, speed, target, mask in dataloader:
            if samples_seen >= num_samples:
                break
            
            img = img.to(self.device)
            speed = speed.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)
            
            self.model.zero_grad()
            
            output = self.model(img, speed)
            if isinstance(output, tuple) and len(output) == 4:
                branches_out, pred_speed, _, _ = output
            else:
                branches_out, pred_speed = output
            
            # 使用输出的log概率作为损失
            loss = F.mse_loss(branches_out * mask, target) + F.mse_loss(pred_speed, speed)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            
            samples_seen += img.size(0)
        
        # 归一化
        for n in fisher:
            fisher[n] /= samples_seen
        
        return fisher
    
    def penalty(self, model):
        """计算EWC惩罚项"""
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss


class KnowledgeDistillation:
    """
    知识蒸馏
    
    使用旧模型的输出作为软标签，引导新模型学习
    """
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5):
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
    
    def distill_loss(self, student_output, img, speed, device):
        """计算蒸馏损失"""
        with torch.no_grad():
            teacher_output = self.teacher(img, speed)
            if isinstance(teacher_output, tuple) and len(teacher_output) == 4:
                teacher_control, teacher_speed, _, _ = teacher_output
            else:
                teacher_control, teacher_speed = teacher_output
        
        if isinstance(student_output, tuple) and len(student_output) == 4:
            student_control, student_speed, _, _ = student_output
        else:
            student_control, student_speed = student_output
        
        # 软标签损失 (MSE for regression)
        control_distill = F.mse_loss(student_control, teacher_control)
        speed_distill = F.mse_loss(student_speed, teacher_speed)
        
        return control_distill + speed_distill


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        dist.barrier()
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def output_log(msg, logger=None, rank=0):
    if rank == 0:
        print(f"[{datetime.datetime.now()}]: {msg}")
        if logger:
            logger.critical(f"[{datetime.datetime.now()}]: {msg}")


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def load_pretrained_weights(model, checkpoint_path, rank=0):
    """加载预训练权重"""
    output_log(f"加载预训练模型: {checkpoint_path}", rank=rank)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 处理DDP前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    if 'epoch' in checkpoint:
        output_log(f"预训练模型训练了 {checkpoint['epoch']} 轮", rank=rank)
    if 'best_prec' in checkpoint:
        output_log(f"预训练模型最佳loss: {checkpoint['best_prec']:.4f}", rank=rank)
    
    return model


def main():
    args = parser.parse_args()
    
    # 初始化分布式
    distributed, rank, world_size, local_rank = setup_distributed()
    args.distributed = distributed
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank if distributed else 0
    
    # 创建目录
    log_dir = os.path.join("./logs", args.id)
    run_dir = os.path.join("./runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    
    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_weight_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, "finetune.log"),
                            level=logging.ERROR)
        tsbd = SummaryWriter(log_dir=run_dir)
        
        # 打印配置
        print("\n" + "="*70)
        print("🚦 红绿灯场景防遗忘微调")
        print("="*70)
        print(f"📁 预训练模型: {args.pretrained}")
        print(f"📁 新数据训练集: {args.new_train_dir}")
        print(f"📁 新数据验证集: {args.new_eval_dir}")
        
        if args.use_mixed_data:
            print(f"\n🔀 混合数据训练:")
            print(f"   旧数据训练集: {args.old_train_dir}")
            print(f"   新数据占比: {args.mix_ratio*100:.0f}%")
            print(f"   混合模式: {args.mix_mode}")
        
        if args.ewc_lambda > 0:
            print(f"\n🛡️ EWC防遗忘:")
            print(f"   Lambda: {args.ewc_lambda}")
            print(f"   采样数: {args.ewc_samples}")
        
        if args.use_distillation:
            print(f"\n📚 知识蒸馏:")
            print(f"   Alpha: {args.distill_alpha}")
            print(f"   Temperature: {args.distill_temperature}")
        
        print(f"\n⚙️ 训练参数:")
        print(f"   学习率: {args.lr}")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Epochs: {args.epochs}")
        print("="*70 + "\n")
    else:
        tsbd = None
        logging.basicConfig(level=logging.ERROR)
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        cudnn.deterministic = True
    
    # 创建模型
    model = FinalNet(args.net_structure)
    model = load_pretrained_weights(model, args.pretrained, rank)
    
    # 创建教师模型 (用于知识蒸馏)
    teacher_model = None
    distiller = None
    if args.use_distillation:
        output_log("创建教师模型用于知识蒸馏...", rank=rank)
        teacher_model = FinalNet(args.net_structure)
        teacher_model = load_pretrained_weights(teacher_model, args.pretrained, rank)
        teacher_model = teacher_model.cuda(args.local_rank)
        teacher_model.eval()
        distiller = KnowledgeDistillation(
            teacher_model, 
            temperature=args.distill_temperature,
            alpha=args.distill_alpha)
    
    model = model.cuda(args.local_rank)
    
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        if teacher_model:
            teacher_model = teacher_model.to(memory_format=torch.channels_last)
    
    # DDP包装
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    
    criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.7, 0.85),
                           weight_decay=args.weight_decay)
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
        verbose=is_main_process(rank))
    
    # 早停
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stop else None
    
    # 混合精度
    scaler = GradScaler() if args.use_amp else None
    
    best_prec = math.inf
    cudnn.benchmark = True
    
    # 数据加载
    batch_size_per_gpu = args.batch_size // world_size
    
    if args.use_mixed_data and args.old_train_dir:
        # 混合数据加载
        output_log("使用混合数据加载器...", rank=rank)
        carla_data = MixedDataLoader(
            old_train_folder=args.old_train_dir,
            old_eval_folder=args.old_eval_dir,
            new_train_folder=args.new_train_dir,
            new_eval_folder=args.new_eval_dir,
            batch_size=batch_size_per_gpu,
            num_workers=args.workers,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            mix_ratio=args.mix_ratio,
            mix_mode=args.mix_mode,
            min_frames=args.min_frames)
    else:
        # 仅新数据
        output_log("仅使用新数据训练...", rank=rank)
        carla_data = CarlaH5DataDDP(
            train_folder=args.new_train_dir,
            eval_folder=args.new_eval_dir,
            batch_size=batch_size_per_gpu,
            num_workers=args.workers,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            min_frames=args.min_frames)
    
    train_loader = carla_data.loaders["train"]
    train_sampler = carla_data.samplers["train"]
    eval_loader = carla_data.loaders["eval"]
    
    # EWC初始化 (在微调前计算Fisher矩阵，记录参数重要性)
    ewc = None
    if args.ewc_lambda > 0:
        output_log(f"计算EWC Fisher信息矩阵 (采样{args.ewc_samples}个样本)...", rank=rank)
        
        # 选择用于计算Fisher的数据
        if args.old_train_dir:
            # 优先使用旧数据计算Fisher (最佳选择)
            output_log("使用旧数据计算Fisher矩阵", rank=rank)
            ewc_loader = CarlaH5DataDDP(
                train_folder=args.old_train_dir,
                eval_folder=args.old_eval_dir,
                batch_size=batch_size_per_gpu,
                num_workers=args.workers,
                distributed=False,  # 单卡计算
                world_size=1,
                rank=0,
                min_frames=args.min_frames).loaders["train"]
        else:
            # 没有旧数据时，使用新数据计算Fisher
            # 这会保护模型在新数据上的初始表现，防止过度拟合
            output_log("⚠️ 无旧数据，使用新数据计算Fisher矩阵", rank=rank)
            ewc_loader = CarlaH5DataDDP(
                train_folder=args.new_train_dir,
                eval_folder=args.new_eval_dir,
                batch_size=batch_size_per_gpu,
                num_workers=args.workers,
                distributed=False,
                world_size=1,
                rank=0,
                min_frames=args.min_frames).loaders["train"]
        
        # 获取原始模型 (去掉DDP包装)
        raw_model = model.module if distributed else model
        ewc = EWC(raw_model, ewc_loader, args.local_rank, args.ewc_samples)
        output_log("EWC初始化完成", rank=rank)
    
    # 训练循环
    for epoch in range(args.epochs):
        if distributed and train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch,
            tsbd, scaler, args, ewc=ewc, distiller=distiller)
        
        eval_loss = evaluate(eval_loader, model, criterion, epoch, tsbd, args)
        
        lr_scheduler.step(eval_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process(rank):
            output_log(f"Epoch {epoch+1} - Train: {train_loss:.4f}, "
                      f"Eval: {eval_loss:.4f}, LR: {current_lr:.2e}", rank=rank)
            if tsbd:
                tsbd.add_scalar('finetune/learning_rate', current_lr, epoch + 1)
        
        # 保存模型
        if is_main_process(rank):
            is_best = eval_loss < best_prec
            best_prec = min(eval_loss, best_prec)
            save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'best_prec': best_prec,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': lr_scheduler.state_dict()},
                args.id, is_best,
                os.path.join(save_weight_dir, f"epoch_{epoch+1}.pth"))
        
        # 早停
        if early_stopper and early_stopper(eval_loss):
            output_log(f"早停触发于 epoch {epoch+1}!", rank=rank)
            break
        
        if distributed:
            dist.barrier()
    
    if is_main_process(rank):
        print("\n" + "="*70)
        print(f"✅ 微调完成! 最佳验证loss: {best_prec:.4f}")
        print(f"📁 最佳模型保存至: save_models/{args.id}_best.pth")
        print("="*70)
    
    cleanup_distributed()


def train(loader, model, criterion, optimizer, epoch, writer, scaler, args,
          ewc=None, distiller=None):
    """训练一个epoch"""
    losses = AverageMeter()
    ewc_losses = AverageMeter()
    distill_losses = AverageMeter()
    
    model.train()
    
    for i, (img, speed, target, mask) in enumerate(loader):
        img = img.cuda(args.local_rank, non_blocking=True)
        speed = speed.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        mask = mask.cuda(args.local_rank, non_blocking=True)
        
        if args.channels_last:
            img = img.to(memory_format=torch.channels_last)
        
        optimizer.zero_grad()
        
        with autocast(enabled=args.use_amp):
            output = model(img, speed)
            if isinstance(output, tuple) and len(output) == 4:
                branches_out, pred_speed, log_var_control, log_var_speed = output
                branch_loss = torch.mean((torch.exp(-log_var_control)
                                          * torch.pow((branches_out - target), 2)
                                          + log_var_control) * 0.5 * mask) * 4
                speed_loss = torch.mean((torch.exp(-log_var_speed)
                                         * torch.pow((pred_speed - speed), 2)
                                         + log_var_speed) * 0.5)
            else:
                branches_out, pred_speed = output
                branch_loss = criterion(branches_out * mask, target) * 4
                speed_loss = criterion(pred_speed, speed)
            
            # 基础损失
            task_loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss
            total_loss = task_loss
            
            # EWC损失
            ewc_loss_val = 0
            if ewc is not None and args.ewc_lambda > 0:
                raw_model = model.module if args.distributed else model
                ewc_loss_val = ewc.penalty(raw_model)
                total_loss = total_loss + args.ewc_lambda * ewc_loss_val
            
            # 蒸馏损失
            distill_loss_val = 0
            if distiller is not None:
                distill_loss_val = distiller.distill_loss(output, img, speed, args.local_rank)
                total_loss = (1 - distiller.alpha) * total_loss + distiller.alpha * distill_loss_val
        
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        # 记录损失
        if args.distributed:
            reduced_loss = reduce_tensor(task_loss.data, args.world_size)
        else:
            reduced_loss = task_loss.data
        
        losses.update(reduced_loss.item(), img.size(0))
        if ewc is not None:
            ewc_losses.update(ewc_loss_val.item() if isinstance(ewc_loss_val, torch.Tensor) else ewc_loss_val, img.size(0))
        if distiller is not None:
            distill_losses.update(distill_loss_val.item() if isinstance(distill_loss_val, torch.Tensor) else distill_loss_val, img.size(0))
        
        if i % args.print_freq == 0 and is_main_process(args.rank):
            extra_info = ""
            if ewc is not None:
                extra_info += f" EWC:{ewc_losses.val:.4f}"
            if distiller is not None:
                extra_info += f" Distill:{distill_losses.val:.4f}"
            output_log(f'Epoch [{epoch+1}][{i}/{len(loader)}] '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}){extra_info}', rank=args.rank)
    
    return losses.avg


def evaluate(loader, model, criterion, epoch, writer, args):
    """验证"""
    losses = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        for img, speed, target, mask in loader:
            img = img.cuda(args.local_rank, non_blocking=True)
            speed = speed.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            mask = mask.cuda(args.local_rank, non_blocking=True)
            
            output = model(img, speed)
            if isinstance(output, tuple) and len(output) == 4:
                branches_out, pred_speed, log_var_control, log_var_speed = output
                branch_loss = torch.mean((torch.exp(-log_var_control)
                                          * torch.pow((branches_out - target), 2)
                                          + log_var_control) * 0.5 * mask) * 4
                speed_loss = torch.mean((torch.exp(-log_var_speed)
                                         * torch.pow((pred_speed - speed), 2)
                                         + log_var_speed) * 0.5)
            else:
                branches_out, pred_speed = output
                branch_loss = criterion(branches_out * mask, target) * 4
                speed_loss = criterion(pred_speed, speed)
            
            loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data
            
            losses.update(reduced_loss.item(), img.size(0))
    
    if is_main_process(args.rank) and writer:
        writer.add_scalar('finetune/eval_loss', losses.avg, epoch + 1)
    
    return losses.avg


if __name__ == '__main__':
    main()
