#!/usr/bin/env python
# coding=utf-8
'''
多卡分布式训练版本 (DistributedDataParallel)
针对6张P100优化

启动命令:
    torchrun --nproc_per_node=6 main_ddp.py --batch-size 168

或使用launch:
    python -m torch.distributed.launch --nproc_per_node=6 main_ddp.py --batch-size 168
'''
import argparse
import os
import random
import time
import datetime
import math
import logging
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from carla_net_ori import CarlaNet, FinalNet
from carla_loader_ddp import CarlaH5DataDDP
from helper import AverageMeter, save_checkpoint

# PyTorch 2.0+ torch.compile 支持检测
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and torch.__version__ >= '2.0'


parser = argparse.ArgumentParser(description='Carla CIL DDP Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers per GPU (default: 8)')
parser.add_argument('--speed-weight', default=0.5, type=float,
                    help='speed weight')
parser.add_argument('--branch-weight', default=1.5, type=float,
                    help='branch weight')
parser.add_argument('--id', default="ddp_demo_bandata", type=str)
parser.add_argument('--train-dir',
                    default=r"/root/data1/carla_cil_pytorch/AgentHuman/SeqTrain",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--eval-dir',
                    default=r"/root/data1/carla_cil_pytorch/AgentHuman/SeqVal",
                    type=str, metavar='PATH',
                    help='evaluation dataset')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=600, type=int,
                    metavar='N', help='total batch size across all GPUs (default: 600, 100 per GPU)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-step', default=10, type=int,
                    help='learning rate step size')
parser.add_argument('--lr-gamma', default=0.5, type=float,
                    help='learning rate gamma')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--net-structure', default=1, type=int,
                    help='Network structure 1|2|3')
# DDP 参数
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use automatic mixed precision (P100不支持Tensor Core，AMP无加速效果，仅省显存)')
parser.add_argument('--sync-bn', action='store_true', default=False,
                    help='use synchronized batch normalization')
parser.add_argument('--gradient-accumulation', default=1, type=int,
                    help='gradient accumulation steps')
# 进阶优化参数
parser.add_argument('--use-compile', action='store_true', default=False,
                    help='use torch.compile for optimization (PyTorch 2.0+)')
parser.add_argument('--bucket-cap-mb', default=25, type=int,
                    help='DDP bucket size in MB for gradient sync optimization')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='use channels_last memory format for better GPU utilization')
parser.add_argument('--pin-memory-device', default='', type=str,
                    help='pin memory to specific CUDA device')
# 早停和学习率自动调节参数
parser.add_argument('--early-stop', action='store_true', default=True,
                    help='enable early stopping')
parser.add_argument('--patience', default=10, type=int,
                    help='early stopping patience (epochs without improvement)')
parser.add_argument('--min-delta', default=1e-4, type=float,
                    help='minimum change to qualify as improvement')
parser.add_argument('--auto-lr', action='store_true', default=True,
                    help='enable automatic learning rate scheduling (ReduceLROnPlateau)')
parser.add_argument('--lr-patience', default=5, type=int,
                    help='patience for ReduceLROnPlateau scheduler')
parser.add_argument('--lr-factor', default=0.5, type=float,
                    help='factor for ReduceLROnPlateau scheduler')
parser.add_argument('--min-lr', default=1e-7, type=float,
                    help='minimum learning rate')
parser.add_argument('--lr-finder', action='store_true', default=False,
                    help='run learning rate finder before training')
parser.add_argument('--lr-finder-steps', default=100, type=int,
                    help='number of steps for learning rate finder')


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']


def find_optimal_lr(model, train_loader, criterion, args, start_lr=1e-7, end_lr=1, num_steps=100):
    """
    学习率查找器 - 使用 LR Range Test 方法自动找到最优学习率
    返回建议的学习率 (loss下降最快的点)
    """
    import copy
    
    # 保存原始模型状态
    original_state = copy.deepcopy(model.state_dict())
    
    # 创建临时优化器
    if args.net_structure != 1:
        params = model.module.uncertain_net.parameters() if args.distributed else model.uncertain_net.parameters()
    else:
        params = model.parameters()
    
    optimizer = optim.Adam(params, lr=start_lr, betas=(0.7, 0.85))
    
    # 计算学习率增长因子
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    model.train()
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            img, speed, target, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img, speed, target, mask = next(data_iter)
        
        img = img.cuda(args.local_rank, non_blocking=True)
        speed = speed.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        mask = mask.cuda(args.local_rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        if args.net_structure != 1:
            branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)
            branch_loss = torch.mean((torch.exp(-log_var_control)
                                      * torch.pow((branches_out - target), 2)
                                      + log_var_control) * 0.5 * mask) * 4
            speed_loss = torch.mean((torch.exp(-log_var_speed)
                                     * torch.pow((pred_speed - speed), 2)
                                     + log_var_speed) * 0.5)
        else:
            branches_out, pred_speed = model(img, speed)
            branch_loss = criterion(branches_out * mask, target) * 4
            speed_loss = criterion(pred_speed, speed)
        
        loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss
        
        # 如果loss爆炸，停止搜索
        if loss.item() > best_loss * 4:
            break
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    # 恢复原始模型状态
    model.load_state_dict(original_state)
    
    # 找到loss下降最快的点 (使用平滑后的梯度)
    if len(losses) < 10:
        return start_lr * 10  # 默认值
    
    # 平滑losses
    smoothed_losses = []
    beta = 0.98
    avg_loss = 0
    for i, loss in enumerate(losses):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (i + 1)))
    
    # 找到梯度最陡的点
    min_grad_idx = 0
    min_grad = float('inf')
    for i in range(1, len(smoothed_losses) - 1):
        grad = smoothed_losses[i] - smoothed_losses[i - 1]
        if grad < min_grad:
            min_grad = grad
            min_grad_idx = i
    
    # 返回最优学习率的1/10作为安全起点
    optimal_lr = lrs[min_grad_idx] / 10
    return max(min(optimal_lr, end_lr / 10), start_lr * 10)


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def output_log(output_str, logger=None, rank=0):
    """只在主进程输出日志"""
    if rank == 0:
        print("[{}]: {}".format(datetime.datetime.now(), output_str))
        if logger is not None:
            logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def reduce_tensor(tensor, world_size):
    """跨GPU聚合tensor"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def main():
    global args
    args = parser.parse_args()
    
    # 初始化分布式
    distributed, rank, world_size, local_rank = setup_distributed()
    args.distributed = distributed
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    
    # 只在主进程创建目录和日志
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    
    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_weight_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "carla_training_ddp.log"),
            level=logging.ERROR)
        tsbd = SummaryWriter(log_dir=run_dir)
        output_log(f"Using {world_size} GPUs for training", logging, rank)
        output_log(f"Total batch size: {args.batch_size}, per GPU: {args.batch_size // world_size}", logging, rank)
    else:
        tsbd = None
        logging.basicConfig(level=logging.ERROR)
    
    if args.seed is not None:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        cudnn.deterministic = True
    
    # 创建模型
    model = FinalNet(args.net_structure)
    
    # 同步BN (可选，对于小batch有帮助)
    if args.sync_bn and distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        output_log("Using Synchronized BatchNorm", logging, rank)
    
    model = model.cuda(local_rank)
    
    # channels_last 内存格式优化 (对CNN有帮助)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        output_log("Using channels_last memory format", logging, rank)
    
    # 包装为DDP，添加通信优化
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # 减少显存拷贝
            bucket_cap_mb=args.bucket_cap_mb  # 优化梯度同步
        )
    
    # torch.compile 优化 (PyTorch 2.0+)
    if args.use_compile and TORCH_COMPILE_AVAILABLE:
        output_log("Using torch.compile() for optimization", logging, rank)
        model = torch.compile(model, mode='reduce-overhead')
    elif args.use_compile and not TORCH_COMPILE_AVAILABLE:
        output_log("torch.compile not available (requires PyTorch 2.0+)", logging, rank)
    
    criterion = nn.MSELoss()
    
    # 优化器
    if args.net_structure != 1:
        optimizer = optim.Adam(
            model.module.uncertain_net.parameters() if distributed else model.uncertain_net.parameters(),
            args.lr, betas=(0.7, 0.85))
    else:
        optimizer = optim.Adam(model.parameters(), args.lr, betas=(0.7, 0.85))
    
    # 学习率调度器选择
    if args.auto_lr:
        # 使用 ReduceLROnPlateau 自动调节学习率
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=is_main_process(rank))
        output_log(f"Using ReduceLROnPlateau scheduler (patience={args.lr_patience}, factor={args.lr_factor})", logging, rank)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # 早停机制
    early_stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='min') if args.early_stop else None
    if args.early_stop:
        output_log(f"Early stopping enabled (patience={args.patience}, min_delta={args.min_delta})", logging, rank)
    
    # 混合精度
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp and is_main_process(rank):
        output_log("Using Automatic Mixed Precision (AMP)", logging, rank)
    
    best_prec = math.inf
    
    # 恢复训练
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log(f"=> loading checkpoint '{args.resume}'", logging, rank)
            map_location = {'cuda:0': f'cuda:{local_rank}'}
            checkpoint = torch.load(args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            best_prec = checkpoint['best_prec']
            # 恢复早停状态
            if early_stopper is not None and 'early_stopper' in checkpoint:
                early_stopper.load_state_dict(checkpoint['early_stopper'])
            output_log(f"=> loaded checkpoint (epoch {checkpoint['epoch']})", logging, rank)
    
    # cuDNN 优化
    cudnn.benchmark = True
    cudnn.enabled = True
    # 对于固定输入尺寸，可以进一步优化
    if hasattr(cudnn, 'allow_tf32'):
        cudnn.allow_tf32 = False  # P100不支持TF32，关闭避免警告
    
    # 数据加载 (使用DDP版本)
    batch_size_per_gpu = args.batch_size // world_size
    carla_data = CarlaH5DataDDP(
        train_folder=args.train_dir,
        eval_folder=args.eval_dir,
        batch_size=batch_size_per_gpu,
        num_workers=args.workers,
        distributed=distributed,
        world_size=world_size,
        rank=rank)
    
    train_loader = carla_data.loaders["train"]
    train_sampler = carla_data.samplers["train"]
    eval_loader = carla_data.loaders["eval"]
    
    # 学习率自动初始化 (LR Finder)
    if args.lr_finder and not args.resume:
        output_log("Running Learning Rate Finder...", logging, rank)
        optimal_lr = find_optimal_lr(
            model, train_loader, criterion, args,
            start_lr=1e-7, end_lr=1, num_steps=args.lr_finder_steps)
        output_log(f"Optimal learning rate found: {optimal_lr:.2e}", logging, rank)
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = optimal_lr
        args.lr = optimal_lr
    
    if args.evaluate:
        evaluate(eval_loader, model, criterion, 0, tsbd, args)
        cleanup_distributed()
        return
    
    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 设置epoch以确保shuffle正确
        if distributed:
            train_sampler.set_epoch(epoch)
        
        train(train_loader, model, criterion, optimizer, epoch, tsbd, scaler, args)
        prec = evaluate(eval_loader, model, criterion, epoch, tsbd, args)
        
        # 学习率调度
        if args.auto_lr:
            lr_scheduler.step(prec)  # ReduceLROnPlateau 需要传入验证指标
        else:
            lr_scheduler.step()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process(rank):
            output_log(f"Current learning rate: {current_lr:.2e}", logging, rank)
            if tsbd is not None:
                tsbd.add_scalar('train/learning_rate', current_lr, epoch + 1)
        
        # 只在主进程保存
        if is_main_process(rank):
            is_best = prec < best_prec
            best_prec = min(prec, best_prec)
            checkpoint_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            # 保存早停状态
            if early_stopper is not None:
                checkpoint_dict['early_stopper'] = early_stopper.state_dict()
            
            save_checkpoint(
                checkpoint_dict,
                args.id,
                is_best,
                os.path.join(save_weight_dir, f"{epoch+1}_{args.id}.pth"))
        
        # 早停检查
        if early_stopper is not None:
            should_stop = early_stopper(prec)
            if should_stop:
                output_log(f"Early stopping triggered at epoch {epoch + 1}! "
                          f"No improvement for {args.patience} epochs.", logging, rank)
                break
        
        if distributed:
            dist.barrier()
    
    if is_main_process(rank):
        output_log(f"Training completed. Best validation loss: {best_prec:.4f}", logging, rank)
    
    cleanup_distributed()


def train(loader, model, criterion, optimizer, epoch, writer, scaler, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    uncertain_losses = AverageMeter()
    branch_losses = AverageMeter()
    speed_losses = AverageMeter()
    
    model.train()
    end = time.time()
    step = epoch * len(loader)
    
    optimizer.zero_grad()
    
    for i, (img, speed, target, mask) in enumerate(loader):
        data_time.update(time.time() - end)
        
        img = img.cuda(args.local_rank, non_blocking=True)
        speed = speed.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        mask = mask.cuda(args.local_rank, non_blocking=True)
        
        # channels_last 格式转换
        if args.channels_last:
            img = img.to(memory_format=torch.channels_last)
        
        # 混合精度前向
        with autocast(enabled=args.use_amp):
            if args.net_structure != 1:
                branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)
                branch_square = torch.pow((branches_out - target), 2)
                branch_loss = torch.mean((torch.exp(-log_var_control)
                                          * branch_square
                                          + log_var_control) * 0.5 * mask) * 4
                speed_square = torch.pow((pred_speed - speed), 2)
                speed_loss = torch.mean((torch.exp(-log_var_speed)
                                         * speed_square
                                         + log_var_speed) * 0.5)
            else:
                branches_out, pred_speed = model(img, speed)
                branch_loss = criterion(branches_out * mask, target) * 4
                speed_loss = criterion(pred_speed, speed)
            
            uncertain_loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss
            uncertain_loss = uncertain_loss / args.gradient_accumulation
        
        # 混合精度反向
        if scaler is not None:
            scaler.scale(uncertain_loss).backward()
        else:
            uncertain_loss.backward()
        
        # 梯度累积
        if (i + 1) % args.gradient_accumulation == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # 聚合loss用于日志
        if args.distributed:
            reduced_loss = reduce_tensor(uncertain_loss.data, args.world_size)
            reduced_branch = reduce_tensor(branch_loss.data, args.world_size)
            reduced_speed = reduce_tensor(speed_loss.data, args.world_size)
        else:
            reduced_loss = uncertain_loss.data
            reduced_branch = branch_loss.data
            reduced_speed = speed_loss.data
        
        uncertain_losses.update(reduced_loss.item() * args.gradient_accumulation, img.size(0))
        branch_losses.update(reduced_branch.item(), img.size(0))
        speed_losses.update(reduced_speed.item(), img.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i % args.print_freq == 0 or i + 1 == len(loader)) and is_main_process(args.rank):
            if writer is not None:
                writer.add_scalar('train/branch_loss', branch_losses.val, step + i)
                writer.add_scalar('train/speed_loss', speed_losses.val, step + i)
                writer.add_scalar('train/uncertain_loss', uncertain_losses.val, step + i)
            
            output_log(
                f'Epoch: [{epoch+1}][{i}/{len(loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {uncertain_losses.val:.4f} ({uncertain_losses.avg:.4f})\t'
                f'Branch {branch_losses.val:.4f} Speed {speed_losses.val:.4f}',
                logging, args.rank)
    
    return branch_losses.avg, speed_losses.avg, uncertain_losses.avg


def evaluate(loader, model, criterion, epoch, writer, args):
    uncertain_losses = AverageMeter()
    
    model.eval()
    with torch.no_grad():
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.local_rank, non_blocking=True)
            speed = speed.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            mask = mask.cuda(args.local_rank, non_blocking=True)
            
            if args.net_structure != 1:
                branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)
                branch_loss = torch.mean((torch.exp(-log_var_control)
                                          * torch.pow((branches_out - target), 2)
                                          + log_var_control) * 0.5 * mask) * 4
                speed_loss = torch.mean((torch.exp(-log_var_speed)
                                         * torch.pow((pred_speed - speed), 2)
                                         + log_var_speed) * 0.5)
            else:
                branches_out, pred_speed = model(img, speed)
                branch_loss = criterion(branches_out * mask, target) * 4
                speed_loss = criterion(pred_speed, speed)
            
            uncertain_loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss
            
            if args.distributed:
                reduced_loss = reduce_tensor(uncertain_loss.data, args.world_size)
            else:
                reduced_loss = uncertain_loss.data
            
            uncertain_losses.update(reduced_loss.item(), img.size(0))
    
    if is_main_process(args.rank):
        if writer is not None:
            writer.add_scalar('eval/loss', uncertain_losses.avg, epoch + 1)
        output_log(f'Epoch Test: [{epoch+1}] Loss {uncertain_losses.avg:.4f}', logging, args.rank)
    
    return uncertain_losses.avg


if __name__ == '__main__':
    main()
