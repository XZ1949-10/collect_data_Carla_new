#!/usr/bin/env python
# coding=utf-8
'''
微调脚本：在转弯场景数据上继续训练最佳模型
用于改善模型在转弯场景的表现

使用方法:
    python finetune.py --pretrained ./save_models/your_best_model.pth \
                       --train-dir ./carla_data/TurnScenes/Train \
                       --eval-dir ./carla_data/TurnScenes/Val \
                       --epochs 30 --lr 1e-5
'''
import argparse
import os
import random
import time
import datetime
import math
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

from carla_net_ori import CarlaNet, FinalNet
from carla_loader import CarlaH5Data
from helper import AverageMeter, save_checkpoint


parser = argparse.ArgumentParser(description='Carla Model Fine-tuning for Turn Scenarios')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--speed-weight', default=0.5, type=float,
                    help='speed weight (降低速度权重，更关注转向)')
parser.add_argument('--branch-weight', default=1.5, type=float,
                    help='branch weight (提高分支权重，更关注转向控制)')
parser.add_argument('--id', default="finetune_turn", type=str,
                    help='experiment id')
parser.add_argument('--pretrained', default='./save_models/test_demo_best.pth', type=str,
                    help='path to pretrained best model')
parser.add_argument('--train-dir',
                    default=r"./carla_data/TurnScenes/Train",
                    type=str, metavar='PATH',
                    help='turn scenario training dataset')
parser.add_argument('--eval-dir',
                    default=r"./carla_data/TurnScenes/Val",
                    type=str, metavar='PATH',
                    help='turn scenario evaluation dataset')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of fine-tuning epochs (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='fine-tuning learning rate (比原始训练小)')
parser.add_argument('--lr-step', default=10, type=int,
                    help='learning rate step size')
parser.add_argument('--lr-gamma', default=0.5, type=float,
                    help='learning rate gamma')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for reproducibility')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--net-structure', default=1, type=int,
                    help='Network structure 1|2|3')
# 微调特定参数
parser.add_argument('--freeze-conv', action='store_true', default=False,
                    help='freeze convolutional layers, only fine-tune FC layers')
parser.add_argument('--freeze-epochs', default=5, type=int,
                    help='epochs to freeze conv layers before unfreezing')


def output_log(output_str, logger=None):
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def load_pretrained_model(model, pretrained_path, logger=None):
    """加载预训练模型权重"""
    if not os.path.exists(pretrained_path):
        output_log(f"预训练模型不存在: {pretrained_path}", logger)
        return False
    
    output_log(f"加载预训练模型: {pretrained_path}", logger)
    
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 兼容不同的checkpoint格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理可能的 'module.' 前缀 (DataParallel保存的模型)
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除 'module.' 前缀
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        # 尝试直接加载
        try:
            model.load_state_dict(new_state_dict, strict=True)
            output_log("✓ 预训练模型加载成功 (strict=True)", logger)
        except RuntimeError as e:
            # 如果strict失败，尝试非严格加载
            output_log(f"strict加载失败，尝试非严格加载: {e}", logger)
            model.load_state_dict(new_state_dict, strict=False)
            output_log("✓ 预训练模型加载成功 (strict=False)", logger)
        
        return True
        
    except Exception as e:
        output_log(f"✗ 加载预训练模型失败: {e}", logger)
        return False


def freeze_conv_layers(model, freeze=True):
    """冻结/解冻卷积层"""
    # 获取实际的模型（处理DataParallel包装）
    actual_model = model.module if hasattr(model, 'module') else model
    
    for param in actual_model.carla_net.conv_block.parameters():
        param.requires_grad = not freeze
    
    status = "冻结" if freeze else "解冻"
    print(f"卷积层已{status}")


def main():
    global args
    args = parser.parse_args()
    
    # 创建目录
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "finetune.log"),
                        level=logging.ERROR)
    tsbd = SummaryWriter(log_dir=run_dir)
    
    output_log("=" * 50, logging)
    output_log("转弯场景微调训练", logging)
    output_log("=" * 50, logging)
    output_log(f"预训练模型: {args.pretrained}", logging)
    output_log(f"训练数据: {args.train_dir}", logging)
    output_log(f"学习率: {args.lr}", logging)
    output_log(f"Epochs: {args.epochs}", logging)
    output_log(f"Branch权重: {args.branch_weight}, Speed权重: {args.speed_weight}", logging)
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    # 创建模型
    model = FinalNet(args.net_structure)
    
    # 加载预训练权重
    if not load_pretrained_model(model, args.pretrained, logging):
        output_log("警告: 未能加载预训练模型，将从头开始训练", logging)
    
    criterion = nn.MSELoss()
    
    # GPU设置
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    # 冻结卷积层（可选）
    if args.freeze_conv:
        freeze_conv_layers(model, freeze=True)
        output_log(f"卷积层已冻结，将在第{args.freeze_epochs}个epoch后解冻", logging)
    
    # 优化器 - 使用较小的学习率进行微调
    if args.net_structure != 1:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr, betas=(0.7, 0.85), weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr, betas=(0.7, 0.85), weight_decay=args.weight_decay)
    
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    cudnn.benchmark = True
    
    # 加载转弯场景数据
    carla_data = CarlaH5Data(
        train_folder=args.train_dir,
        eval_folder=args.eval_dir,
        batch_size=args.batch_size,
        num_workers=args.workers)
    
    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]
    
    output_log(f"训练集大小: {len(train_loader.dataset)}", logging)
    output_log(f"验证集大小: {len(eval_loader.dataset)}", logging)
    
    best_prec = math.inf
    
    # 微调训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 在指定epoch后解冻卷积层
        if args.freeze_conv and epoch == args.freeze_epochs:
            freeze_conv_layers(model, freeze=False)
            # 重新创建优化器以包含所有参数
            optimizer = optim.Adam(
                model.parameters(),
                args.lr * 0.1,  # 解冻后使用更小的学习率
                betas=(0.7, 0.85), weight_decay=args.weight_decay)
            output_log("卷积层已解冻，学习率降低10倍", logging)
        
        # 训练
        train(train_loader, model, criterion, optimizer, epoch, tsbd)
        
        # 验证
        prec = evaluate(eval_loader, model, criterion, epoch, tsbd)
        
        lr_scheduler.step()
        
        # 保存最佳模型
        is_best = prec < best_prec
        best_prec = min(prec, best_prec)
        
        save_checkpoint(
            {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'best_prec': best_prec,
             'scheduler': lr_scheduler.state_dict(),
             'optimizer': optimizer.state_dict()},
            args.id,
            is_best,
            os.path.join(save_weight_dir, f"{epoch+1}_{args.id}.pth"))
        
        if is_best:
            output_log(f"★ 新的最佳模型! Loss: {best_prec:.4f}", logging)
    
    output_log("=" * 50, logging)
    output_log(f"微调完成! 最佳验证Loss: {best_prec:.4f}", logging)
    output_log(f"最佳模型保存在: ./save_models/{args.id}_best.pth", logging)


def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    branch_losses = AverageMeter()
    speed_losses = AverageMeter()
    total_losses = AverageMeter()

    model.train()
    end = time.time()
    step = epoch * len(loader)
    
    for i, (img, speed, target, mask) in enumerate(loader):
        data_time.update(time.time() - end)

        img = img.cuda(args.gpu, non_blocking=True)
        speed = speed.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        mask = mask.cuda(args.gpu, non_blocking=True)

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
        
        total_loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss

        total_losses.update(total_loss.item(), args.batch_size)
        branch_losses.update(branch_loss.item(), args.batch_size)
        speed_losses.update(speed_loss.item(), args.batch_size)

        optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪，防止微调时梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(loader):
            writer.add_scalar('finetune/branch_loss', branch_losses.val, step + i)
            writer.add_scalar('finetune/speed_loss', speed_losses.val, step + i)
            writer.add_scalar('finetune/total_loss', total_losses.val, step + i)
            
            output_log(
                f'Epoch: [{epoch+1}][{i}/{len(loader)}] '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                f'Loss {total_losses.val:.4f} ({total_losses.avg:.4f}) '
                f'Branch {branch_losses.val:.4f} Speed {speed_losses.val:.4f}',
                logging)

    return total_losses.avg


def evaluate(loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    branch_losses = AverageMeter()
    speed_losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.gpu, non_blocking=True)
            speed = speed.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)

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
            
            total_loss = args.branch_weight * branch_loss + args.speed_weight * speed_loss
            
            total_losses.update(total_loss.item(), args.batch_size)
            branch_losses.update(branch_loss.item(), args.batch_size)
            speed_losses.update(speed_loss.item(), args.batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

        writer.add_scalar('finetune_eval/total_loss', total_losses.avg, epoch + 1)
        writer.add_scalar('finetune_eval/branch_loss', branch_losses.avg, epoch + 1)
        writer.add_scalar('finetune_eval/speed_loss', speed_losses.avg, epoch + 1)
        
        output_log(
            f'Eval [{epoch+1}] '
            f'Time {batch_time.avg:.3f} '
            f'Loss {total_losses.avg:.4f} '
            f'Branch {branch_losses.avg:.4f} Speed {speed_losses.avg:.4f}',
            logging)
    
    return total_losses.avg


if __name__ == '__main__':
    main()
