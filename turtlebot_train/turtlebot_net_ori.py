#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 网络结构

输出: 直接预测 linear_vel 和 angular_vel (2维)
而不是 CARLA 的 steer, throttle, brake (3维)

分支结构:
    - 4 个分支对应 4 种导航命令 (Follow, Left, Right, Straight)
    - 每个分支输出 2 维: [linear_vel, angular_vel]
    - 总输出: 4 * 2 = 8 维

速度输入:
    - 使用当前速度作为条件输入
    - 归一化: speed / 25 (与 CARLA 一致)
'''

import torch
import torch.nn as nn


# ============ 配置常量 ============
# 每个分支的输出维度
BRANCH_OUTPUT_DIM = 2  # [linear_vel, angular_vel]

# 分支数量 (Follow, Left, Right, Straight)
NUM_BRANCHES = 4


class TurtleBotNet(nn.Module):
    """
    TurtleBot 端到端网络
    
    输入:
        - img: (B, 3, 88, 200) RGB 图像
        - speed: (B, 1) 归一化速度
        
    输出:
        - pred_control: (B, 8) 4个分支 × 2维控制
        - pred_speed: (B, 1) 预测速度
    """
    def __init__(self, dropout_vec=None):
        super(TurtleBotNet, self).__init__()
        
        # 卷积特征提取 (与 CARLA 网络相同)
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 图像特征全连接
        self.img_fc = nn.Sequential(
            nn.Linear(8192, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        # 速度特征全连接
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # 融合特征全连接
        self.emb_fc = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # 控制分支 (4个分支，每个输出2维)
        self.control_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, BRANCH_OUTPUT_DIM),  # 输出 2 维
            ) for _ in range(NUM_BRANCHES)
        ])

        # 速度预测分支
        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, img, speed):
        # 卷积特征提取
        img = self.conv_block(img)
        img = img.reshape(-1, 8192)
        img = self.img_fc(img)

        # 速度特征
        speed = self.speed_fc(speed)
        
        # 特征融合
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        # 控制输出 (4个分支拼接)
        pred_control = torch.cat(
            [branch(emb) for branch in self.control_branches], dim=1)
        
        # 速度预测
        pred_speed = self.speed_branch(img)
        
        return pred_control, pred_speed, img, emb


class TurtleBotUncertainNet(nn.Module):
    """
    TurtleBot 不确定性网络
    
    用于估计预测的不确定性 (aleatoric uncertainty)
    """
    def __init__(self, structure=2, dropout_vec=None):
        super(TurtleBotUncertainNet, self).__init__()
        self.structure = structure

        if self.structure < 2 or self.structure > 3:
            raise ValueError("Structure must be one of 2|3")

        # 速度不确定性分支
        self.uncert_speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        if self.structure == 2:
            # 每个分支独立的不确定性
            self.uncert_control_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, BRANCH_OUTPUT_DIM),  # 输出 2 维
                ) for _ in range(NUM_BRANCHES)
            ])

        if self.structure == 3:
            # 共享的不确定性
            self.uncert_control_branches = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, BRANCH_OUTPUT_DIM),  # 输出 2 维
            )

    def forward(self, img_emb, emb):
        if self.structure == 2:
            log_var_control = torch.cat(
                [branch(emb) for branch in self.uncert_control_branches], dim=1)
        if self.structure == 3:
            log_var_control = self.uncert_control_branches(emb)
            log_var_control = torch.cat(
                [log_var_control for _ in range(NUM_BRANCHES)], dim=1)

        log_var_speed = self.uncert_speed_branch(img_emb)

        return log_var_control, log_var_speed


class FinalNet(nn.Module):
    """
    TurtleBot 完整网络
    
    包含:
        - TurtleBotNet: 主网络
        - TurtleBotUncertainNet: 不确定性网络 (可选)
    """
    def __init__(self, structure=2, dropout_vec=None):
        super(FinalNet, self).__init__()
        self.structure = structure

        self.turtlebot_net = TurtleBotNet(dropout_vec=dropout_vec)
        if structure != 1:
            self.uncertain_net = TurtleBotUncertainNet(structure)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, img, speed):
        pred_control, pred_speed, img_emb, emb = self.turtlebot_net(img, speed)
        if self.structure != 1:
            log_var_control, log_var_speed = self.uncertain_net(img_emb, emb)
            return pred_control, pred_speed, log_var_control, log_var_speed
        return pred_control, pred_speed


# ============ 兼容性别名 ============
# 保持与旧代码的兼容性
CarlaNet = TurtleBotNet
UncertainNet = TurtleBotUncertainNet
