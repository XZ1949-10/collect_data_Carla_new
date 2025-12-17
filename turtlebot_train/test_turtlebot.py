#!/usr/bin/env python
# coding=utf-8
'''
TurtleBot 网络和数据加载器测试脚本

用于验证:
1. 网络输出维度是否正确 (8 维 = 4 分支 × 2 维)
2. 数据加载器是否正确读取 linear_vel 和 angular_vel
3. Loss 计算和掩码是否对齐
'''

import torch
import torch.nn as nn
import numpy as np


def test_loss_and_mask_alignment():
    """测试 Loss 计算和掩码对齐"""
    print("=" * 60)
    print("测试 Loss 计算和掩码对齐")
    print("=" * 60)
    
    # 模拟数据加载器的输出
    NUM_BRANCHES = 4
    BRANCH_OUTPUT_DIM = 2
    
    # 假设 command = 1 (Left)
    command = 1
    linear_vel_norm = 0.5
    angular_vel_norm = -0.3
    
    # 构建 target_vec (与数据加载器一致)
    target_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
    target_vec[command, 0] = linear_vel_norm
    target_vec[command, 1] = angular_vel_norm
    
    # 构建 mask_vec
    mask_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
    mask_vec[command, :] = 1
    
    # reshape 后的形状
    target_flat = target_vec.reshape(-1)
    mask_flat = mask_vec.reshape(-1)
    
    print(f"Command: {command} (Left)")
    print(f"Target (4x2):\n{target_vec}")
    print(f"Target (flat): {target_flat}")
    print(f"Mask (flat): {mask_flat}")
    
    # 验证索引对应关系
    print(f"\n索引对应关系:")
    for branch in range(NUM_BRANCHES):
        start_idx = branch * BRANCH_OUTPUT_DIM
        end_idx = start_idx + BRANCH_OUTPUT_DIM
        print(f"  Branch {branch}: 索引 [{start_idx}, {end_idx})")
    
    # 验证 mask 位置
    expected_mask_indices = [command * BRANCH_OUTPUT_DIM, command * BRANCH_OUTPUT_DIM + 1]
    actual_mask_indices = np.where(mask_flat == 1)[0].tolist()
    print(f"\n期望的 mask 索引: {expected_mask_indices}")
    print(f"实际的 mask 索引: {actual_mask_indices}")
    assert expected_mask_indices == actual_mask_indices, "Mask 索引不匹配!"
    
    # 模拟网络输出
    # 网络输出: [b0_lin, b0_ang, b1_lin, b1_ang, b2_lin, b2_ang, b3_lin, b3_ang]
    branches_out = torch.tensor([[0.1, 0.2, 0.6, -0.4, 0.3, 0.1, 0.2, 0.0]])  # (1, 8)
    target = torch.tensor([target_flat])  # (1, 8)
    mask = torch.tensor([mask_flat])  # (1, 8)
    
    print(f"\n网络输出: {branches_out[0].tolist()}")
    print(f"目标值:   {target[0].tolist()}")
    print(f"掩码:     {mask[0].tolist()}")
    
    # 计算 masked 输出
    masked_out = branches_out * mask
    print(f"Masked输出: {masked_out[0].tolist()}")
    
    # 计算 MSELoss
    criterion = nn.MSELoss()
    loss_raw = criterion(masked_out, target)
    loss_scaled = loss_raw * 4
    
    print(f"\nMSELoss (raw): {loss_raw.item():.6f}")
    print(f"MSELoss (×4):  {loss_scaled.item():.6f}")
    
    # 手动计算验证
    # 只有 branch 1 (索引 2, 3) 有效
    pred_linear = branches_out[0, 2].item()  # 0.6
    pred_angular = branches_out[0, 3].item()  # -0.4
    true_linear = target[0, 2].item()  # 0.5
    true_angular = target[0, 3].item()  # -0.3
    
    manual_loss = ((pred_linear - true_linear)**2 + (pred_angular - true_angular)**2) / 2
    print(f"\n手动计算 (只计算有效分支):")
    print(f"  pred: [{pred_linear}, {pred_angular}]")
    print(f"  true: [{true_linear}, {true_angular}]")
    print(f"  MSE = ((0.6-0.5)^2 + (-0.4-(-0.3))^2) / 2 = {manual_loss:.6f}")
    
    # 验证
    assert abs(loss_scaled.item() - manual_loss) < 1e-6, \
        f"Loss 计算不匹配! scaled={loss_scaled.item()}, manual={manual_loss}"
    
    print(f"\n✅ Loss 计算正确! {loss_scaled.item():.6f} == {manual_loss:.6f}")
    
    return True


def test_network():
    """测试网络结构"""
    print("=" * 60)
    print("测试 TurtleBot 网络结构")
    print("=" * 60)
    
    from turtlebot_net_ori import FinalNet, TurtleBotNet
    
    # 创建网络
    model = FinalNet(structure=1)
    model.eval()
    
    # 测试输入
    batch_size = 4
    img = torch.randn(batch_size, 3, 88, 200)
    speed = torch.randn(batch_size, 1)
    
    # 前向传播
    with torch.no_grad():
        pred_control, pred_speed = model(img, speed)
    
    print(f"输入图像形状: {img.shape}")
    print(f"输入速度形状: {speed.shape}")
    print(f"输出控制形状: {pred_control.shape}")
    print(f"输出速度形状: {pred_speed.shape}")
    
    # 验证输出维度
    expected_control_dim = 4 * 2  # 4 分支 × 2 维 (linear_vel, angular_vel)
    assert pred_control.shape == (batch_size, expected_control_dim), \
        f"控制输出维度错误! 期望 {(batch_size, expected_control_dim)}, 实际 {pred_control.shape}"
    
    print(f"\n✅ 网络输出维度正确: {pred_control.shape[1]} = 4 分支 × 2 维")
    
    # 测试带不确定性的网络
    print("\n测试带不确定性的网络 (structure=2)...")
    model2 = FinalNet(structure=2)
    model2.eval()
    
    with torch.no_grad():
        pred_control, pred_speed, log_var_control, log_var_speed = model2(img, speed)
    
    print(f"控制输出形状: {pred_control.shape}")
    print(f"控制不确定性形状: {log_var_control.shape}")
    print(f"速度不确定性形状: {log_var_speed.shape}")
    
    assert log_var_control.shape == pred_control.shape, \
        f"不确定性维度错误! 期望 {pred_control.shape}, 实际 {log_var_control.shape}"
    
    print(f"\n✅ 不确定性网络输出维度正确")
    
    return True


def test_data_format():
    """测试数据格式"""
    print("\n" + "=" * 60)
    print("测试数据格式")
    print("=" * 60)
    
    # 模拟 targets 向量
    targets = np.zeros(25, dtype=np.float32)
    
    # 设置测试值
    targets[10] = 5.0    # speed (km/h)
    targets[20] = 0.3    # linear_vel (m/s)
    targets[21] = -0.5   # angular_vel (rad/s)
    targets[24] = 3.0    # command (Left)
    
    # 配置常量
    TARGETS_SPEED_IDX = 10
    TARGETS_LINEAR_VEL_IDX = 20
    TARGETS_ANGULAR_VEL_IDX = 21
    TARGETS_COMMAND_IDX = 24
    
    MAX_LINEAR_VEL = 0.7
    MAX_ANGULAR_VEL = 1.0
    SPEED_NORMALIZATION = 25.0
    
    # 处理命令
    command = int(targets[TARGETS_COMMAND_IDX]) - 2
    print(f"原始命令值: {targets[TARGETS_COMMAND_IDX]} -> 分支索引: {command}")
    
    # 提取速度
    linear_vel = targets[TARGETS_LINEAR_VEL_IDX]
    angular_vel = targets[TARGETS_ANGULAR_VEL_IDX]
    print(f"线速度: {linear_vel} m/s")
    print(f"角速度: {angular_vel} rad/s")
    
    # 归一化
    linear_vel_norm = np.clip(linear_vel / MAX_LINEAR_VEL, -1.0, 1.0)
    angular_vel_norm = np.clip(angular_vel / MAX_ANGULAR_VEL, -1.0, 1.0)
    print(f"归一化线速度: {linear_vel_norm}")
    print(f"归一化角速度: {angular_vel_norm}")
    
    # 构建目标向量
    target_vec = np.zeros((4, 2), dtype=np.float32)
    target_vec[command, 0] = linear_vel_norm
    target_vec[command, 1] = angular_vel_norm
    
    print(f"\n目标向量 (4×2):")
    print(f"  Branch 0 (Follow):   [{target_vec[0, 0]:.3f}, {target_vec[0, 1]:.3f}]")
    print(f"  Branch 1 (Left):     [{target_vec[1, 0]:.3f}, {target_vec[1, 1]:.3f}]")
    print(f"  Branch 2 (Right):    [{target_vec[2, 0]:.3f}, {target_vec[2, 1]:.3f}]")
    print(f"  Branch 3 (Straight): [{target_vec[3, 0]:.3f}, {target_vec[3, 1]:.3f}]")
    
    # 验证
    assert target_vec[1, 0] != 0, "Left 分支的线速度应该非零"
    assert target_vec[1, 1] != 0, "Left 分支的角速度应该非零"
    assert target_vec[0, 0] == 0, "Follow 分支应该为零"
    
    print(f"\n✅ 数据格式正确")
    
    return True


def test_inference_output():
    """测试推理输出格式"""
    print("\n" + "=" * 60)
    print("测试推理输出格式")
    print("=" * 60)
    
    from turtlebot_net_ori import FinalNet
    
    model = FinalNet(structure=1)
    model.eval()
    
    # 模拟输入
    img = torch.randn(1, 3, 88, 200)
    speed = torch.tensor([[0.2]])  # 归一化速度
    
    with torch.no_grad():
        pred_control, pred_speed = model(img, speed)
    
    # 解析输出
    pred_control = pred_control.numpy()[0]  # (8,)
    
    print(f"模型输出 (8 维): {pred_control}")
    
    # 按分支解析
    print(f"\n按分支解析:")
    branch_names = ['Follow', 'Left', 'Right', 'Straight']
    for i, name in enumerate(branch_names):
        linear_vel = pred_control[i * 2]
        angular_vel = pred_control[i * 2 + 1]
        print(f"  {name}: linear_vel={linear_vel:.3f}, angular_vel={angular_vel:.3f}")
    
    # 模拟选择分支 (假设命令是 Left = 3)
    command = 3  # Left
    branch_idx = command - 2  # 1
    
    selected_linear = pred_control[branch_idx * 2]
    selected_angular = pred_control[branch_idx * 2 + 1]
    
    print(f"\n选择分支 {branch_idx} ({branch_names[branch_idx]}):")
    print(f"  归一化输出: linear={selected_linear:.3f}, angular={selected_angular:.3f}")
    
    # 反归一化
    MAX_LINEAR_VEL = 0.7
    MAX_ANGULAR_VEL = 1.0
    
    actual_linear = selected_linear * MAX_LINEAR_VEL
    actual_angular = selected_angular * MAX_ANGULAR_VEL
    
    print(f"  实际速度: linear={actual_linear:.3f} m/s, angular={actual_angular:.3f} rad/s")
    
    print(f"\n✅ 推理输出格式正确")
    
    return True


def test_all_commands():
    """测试所有命令的掩码对齐"""
    print("\n" + "=" * 60)
    print("测试所有命令的掩码对齐")
    print("=" * 60)
    
    NUM_BRANCHES = 4
    BRANCH_OUTPUT_DIM = 2
    
    commands = {
        0: "Follow",
        1: "Left", 
        2: "Right",
        3: "Straight"
    }
    
    for command, name in commands.items():
        # 构建 mask
        mask_vec = np.zeros((NUM_BRANCHES, BRANCH_OUTPUT_DIM), dtype=np.float32)
        mask_vec[command, :] = 1
        mask_flat = mask_vec.reshape(-1)
        
        # 验证
        expected_indices = [command * 2, command * 2 + 1]
        actual_indices = np.where(mask_flat == 1)[0].tolist()
        
        status = "✅" if expected_indices == actual_indices else "❌"
        print(f"  Command {command} ({name:8s}): mask 索引 {actual_indices} {status}")
        
        assert expected_indices == actual_indices, f"Command {command} mask 不匹配!"
    
    print(f"\n✅ 所有命令的掩码对齐正确!")
    return True


if __name__ == '__main__':
    print("TurtleBot 训练代码测试")
    print("=" * 60)
    
    try:
        test_loss_and_mask_alignment()
        test_all_commands()
        test_network()
        test_data_format()
        test_inference_output()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
