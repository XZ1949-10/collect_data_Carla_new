#!/usr/bin/env python
# coding=utf-8
'''
模型加载模块
负责加载和管理深度学习模型
'''

import torch
from network.carla_net_old import FinalNet


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, model_path, device, net_structure=2):
        """
        初始化模型加载器
        
        参数:
            model_path (str): 模型权重路径
            device: torch.device 对象
            net_structure (int): 网络结构类型
        """
        self.model_path = model_path
        self.device = device
        self.net_structure = net_structure
        self.model = None
        
    def load(self):
        """加载模型（只加载 CarlaNet 部分的权重）"""
        print(f"正在加载模型: {self.model_path}")
        
        self.model = FinalNet(structure=self.net_structure)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理checkpoint格式
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 处理DataParallel模型（移除"module."前缀）
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        
        # 打印前几个key用于调试
        sample_keys = list(new_state_dict.keys())[:3]
        print(f"   - Checkpoint key 示例: {sample_keys}")
        
        # 检查 key 格式并处理
        has_carla_net_prefix = any(k.startswith('carla_net.') for k in new_state_dict.keys())
        
        if has_carla_net_prefix:
            # 如果已有 carla_net. 前缀，只提取 CarlaNet 部分
            carla_net_state_dict = {k: v for k, v in new_state_dict.items() if k.startswith('carla_net.')}
        else:
            # 如果没有前缀，说明 checkpoint 是纯 CarlaNet，需要添加前缀
            carla_net_state_dict = {'carla_net.' + k: v for k, v in new_state_dict.items()}
        
        # 加载权重（strict=False 允许部分加载）
        missing_keys, unexpected_keys = self.model.load_state_dict(carla_net_state_dict, strict=False)
        
        # 统计实际加载的参数（排除 UncertainNet）
        loaded_count = len([k for k in carla_net_state_dict.keys() if k in dict(self.model.named_parameters())])
        
        print(f"✅ CarlaNet 权重加载成功！")
        print(f"   - 已加载参数: {len(carla_net_state_dict)} 个")
        if missing_keys:
            uncertain_missing = [k for k in missing_keys if 'uncertain_net' in k]
            print(f"   - UncertainNet 未加载参数: {len(uncertain_missing)} 个")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型设备: {self.device}")
        return self.model
