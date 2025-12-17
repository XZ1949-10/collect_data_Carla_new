#!/usr/bin/env python
# coding=utf-8
'''
模型加载模块
支持 CARLA 格式和 TurtleBot 格式的网络
'''

import torch


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
        
    def load(self, network_module=None):
        """
        加载模型
        
        参数:
            network_module: 网络模块，如果为 None 则尝试从默认路径导入
            
        返回:
            model: 加载好的模型
        """
        print(f"正在加载模型: {self.model_path}")
        
        # 导入网络结构
        if network_module is None:
            # 尝试导入 TurtleBot 网络 (FinalNet)
            try:
                from turtlebot_train.turtlebot_net import FinalNet
                print("使用 TurtleBot FinalNet 网络 (输出 2 维)")
            except ImportError:
                # 回退到 CARLA 网络
                try:
                    from network.carla_net_old import FinalNet
                    print("使用 CARLA FinalNet 网络 (输出 3 维)")
                except ImportError:
                    raise ImportError(
                        "无法导入网络模块，请确保 turtlebot_train/turtlebot_net.py 或 "
                        "network/carla_net_old.py 存在，或者通过 network_module 参数传入网络类"
                    )
        else:
            FinalNet = network_module
        
        self.model = FinalNet(structure=self.net_structure)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理 checkpoint 格式
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 处理 DataParallel 模型（移除 "module." 前缀）
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        
        # 检查 key 格式并处理 (支持 carla_net 和 turtlebot_net 前缀)
        has_carla_net_prefix = any(k.startswith('carla_net.') for k in new_state_dict.keys())
        has_turtlebot_net_prefix = any(k.startswith('turtlebot_net.') for k in new_state_dict.keys())
        
        if has_turtlebot_net_prefix:
            # TurtleBot 网络格式
            net_state_dict = {k: v for k, v in new_state_dict.items() 
                             if k.startswith('turtlebot_net.') or k.startswith('uncertain_net.')}
        elif has_carla_net_prefix:
            # CARLA 网络格式
            net_state_dict = {k: v for k, v in new_state_dict.items() 
                             if k.startswith('carla_net.') or k.startswith('uncertain_net.')}
        else:
            # 尝试添加前缀
            # 检查模型是否有 turtlebot_net 属性
            if hasattr(self.model, 'turtlebot_net'):
                net_state_dict = {'turtlebot_net.' + k: v for k, v in new_state_dict.items()}
            else:
                net_state_dict = {'carla_net.' + k: v for k, v in new_state_dict.items()}
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(
            net_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"⚠️ 缺失的参数: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"⚠️ 多余的参数: {len(unexpected_keys)} 个")
        
        print(f"✅ 模型加载成功！已加载参数: {len(net_state_dict)} 个")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型设备: {self.device}")
        return self.model
    
    def get_model(self):
        """获取模型"""
        return self.model
