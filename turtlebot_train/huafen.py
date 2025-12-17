import os
import random
import shutil
from pathlib import Path

# 配置参数
SOURCE_DIR = "/root/data1/carla_cil_pytorch/AgentHuman"
TRAIN_DIR = os.path.join(SOURCE_DIR, "SeqTrain")
VAL_DIR = os.path.join(SOURCE_DIR, "SeqVal")
SPLIT_RATIO = 0.8  # 训练集比例
SEED = 42  # 随机种子，保证划分结果可复现

def split_h5_files():
    # 创建目标目录（如果不存在）
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # 获取所有.h5文件
    h5_files = [f for f in os.listdir(SOURCE_DIR) 
                if f.lower().endswith('.h5') and os.path.isfile(os.path.join(SOURCE_DIR, f))]
    
    if not h5_files:
        print("错误：未找到任何.h5文件！")
        return
    
    # 打乱文件列表（固定随机种子）
    random.seed(SEED)
    random.shuffle(h5_files)
    
    # 计算划分数量
    split_idx = int(len(h5_files) * SPLIT_RATIO)
    train_files = h5_files[:split_idx]
    val_files = h5_files[split_idx:]
    
    # 复制文件到训练目录
    print(f"开始复制 {len(train_files)} 个训练文件...")
    for file_name in train_files:
        src_path = os.path.join(SOURCE_DIR, file_name)
        dst_path = os.path.join(TRAIN_DIR, file_name)
        shutil.copy2(src_path, dst_path)  # copy2会保留文件元数据
    
    # 复制文件到验证目录
    print(f"开始复制 {len(val_files)} 个验证文件...")
    for file_name in val_files:
        src_path = os.path.join(SOURCE_DIR, file_name)
        dst_path = os.path.join(VAL_DIR, file_name)
        shutil.copy2(src_path, dst_path)
    
    # 输出统计信息
    print("\n文件划分完成！")
    print(f"总.h5文件数: {len(h5_files)}")
    print(f"训练集(SeqTrain): {len(train_files)} 个文件 ({len(train_files)/len(h5_files)*100:.1f}%)")
    print(f"验证集(SeqVal): {len(val_files)} 个文件 ({len(val_files)/len(h5_files)*100:.1f}%)")

if __name__ == "__main__":
    # 确认操作
    confirm = input(f"即将处理 {SOURCE_DIR} 目录下的.h5文件，按8:2划分到SeqTrain/SeqVal目录。\n"
                    "注意：这会复制文件（不会删除原文件），是否继续？(y/n): ")
    if confirm.lower() == 'y':
        split_h5_files()
    else:
        print("操作已取消。")