# 红绿灯场景防遗忘微调

针对已训练模型在红绿灯场景表现不佳的问题，提供防止灾难性遗忘的微调方案。

## 问题背景

- 原模型在红绿灯路口不会停车
- 收集了专门的红绿灯场景数据
- 需要在不遗忘原有驾驶能力的前提下，学习红绿灯停车

## 三种防遗忘策略

### 1. EWC (Elastic Weight Consolidation) 弹性权重巩固

**原理**: 计算参数对旧任务的重要性（Fisher信息矩阵），在微调时对重要参数施加约束。

**优点**: 不需要保留旧数据
**缺点**: 需要调节 lambda 参数

```bash
python finetune_anti_forget.py \
    --pretrained /path/to/model.pth \
    --new-train-dir /path/to/traffic_light/train \
    --new-eval-dir /path/to/traffic_light/val \
    --ewc-lambda 5000
```

### 2. 混合数据训练 (推荐)

**原理**: 将新旧数据按比例混合，同时学习新旧任务。

**优点**: 效果最稳定
**缺点**: 需要保留旧数据

```bash
python finetune_anti_forget.py \
    --pretrained /path/to/model.pth \
    --old-train-dir /path/to/original/train \
    --old-eval-dir /path/to/original/val \
    --new-train-dir /path/to/traffic_light/train \
    --new-eval-dir /path/to/traffic_light/val \
    --use-mixed-data \
    --mix-ratio 0.3  # 新数据占30%
```

### 3. 知识蒸馏

**原理**: 用旧模型的输出作为软标签，引导新模型保持旧知识。

**优点**: 不需要旧数据，效果好
**缺点**: 需要额外显存存储教师模型

```bash
python finetune_anti_forget.py \
    --pretrained /path/to/model.pth \
    --new-train-dir /path/to/traffic_light/train \
    --new-eval-dir /path/to/traffic_light/val \
    --use-distillation \
    --distill-alpha 0.5
```

## 推荐配置

### 场景1: 有旧数据，显存充足 (最佳效果)

```bash
./run_finetune.sh
```

使用: 混合数据 + 知识蒸馏

### 场景2: 没有旧数据

```bash
./run_finetune_simple.sh
```

使用: EWC防遗忘

### 场景3: 自定义配置

```bash
python finetune_anti_forget.py \
    --pretrained /path/to/model.pth \
    --old-train-dir /path/to/original/train \
    --old-eval-dir /path/to/original/val \
    --new-train-dir /path/to/traffic_light/train \
    --new-eval-dir /path/to/traffic_light/val \
    --use-mixed-data --mix-ratio 0.3 \
    --use-distillation --distill-alpha 0.3 \
    --ewc-lambda 1000 \
    --lr 5e-5 \
    --epochs 30
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mix-ratio` | 0.5 | 新数据占比，0.3表示新数据30%，旧数据70% |
| `--ewc-lambda` | 0 | EWC正则化强度，推荐1000-10000 |
| `--distill-alpha` | 0.5 | 蒸馏损失权重，越大越保守 |
| `--lr` | 5e-5 | 学习率，微调应比预训练小 |
| `--epochs` | 30 | 训练轮数 |

## 文件结构

```
carla_train_traffic/
├── finetune_anti_forget.py    # 主训练脚本
├── carla_loader_dynamic.py    # 动态帧数数据加载器
├── carla_loader_mixed.py      # 混合数据加载器
├── carla_net_ori.py           # 网络结构
├── helper.py                  # 辅助函数
├── run_finetune.sh            # 完整版启动脚本
├── run_finetune_simple.sh     # 简化版启动脚本
└── README.md                  # 本文档
```

## 注意事项

1. **学习率**: 微调学习率应该比预训练小，推荐 1e-5 ~ 5e-5
2. **mix-ratio**: 如果新数据量少，可以增大比例(如0.5)；数据量大可以减小(如0.2)
3. **ewc-lambda**: 太大会导致学不到新知识，太小防遗忘效果差
4. **distill-alpha**: 0.3-0.5 通常效果较好
5. **数据格式**: 支持不同帧数的h5文件混合训练
