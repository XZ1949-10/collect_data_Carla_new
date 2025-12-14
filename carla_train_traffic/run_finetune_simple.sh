#!/bin/bash
# 简化版微调脚本 - 仅使用新数据 + EWC防遗忘
# 适用于没有旧数据或旧数据太大的情况

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
NUM_GPUS=6

# ============================================================
# 配置区域 - 请修改为你的实际路径
# ============================================================

# 预训练模型
PRETRAINED_MODEL="/path/to/your/best_model.pth"

# 新数据 (红绿灯场景)
NEW_TRAIN_DIR="/path/to/traffic_light/train"
NEW_EVAL_DIR="/path/to/traffic_light/val"

# ============================================================

export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1

LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/finetune_simple_${TIMESTAMP}.log"

echo "============================================================"
echo "🚦 简化版微调 (仅新数据 + EWC)"
echo "============================================================"
echo "日志: $LOG_FILE"

# 使用EWC防遗忘，不需要旧数据
# ewc-lambda: 越大越保守，推荐 1000-10000

torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 finetune_anti_forget.py \
    --pretrained "$PRETRAINED_MODEL" \
    --new-train-dir "$NEW_TRAIN_DIR" \
    --new-eval-dir "$NEW_EVAL_DIR" \
    --batch-size 768 \
    --workers 6 \
    --lr 5e-5 \
    --epochs 30 \
    --ewc-lambda 5000 \
    --ewc-samples 3000 \
    --early-stop \
    --patience 8 \
    --channels-last \
    --id finetune_ewc_only \
    "$@" 2>&1 | tee $LOG_FILE
