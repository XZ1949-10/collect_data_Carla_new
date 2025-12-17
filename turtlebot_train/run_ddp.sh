#!/bin/bash
# 6卡P100分布式训练启动脚本
#
# 使用方法:
#   ./run_ddp.sh                    - 使用默认参数
#   ./run_ddp.sh --epochs 50        - 自定义参数
#   ./run_ddp.sh --channels-last    - 启用channels_last优化
#   ./run_ddp.sh --lr-finder        - 启用学习率自动查找
#   ./run_ddp.sh --no-early-stop    - 禁用早停
#   ./run_ddp.sh --patience 15      - 设置早停耐心值
#   ./run_ddp.sh --lr-patience 8    - 设置学习率调度耐心值

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=4
# NCCL优化
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# 日志目录和文件
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "日志将保存到: $LOG_FILE"

# P100 是 SM 6.0 架构，不支持 Tensor Core，AMP 无加速效果
# 推荐优化: --channels-last (CNN内存布局优化)
# PyTorch 2.0.1 启用 torch.compile 优化
torchrun --nproc_per_node=6 --master_port=29500 main_ddp.py \
    --batch-size 1536 \
    --workers 8 \
    --lr 1e-4 \
    --epochs 90 \
    --bucket-cap-mb 100 \
    --channels-last \
    --id ddp_6gpu_5 \
    --early-stop \
    --patience 12 \
    --auto-lr \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --min-lr 1e-7 \
    "$@" 2>&1 | tee $LOG_FILE

# 如果显存不够可以加 --use-amp 省显存，但不会加速
