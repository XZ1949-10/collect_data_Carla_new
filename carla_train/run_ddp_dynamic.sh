#!/bin/bash
# 动态帧数版本的分布式训练启动脚本
# 支持不同帧数的h5文件混合训练
#
# 使用方法:
#   ./run_ddp_dynamic.sh                    - 使用默认参数
#   ./run_ddp_dynamic.sh --min-frames 50    - 设置最小帧数阈值

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# 日志目录和文件
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_dynamic_${TIMESTAMP}.log"

echo "日志将保存到: $LOG_FILE"
echo "使用动态帧数加载器，支持不同帧数的h5文件"

torchrun --nproc_per_node=6 --master_port=29500 main_ddp.py \
    --batch-size 1536 \
    --workers 8 \
    --lr 1e-4 \
    --epochs 90 \
    --bucket-cap-mb 100 \
    --channels-last \
    --id ddp_dynamic_1 \
    --early-stop \
    --patience 12 \
    --auto-lr \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --min-lr 1e-7 \
    --dynamic-loader \
    --min-frames 10 \
    "$@" 2>&1 | tee $LOG_FILE
