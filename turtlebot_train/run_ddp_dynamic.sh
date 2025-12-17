#!/bin/bash
# TurtleBot 动态帧数版本的分布式训练启动脚本
# 
# 特点:
#   - 支持不同帧数的 h5 文件混合训练
#   - 预测 linear_vel 和 angular_vel (而不是 steer, throttle, brake)
#   - 输出维度: 4 分支 × 2 维 = 8 维
#
# 使用方法:
#   ./run_ddp_dynamic.sh                    - 使用默认参数
#   ./run_ddp_dynamic.sh --min-frames 50    - 设置最小帧数阈值
#
# 数据格式:
#   H5 文件中的 targets 向量:
#     targets[10] = speed (km/h)
#     targets[20] = linear_vel (m/s)
#     targets[21] = angular_vel (rad/s)
#     targets[24] = command (2/3/4/5)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# 日志目录和文件
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_turtlebot_${TIMESTAMP}.log"

echo "=========================================="
echo "TurtleBot 训练 (linear_vel, angular_vel)"
echo "=========================================="
echo "日志将保存到: $LOG_FILE"
echo "使用动态帧数加载器，支持不同帧数的 h5 文件"
echo ""

torchrun --nproc_per_node=6 --master_port=29500 main_ddp.py \
    --batch-size 1536 \
    --workers 8 \
    --lr 1e-4 \
    --epochs 90 \
    --bucket-cap-mb 100 \
    --channels-last \
    --id turtlebot_ddp_1 \
    --early-stop \
    --patience 12 \
    --auto-lr \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --min-lr 1e-7 \
    --dynamic-loader \
    --min-frames 10 \
    --max-linear-vel 0.7 \
    --max-angular-vel 1.0 \
    "$@" 2>&1 | tee $LOG_FILE
