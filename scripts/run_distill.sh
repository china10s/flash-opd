#!/bin/bash
# FlashOPD 分布式训练启动脚本
# 用法: bash scripts/run_distill.sh [CONFIG_PATH]

set -e

CONFIG=${1:-"configs/default.yaml"}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}

echo "============================================"
echo "  FlashOPD — On-Policy Distillation"
echo "  Config:  $CONFIG"
echo "  GPUs:    $NUM_GPUS"
echo "============================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=${MASTER_PORT:-29500} \
        -m flashopd.cli \
        --config $CONFIG
else
    python -m flashopd.cli --config $CONFIG
fi
