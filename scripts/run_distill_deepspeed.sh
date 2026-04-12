#!/bin/bash
# FlashOPD + DeepSpeed ZeRO-2 分布式训练
# 用法: bash scripts/run_distill_deepspeed.sh

set -e

CONFIG=${1:-"configs/default.yaml"}
DS_CONFIG=${2:-"configs/ds_zero2.json"}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "  FlashOPD + DeepSpeed"
echo "  Config:  $CONFIG"
echo "  DS:      $DS_CONFIG"
echo "  GPUs:    $NUM_GPUS"
echo "============================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m flashopd.cli \
    --config $CONFIG \
    --deepspeed $DS_CONFIG
