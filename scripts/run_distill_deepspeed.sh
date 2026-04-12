#!/bin/bash
# FlashOPD + DeepSpeed ZeRO-2 分布式训练
#
# 单机多卡:
#   bash scripts/run_distill_deepspeed.sh
#
# 多机多卡 (每台机器都要跑):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 NNODES=2 bash scripts/run_distill_deepspeed.sh  # 主节点
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 NNODES=2 bash scripts/run_distill_deepspeed.sh  # 其他节点

set -e

CONFIG=${1:-"configs/default.yaml"}
DS_CONFIG=${2:-"configs/ds_zero2.json"}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "  FlashOPD + DeepSpeed"
echo "  Config:  $CONFIG"
echo "  DS:      $DS_CONFIG"
echo "  Nodes:   $NNODES (rank=$NODE_RANK)"
echo "  GPUs/node: $NUM_GPUS"
echo "  Master:  $MASTER_ADDR:$MASTER_PORT"
echo "============================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m flashopd.cli \
    --config $CONFIG \
    --deepspeed $DS_CONFIG
