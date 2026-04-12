#!/bin/bash
# FlashOPD 分布式训练启动脚本
#
# 单机单卡:
#   bash scripts/run_distill.sh
#
# 单机多卡 (DDP):
#   NUM_GPUS=4 bash scripts/run_distill.sh
#
# 多机多卡:
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 NNODES=2 NUM_GPUS=8 bash scripts/run_distill.sh  # 主节点
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 NNODES=2 NUM_GPUS=8 bash scripts/run_distill.sh  # 其他节点

set -e

CONFIG=${1:-"configs/default.yaml"}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "  FlashOPD — On-Policy Distillation"
echo "  Config:  $CONFIG"
echo "  Nodes:   $NNODES (rank=$NODE_RANK)"
echo "  GPUs/node: $NUM_GPUS"
echo "============================================"

if [ "$NUM_GPUS" -gt 1 ] || [ "$NNODES" -gt 1 ]; then
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m flashopd.cli \
        --config $CONFIG
else
    python -m flashopd.cli --config $CONFIG
fi
