#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29500}


$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG  --seed 0 --launcher pytorch ${PY_ARGS}
