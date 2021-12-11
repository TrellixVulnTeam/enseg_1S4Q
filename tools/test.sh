#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WEIGHT=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS python $(dirname "$0")/test.py $CONFIG $WEIGHT --eval mIoU
