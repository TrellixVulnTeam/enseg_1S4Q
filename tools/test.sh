#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WEIGHT=$3
MS=$4
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
if test $MS = 'ms'
then
    echo use multiscale and flip
    CUDA_VISIBLE_DEVICES=$GPUS python $(dirname "$0")/test.py $CONFIG $WEIGHT --eval mIoU --aug-test
else
    CUDA_VISIBLE_DEVICES=$GPUS python $(dirname "$0")/test.py $CONFIG $WEIGHT --eval mIoU
fi


