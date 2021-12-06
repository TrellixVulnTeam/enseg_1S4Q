#!/usr/bin/env bash

CONFIG=$1
WEIGHT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $WEIGHT --eval mIoU
