import pynvml

pynvml.nvmlInit()
import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a segmentor",
        usage="python auto_start.py 1 2 'command' memory=10",
    )
    parser.add_argument(
        "gpus",
        type=int,
        nargs="+",
        help="GPU to be used",
    )
    parser.add_argument(
        "command",
        type=str,
        help="command to be used",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=-1,
        help="memory to be used(GB)",
    )
    return parser.parse_args()


def allow_execution(gpu_id, memory):
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_id))
    if memory == -1:
        memory = meminfo.total * 0.9 / 1e9
    else:
        memory = memory
    return memory, meminfo.free / 1e9


args = parse_args()
flag = False
print("###auto_start: Wait for start...")
for gpu in args.gpus:
    memory, free = allow_execution(gpu, args.memory)
    if memory < free:
        print(f"gpu{gpu}: need {memory:.2f}GB<free {free:.2f}GB")
    else:
        print(f"gpu{gpu}: need {memory:.2f}GB>free {free:.2f}GB")
while not flag:
    flag = True
    for gpu in args.gpus:
        memory, free = allow_execution(gpu, args.memory)
        if memory > free:
            flag = False
            time.sleep(60)
            break
print("###auto_start: Start!")
os.system(args.command)
print("###auto_start: Done!")
