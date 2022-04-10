from typing import Dict, List
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import json
import shutil

work_dir = "/home/wzx/weizhixiang/papercode/ensegeration/work_dirs"
rebuild = True
os.chdir(work_dir)
log_dirs = os.listdir(work_dir)
log_dirs = filter(osp.isdir, log_dirs)


def record(writer: SummaryWriter, datas: Dict[str, float], step: int):
    for key, value in datas.items():
        if isinstance(value, dict):
            record(
                writer,
                {f"{key}/{subkey}": subval for subkey, subval in value.items()},
                step,
            )
        else:
            writer.add_scalar(key, value, global_step=step)


def find_image(folder) -> List[str]:
    result = []
    for x in os.listdir(folder):
        if x[-4:] == ".png":
            result.append(osp.join(folder, x))
        elif osp.isdir(x):
            result += find_image(osp.join(folder, x))
    return result


for log_dir in log_dirs:
    jsons = [
        osp.join(log_dir, file) for file in os.listdir(log_dir) if file[-5:] == ".json"
    ]

    for js in jsons:
        board_dir = js[:-9]
        if osp.isdir(board_dir):
            if rebuild:
                print(f"rebuild {js}...")
                shutil.rmtree(board_dir)
            else:
                print(f"skip {js}")
                continue
        else:
            print(f"generate {js}")
        with open(js, "r") as load_f:
            lines_f = load_f.readlines()
            if len(lines_f) < 100:
                continue
            writer = SummaryWriter(board_dir)
            val_idx = 0
            for line in lines_f:
                content = json.loads(line)
                popif = lambda d, k: d.pop(k) if k in d else None
                mode = popif(content, "mode")
                epoch = popif(content, "epoch")
                idx = popif(content, "iter")
                time = popif(content, "time")
                memory = popif(content, "memory")
                if mode == "train":
                    val_idx = idx
                    record(writer, content, idx)
                elif mode == "val":
                    popif(content, "lr")
                    record(writer, content, val_idx)
                else:
                    continue
        # images=sorted(find_image(log_dir))
        # for img in images:
        #     imgname=osp.basename(img)
        #     iter=
