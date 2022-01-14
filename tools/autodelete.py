import mmcv
import os
import os.path as osp
import json
import os
import time
import re
import shutil


def delete_model(root):
    files = mmcv.scandir(root, ".pth", recursive=True)
    # print(list(files))
    need_deletes = [
        "iter_8000.pth",
        "iter_16000.pth",
        "iter_24000.pth",
        "iter_32000.pth",
        "iter_40000.pth",
        "iter_48000.pth",
        "iter_56000.pth",
        "iter_64000.pth",
    ]
    for file in files:
        if sum(map(lambda x: osp.basename(file) == x, need_deletes)):
            path = osp.join(root, file)
            os.remove(path)
            print(f"delete {file}")


def json_need_delete(json_file):
    cur_date = time.localtime()
    file_date = time.strptime(osp.basename(json_file)[:8], "%Y%m%d")
    if (
        file_date.tm_year == cur_date.tm_year
        and file_date.tm_mon == cur_date.tm_mon
        and abs(file_date.tm_mday - cur_date.tm_mday) < 2
    ):
        return False
    date = osp.basename(json_file)[:8]
    with open(json_file, "r") as f:
        content = f.readlines()
        total_iters = json.loads(content[0]).get("total_iters", 80000)
        if len(content) < total_iters // 2:
            return True
        info = json.loads(content[-1])
        if info[-1]["mode"] == "val":
            info = json.loads[content[-2]]
        return info["iter"] < total_iters - 10


def delete_log(root):
    files = mmcv.scandir(root, ".json", recursive=True)
    for file in files:
        path = osp.join(root, file)
        if json_need_delete(path):
            c = input(f"delete {file}, y/n")
            if c == "y":
                json_path = path
                log_path = path[:-5]
                tf_path = log_path[:-4] + "_tf_logs"
                os.remove(json_path)
                if osp.isfile(log_path):
                    os.remove(log_path)
                if osp.isdir(tf_path):
                    shutil.rmtree(tf_path)


if __name__ == "__main__":
    root = "/home/wzx/weizhixiang/ensegment/work_dirs"
    delete_model(root)
    # delete_log(root)
    # print(json_need_delete('/home/wzx/weizhixiang/ensegment/work_dirs/ugev1.1/20211201_191852.log.json'))
