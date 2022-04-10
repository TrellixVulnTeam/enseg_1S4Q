import mmcv
import os
import os.path as osp
import json
import os
import time
import re
import shutil


def remove_by_input(dir_path, prompt="", forbid_delete_hour=72):
    print("\n*******************************\n")
    print(
        f"You will Delete {dir_path} Because {prompt}, press Y/y to confim, and any other key to deny."
    )
    mtime = osp.getmtime(dir_path)
    for root, dirs, files in os.walk(dir_path):
        mtime = max(
            mtime,
            max(osp.getmtime(osp.join(root, f)) for f in dirs + files),
        )
    strtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    print(f"This file modified in {strtime}")
    if forbid_delete_hour:
        if time.time() - mtime < forbid_delete_hour * 3600:
            print(
                f"Deleting files modified within {forbid_delete_hour} hours is not allowed.This file modified in{strtime}"
            )
            return
    print("This is file list of this dir.")
    os.system(f"ls {dir_path}")
    key = input("Press:")
    if key.lower() == "y":
        shutil.rmtree(dir_path)
        print(f"Remove {dir_path} done")
    else:
        print(f"Skip {dir_path} done")


def remove_by_max_iter(work_dirs_path, given_iter=80000):
    for work_dir in os.listdir(work_dirs_path):
        work_dir_path = osp.join(work_dirs_path, work_dir)
        weights = list(
            filter(lambda name: name[-4:] == ".pth", os.listdir(work_dir_path))
        )
        max_iter = (
            max(
                int(res.group())
                for res in map(lambda f: re.search(r"\d+", f), weights)
                if res
            )
            if len(weights) > 0
            else 0
        )
        if max_iter < given_iter:
            remove_by_input(
                work_dir_path, f"This max iter is {max_iter} < {given_iter}"
            )


if __name__ == "__main__":
    remove_by_max_iter("/home/wzx/weizhixiang/ensegment/work_dirs")
