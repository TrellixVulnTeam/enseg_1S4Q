import mmcv
import os
import os.path as osp

root = "/home/wzx/weizhixiang/ensegment/work_dirs"
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

