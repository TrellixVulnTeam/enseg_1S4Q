from typing import Dict, List
import os
import os.path as osp
import json
import shutil

work_dir = "/home/wzx/weizhixiang/exp_enseg/work_dirs"
rebuild = True
os.chdir(work_dir)
log_dirs = os.listdir(work_dir)
log_dirs = filter(osp.isdir, log_dirs)

for log_dir in log_dirs:
    logs=[
        osp.join(log_dir, file) for file in os.listdir(log_dir) if file[-4:] == ".log"
    ]

    for log_name in logs:
        js=log_name+'.json'
        if not osp.isfile(js):
            os.remove(log_name)
            continue
        need_delete=False
        with open(js, "r") as load_f:
            lines_f = load_f.readlines()
            content = json.loads(lines_f[-1])
            if len(lines_f)<30 or content['iter']<2000:
                need_delete=True
        if need_delete:
            os.remove(js)
            os.remove(log_name)