from mmcv.utils import Config
import mmcv
import os.path as osp
import os
import argparse
import time
import copy

import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from enseg.utils import parse_args
from enseg.utils import get_available_gpu
from enseg.utils import get_root_logger
from enseg.utils import collect_env
from enseg.utils import set_random_seed
from enseg.datasets import build_dataset


def main():
    args = parse_args()
    args.config = "/home/wzx/weizhixiang/exp_enseg/configs/seg/deeplabv3plus_r50-d8_h256w512_80k_nightcity+clahe.py"
    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join(
        "/home/wzx/weizhixiang/exp_enseg/tests/store",
        'MCIE-1',
    )
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = get_available_gpu([0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    board_file = osp.join(cfg.work_dir, f"{timestamp} board")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    logger.info(f"Config:\n{cfg.pretty_text}")
    board = SummaryWriter(board_file)
    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: " f"{args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    from enseg.utils import segmap2colormap, de_normalize

    for idx, data in enumerate(datasets[0]):
        colormap = segmap2colormap(data["gt_semantic_seg"].data)
        img = de_normalize(data["img"].data, data["img_metas"].data["img_norm_cfg"])
        grid = make_grid(
            [img, data["origin"].data], nrow=2, normalize=True, value_range=(0, 255)
        )
        board.add_image("train", grid, global_step=idx)
        print(idx)
        if idx >= 50:
            break
    board.close()


if __name__ == "__main__":
    main()
