# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import os.path as osp
from subprocess import run

from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook
import torch
import torch.nn.functional as F
from enseg.ops.wrappers import resize
from torch import Tensor
from enseg.utils.image_visual import segmap2colormap, de_normalize
from torchvision.utils import make_grid
from typing import Dict


def get_grid(visual: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    nrow = 3
    collected_img = defaultdict(list)
    grids = {}
    for grid_title, imgs_titles in visual.items():
        if "img_metas" in grid_title:
            continue
        if imgs_titles.ndim == 4:
            imgs_titles = imgs_titles[0]
        if "logits" in grid_title:
            img = segmap2colormap(imgs_titles.argmax(dim=0)) / 255.0
        elif "gt" in grid_title:
            img = segmap2colormap(imgs_titles) / 255.0
        else:
            img_metas = (
                visual[f"img_metas{grid_title[-1]}"]
                if f"img_metas{grid_title[-1]}" in visual
                else visual["img_metas"]
            )
            img_norm_cfg = img_metas[0]["img_norm_cfg"]
            img = de_normalize(imgs_titles, img_norm_cfg, div_by_255=True)
        try:
            a, b = grid_title.split("/")
        except:
            a, b = "", grid_title
        collected_img[a].append([img, b])
    for grid_title in collected_img.keys():
        imgs_titles = sorted(collected_img[grid_title], key=lambda x: x[1])
        imgs = list(map(lambda img_title: img_title[0], imgs_titles))
        strs = list(map(lambda img_title: img_title[1], imgs_titles))
        pre_title = [
            " | ".join(strs[idx : idx + nrow]) for idx in range(0, len(strs), nrow)
        ]
        title = "{}: {}".format(grid_title, " ; ".join(pre_title))
        max_h, max_w = max(imgs, key=torch.Tensor.numel).shape[-2:]
        # F.pad(
        #     img,
        #     (0, max_w - img.shape[-1], 0, max_h - img.shape[-2]),
        #     "constant",
        #     255,
        # )
        imgs = [
            F.interpolate(
                img.unsqueeze(0),
                size=[max_h, max_w],
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            for img in imgs
        ]
        grid = make_grid(imgs, nrow)
        grids[title] = grid
    return grids


@HOOKS.register_module()
class VisualizationHook(LoggerHook):
    def __init__(
        self,
        log_dir=None,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        by_epoch=True,
    ):
        super(VisualizationHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch
        )
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(VisualizationHook, self).before_run(runner)
        if TORCH_VERSION == "parrots" or digit_version(TORCH_VERSION) < digit_version(
            "1.1"
        ):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, f"{runner.timestamp}_tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        self.visual(runner)
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    @torch.no_grad()
    def visual(self, runner):
        grids = get_grid(runner.outputs["visual"])
        for title, grid in grids.items():
            self.writer.add_image(title, grid, self.get_iter(runner))
        self.writer.flush()

    @master_only
    def after_run(self, runner):
        self.writer.close()
