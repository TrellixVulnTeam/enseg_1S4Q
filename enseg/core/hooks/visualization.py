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
        visual = runner.outputs["visual"]
        collected = defaultdict(list)
        for k, v in visual.items():
            if "img_metas" in k:
                continue
            if v.ndim == 4:
                v = v[0]
            if "logits" in k:
                img = segmap2colormap(v.argmax(dim=0)) / 255.0
            elif "gt" in k:
                img = segmap2colormap(v) / 255.0
            else:
                img_metas = (
                    visual[f"img_metas{k[-1]}"]
                    if f"img_metas{k[-1]}" in visual
                    else visual["img_metas"]
                )
                img_norm_cfg = img_metas[0]["img_norm_cfg"]
                img = de_normalize(v, img_norm_cfg, div_by_255=True)
            a, b = k.split("/")
            collected[a].append([img, b])
        for k in collected.keys():
            v = sorted(collected[k], key=lambda x: x[1])
            imgs = [vv[0] for vv in v]
            strs = [vv[1] for vv in v]
            title = f"{k}: {' | '.join(strs)}"
            max_shape = max(imgs, key=Tensor.numel).shape[-2:]
            imgs = [
                F.interpolate(
                    img.unsqueeze(0),
                    size=max_shape,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
                if img.shape[-2:] != max_shape
                else img
                for img in imgs
            ]
            self.writer.add_image(title, torch.cat(imgs, 2), self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()
