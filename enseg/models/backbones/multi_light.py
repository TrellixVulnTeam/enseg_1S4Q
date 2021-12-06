import warnings

import torch.nn as nn
from mmcv.runner import BaseModule
import torch

from ..builder import BACKBONES
from .resnet import ResNetV1c


@BACKBONES.register_module()
class MultiLight(ResNetV1c):
    def __init__(self, gammas, **kwargs):
        super().__init__(in_channels=3 * (len(gammas) + 3), **kwargs)
        self.gammas = [torch.tensor(g) for g in gammas]

    def forward(self, x, return_stem=False):
        B, C, H, W = x.shape
        multi = [x]
        normalize_x = (x - x.min()) / (x.max() - x.min())
        multi.append(normalize_x)
        for gamma in self.gammas:
            multi.append(torch.pow(normalize_x, gamma))
        multi.append(x.max() - x)
        return super().forward(torch.cat(multi, 1), return_stem)

