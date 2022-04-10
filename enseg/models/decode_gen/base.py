# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from enseg.models.decode_seg.decode_head import BaseDecodeHead
from enseg.models.utils.init_gan import generation_init_weights
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch.nn as nn
from enseg.ops import resize
import torch.nn.functional as F


class BaseGen(BaseModule):
    """Base class for BaseGen.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        in_index (int|Sequence[int]): Input feature index
    """

    def __init__(
        self,
        in_channels,
        channels,
        img_channels=3,
        dropout_ratio=0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        padding_mode="zeros",
        in_index=-1,
        input_transform=None,
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01),
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.img_channels=img_channels
        self.conv_img = nn.Conv2d(channels, img_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.padding_mode = padding_mode
        self.fp16_enabled = False
        self.init_type = (
            "normal" if self.init_cfg is None else self.init_cfg.get("type", "normal")
        )
        self.init_gain = (
            0.02 if self.init_cfg is None else self.init_cfg.get("gain", 0.02)
        )

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def output(self, feature):
        x = self.conv_img(feature)
        x = torch.tanh(x)
        return x

    @staticmethod
    def tanh_normalize(img, norm_cfg):
        # normalize -m/s~(255-m)/s img to -1,1
        if isinstance(img, list):
            device = img[0].device
        else:
            device = img.device
        mean = torch.tensor(norm_cfg["mean"], device=device, requires_grad=False).view(
            1, 3, 1, 1
        )
        std = torch.tensor(norm_cfg["std"], device=device, requires_grad=False).view(
            1, 3, 1, 1
        )
        a = 2 * std / 255
        b = (2 * mean - 255) / 255
        return a * img + b

    @staticmethod
    def tanh_denormalize(img, norm_cfg):
        # normalize -1~1 img to 0~255, then normalize it by (x-mean)/std
        if isinstance(img, list):
            device = img[0].device
        else:
            device = img.device
        mean = torch.tensor(norm_cfg["mean"], device=device, requires_grad=False).view(
            1, 3, 1, 1
        )
        std = torch.tensor(norm_cfg["std"], device=device, requires_grad=False).view(
            1, 3, 1, 1
        )
        a = 127.5 / std
        b = (127.5 - mean) / std
        return a * img + b

    @auto_fp16()
    def forward(self, features, origin_img, norm_cfg):
        if norm_cfg is None:
            norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1],)
        norm_translated = self.forward_model(features, origin_img)
        return self.tanh_denormalize(norm_translated, norm_cfg)

    @abstractmethod
    def forward_model(self, features, origin_img=None):
        ...

    def forward_train(self, img, norm_cfg=None, ground_truth=None):
        losses = {}
        generated = self(img, norm_cfg)
        if ground_truth is None:
            ground_truth = img
        if hasattr(self, "creterions"):
            for creterion in self.creterions:
                losses[f"loss_{creterion.name}"] = creterion(
                    generated, ground_truth, norm_cfg
                )
        return losses, generated

