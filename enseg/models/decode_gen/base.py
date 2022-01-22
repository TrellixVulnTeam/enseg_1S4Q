# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from enseg.models.decode_seg.decode_head import BaseDecodeHead
from enseg.models.utils.init_gan import generation_init_weights


class BaseGen(BaseDecodeHead):
    """Base class for BaseGen.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        in_index (int|Sequence[int]): Input feature index
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')        
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self, padding_mode="zeros", **kwargs,
    ):
        super().__init__(
            input_transform="multiple_select", dropout_ratio=0.0,num_classes=1, **kwargs,
        )
        self.padding_mode = padding_mode
        self.fp16_enabled = False
        self.init_type = (
            "normal" if self.init_cfg is None else self.init_cfg.get("type", "normal")
        )
        self.init_gain = (
            0.02 if self.init_cfg is None else self.init_cfg.get("gain", 0.02)
        )

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

    def forward(self, img, norm_cfg=None):
        if norm_cfg is None:
            norm_cfg = dict(
                mean=[img[:, i].mean() for i in range(3)],
                std=[img[:, i].std() for i in range(3)],
            )
        norm_img = img
        # norm_img = self.tanh_normalize(img, norm_cfg)
        norm_translated = self.forward_model(norm_img)
        return self.tanh_denormalize(norm_translated, norm_cfg)

    @abstractmethod
    def forward_model(self, norm_img):
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

    # def init_weights(self):
    #     """Initialize weights for the model.

    #     Args:
    #         pretrained (str, optional): Path for pretrained weights. If given
    #             None, pretrained weights will not be loaded. Default: None.
    #         strict (bool, optional): Whether to allow different params for the
    #             model and checkpoint. Default: True.
    #     """
    #     generation_init_weights(
    #         self, init_type=self.init_type, init_gain=self.init_gain
    #     )

