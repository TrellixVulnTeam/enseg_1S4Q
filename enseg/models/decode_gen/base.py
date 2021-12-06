# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from enseg.core import build_pixel_sampler
from enseg.models.utils.init_gan import generation_init_weights
from enseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseGen(BaseModule, metaclass=ABCMeta):
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
        self,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        padding_mode="reflect",
        align_corners=False,
        init_cfg=None,
    ):
        super(BaseGen, self).__init__(init_cfg)
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.padding_mode = padding_mode

        self.align_corners = align_corners

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
        self.init_type = (
            "normal" if init_cfg is None else init_cfg.get("type", "normal")
        )
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get("gain", 0.02)

    def init_weights(self):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        generation_init_weights(
            self, init_type=self.init_type, init_gain=self.init_gain
        )

