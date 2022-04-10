# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import norm
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from enseg.ops import resize
from .keep_size_module import KeepSizeLayer
from ..builder import DECODE_GEN
from .base import BaseGen
from enseg.models.decode_seg.psp_head import PPM


@DECODE_GEN.register_module()
class UnetGen(BaseGen):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), keep_size=False, **kwargs):
        super(UnetGen, self).__init__(input_transform="multiple_select", **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.keep_size = keep_size
        if keep_size:
            self.keep_size_layer = KeepSizeLayer(
                self.channels,
                self.channels,
                self.conv_cfg,
                self.norm_cfg,
                self.act_cfg,
                self.padding_mode,
                self.align_corners,
                self.init_cfg,
            )

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward_model(self, features, origin_img=None):
        """Forward function."""

        inputs = self._transform_inputs(features)

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        # 1/4,1/8,1/16,1/32
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 2, -1, -1):
            laterals[i] = self.fpn_convs[i](
                laterals[i]
                + resize(
                    laterals[i + 1],
                    size=laterals[i].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
        # build outputs
        fpn_outs = laterals
        # append psp feature

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        if self.keep_size:
            output = self.keep_size_layer(output,origin_img)
        output = self.output(output)
        return output
