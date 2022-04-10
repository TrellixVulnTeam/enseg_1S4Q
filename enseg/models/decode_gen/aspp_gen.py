# Copyright (c) OpenMMLab. All rights reserved.
import imp

import torch
import torch.nn as nn
from enseg.models.decode_seg.psp_head import PPM
from enseg.models.decode_seg.sep_aspp_head import DepthwiseSeparableASPPModule
from enseg.ops import resize
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import DECODE_GEN
from .base import BaseGen


@DECODE_GEN.register_module()
class ASPPGen(BaseGen):
    def __init__(self, c1_in_channels, c1_channels, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPGen, self).__init__(input_transform=None, **kwargs)
        assert isinstance(dilations, (list, tuple))
        assert c1_in_channels >= 0
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward_model(self, inputs, origin_img=None):
        """Forward function."""

        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.output(output)
        return output
