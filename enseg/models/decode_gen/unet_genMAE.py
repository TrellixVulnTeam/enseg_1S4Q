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
class UnetGenMAE(BaseGen):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), upsample_x2_level=0, **kwargs):
        super(UnetGenMAE, self).__init__(input_transform="multiple_select", **kwargs)
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

        self.upsample_x2_level = upsample_x2_level
        if upsample_x2_level:
            upsample_model = []
            in_channels = len(self.in_channels) * self.channels
            for i in range(upsample_x2_level):
                out_channels = max(in_channels // 2, self.channels)
                upsample_model += [
                    ConvModule(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        conv_cfg=dict(type="deconv", output_padding=1),
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                ]
                in_channels = out_channels
            upsample_model += [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=self.img_channels,
                    kernel_size=7,
                    padding=3,
                    bias=True,
                    norm_cfg=None,
                    act_cfg=dict(type="Tanh"),
                )
            ]
            del self.conv_img
            self.upsample_model = nn.Sequential(*upsample_model)
        else:
            self.fpn_bottleneck = ConvModule(
                len(self.in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def output(self, feature):
        x = self.conv_img(feature)
        return x

    def forward(self, features):
        return self.forward_model(features)

    def forward_model(self, features):
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
        if self.upsample_x2_level:
            output = self.upsample_model(fpn_outs)
        else:
            output = self.fpn_bottleneck(fpn_outs)
            output = self.output(output)
        return output
