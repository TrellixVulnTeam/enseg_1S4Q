from turtle import forward
import torch
import torch.nn as nn
import mmcv
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

from enseg.ops import resize


class KeepSizeLayer(BaseModule):
    def __init__(
        self,
        in_channel,
        base_channel=32,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        padding_mode="zeros",
        align_corners=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.align_corners = align_corners
        self.stem_layer = ConvModule(
            3,
            base_channel,
            4,
            2,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            padding_mode=padding_mode,
        )
        self.fit_layer = ConvModule(
            base_channel,
            base_channel,
            1,
            1,
            0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.bottleneck_layer = ConvModule(
            base_channel + in_channel,
            min(base_channel, in_channel),
            3,
            1,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            groups=1,
            padding_mode=padding_mode,
        )
        self.output_layer = ConvModule(
            min(base_channel, in_channel),
            in_channel,
            7,
            1,
            3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            groups=1,
            padding_mode=padding_mode,
        )

    def forward(self, high_feature, img):
        low_feature = self.stem_layer(img)
        low_feature = self.fit_layer(low_feature)
        high_feature = resize(
            high_feature,
            size=low_feature.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        feature = torch.cat([low_feature, high_feature], 1)
        feature = self.bottleneck_layer(feature)
        feature=resize(
            high_feature,
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return self.output_layer(feature)

