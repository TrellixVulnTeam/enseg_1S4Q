# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from enseg.ops import resize
from enseg.models.builder import DECODE_SEG
from enseg.models.decode_gen.base import BaseGen
from enseg.models.decode_seg.psp_head import PPM


@DECODE_SEG.register_module()
class BackGen(BaseGen):
    """
    输入是seg头的输出特征。一般分割头的处理过程为:feature->high dimension feature(BFHW)->seg logits(BCHW)
    我们取高维特征BFHW, 以及输入原始图像，进行图像翻译
    对于3*256*512图像:
    高维特征512*64*128
    原始图像3*256*512
    3*256*512+++++++++++++++++++++++++++++++|->3*256*512
        |->64*128*256++++++++++++++++++|->64*128*256
            |->128*64*128+512*64*128->128*64*128

    """

    def __init__(
        self, base_dim=64, seg_dim=512, act_cfg=dict(type="LeakyReLU"), **kwargs
    ):
        super(BackGen, self).__init__(
            act_cfg=act_cfg, in_channels=[], in_index=[], channels=1, **kwargs
        )
        self.down_layers = nn.ModuleList()
        self.downdeal_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.updeal_layers = nn.ModuleList()
        norm_act_cfg = dict(norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,)
        cur_dim = base_dim
        channels = [
            (3, base_dim),
            (base_dim, base_dim * 2),
        ]
        for depth, channel in enumerate(channels):
            in_ch, out_ch = channel
            downdeal_layer = nn.Sequential(
                ConvModule(in_ch, out_ch // 2, 3, 1, 1, **norm_act_cfg),
                ConvModule(out_ch // 2, out_ch, 3, 1, 1, **norm_act_cfg),
            )
            down_layer = ConvModule(out_ch, out_ch, 4, 2, 1, **norm_act_cfg)
            in_ch = out_ch
            out_ch += out_ch
            self.downdeal_layers.append(downdeal_layer)
            self.down_layers.append(down_layer)
        self.neck = nn.Sequential(
            ConvModule(channels[-1][-1], seg_dim // 2, 3, 1, 1, **norm_act_cfg),
            ConvModule(seg_dim // 2, seg_dim, 3, 1, 1, **norm_act_cfg),
        )
        channels = [
            (seg_dim + seg_dim, base_dim*2),
            (base_dim*2+base_dim*2, base_dim),
            (base_dim + base_dim, base_dim),
        ]
        for depth, channel in enumerate(channels):
            in_ch, out_ch = channel
            updeal_layer = nn.Sequential(
                ConvModule(in_ch, in_ch // 2, 3, 1, 1, **norm_act_cfg),
                ConvModule(in_ch // 2, out_ch, 3, 1, 1, **norm_act_cfg),
            )
            up_layer = ConvModule(
                out_ch, out_ch, 4, 2, 1, **norm_act_cfg, conv_cfg=dict(type="deconv")
            )
            in_ch = out_ch
            out_ch += out_ch
            self.updeal_layers.append(updeal_layer)
            self.up_layers.append(up_layer)
        self.output = nn.Sequential(
            ConvModule(base_dim, base_dim, 7, 1, 3, **norm_act_cfg),
            ConvModule(base_dim, 3, 1, 1, norm_cfg=None, act_cfg=dict(type="Tanh")),
        )

    def forward_model(self, inputs, origin_img):
        """Forward function."""

        seg_feature = inputs
        img = origin_img
        x = img
        downdeal_features = []
        for down_layer, downdeal_layer in zip(self.down_layers, self.downdeal_layers):
            x = downdeal_layer(x)
            downdeal_features.append(x)
            x = down_layer(x)
        x = self.neck(x)
        downdeal_features.append(seg_feature)
        for up_layer, updeal_layer in zip(self.up_layers, self.updeal_layers):
            x = torch.cat([x, downdeal_features.pop()], 1)
            x = updeal_layer(x)
            x = up_layer(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    BackGen()
