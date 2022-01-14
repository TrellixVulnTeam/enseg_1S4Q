import torch
import torch.nn as nn
from enseg.ops.wrappers import resize
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer

from ..builder import DECODE_GEN, NECKS
from .base import BaseGen
from .modules import ResidualBlockWithDropout


@NECKS.register_module()
class UnetNeck(nn.Module):
    # dim; scale factor to img;
    ResNetV1c = {
        "x": [[3, 1], [64, 2], [256, 4], [512, 8], [1024, 8], [2048, 8]],
        "y": [[32, 1], [64, 2], [256, 4], [512, 8], [1024, 16], [2048, 32]],
    }
    SwinTransformer = {
        "x": [[3, 1], [128, 4], [256, 8], [512, 16], [1024, 32]],
        "y": [[32, 1], [64, 4], [128, 8], [256, 16], [512, 32]],
    }

    def __init__(self, arch, conv_cfg, norm_cfg, act_cfg, padding_mode):
        super().__init__()
        self.arch = arch
        self.models = nn.ModuleList()
        for x, y in zip(arch["x"], arch["y"]):
            x_dim, x_factor = x
            y_dim, y_factor = y
            if x_factor == y_factor:
                if x_dim < y_dim:
                    kernel, stride, pad, dilation = 3, 1, 1, 1
                else:
                    kernel, stride, pad, dilation = 1, 1, 0, 1
            elif 2 * x_factor == y_factor:
                kernel, stride, pad, dilation = 4, 2, 1, 1
            elif 4 * x_factor == y_factor:
                kernel, stride, pad, dilation = 4, 4, 0, 1
            if x_dim == 3:
                self.models.append(
                    nn.Sequential(
                        ResidualBlockWithDropout(x_dim, padding_mode, norm_cfg, True),
                        ConvModule(
                            x_dim,
                            y_dim,
                            kernel,
                            stride,
                            pad,
                            dilation,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            padding_mode=padding_mode,
                        ),
                        ResidualBlockWithDropout(y_dim, padding_mode, norm_cfg, True),
                    )
                )
            else:
                self.models.append(
                    ConvModule(
                        x_dim,
                        y_dim,
                        kernel,
                        stride,
                        pad,
                        dilation,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        padding_mode=padding_mode,
                    ),
                )

    def forward(self, features):
        return [m(f) for f, m in zip(features, self.models)]


@DECODE_GEN.register_module()
class UnetGen(BaseGen):
    """
    """

    def __init__(
        self,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        padding_mode="reflect",
        neck_type="ResNetV1c",
        align_corners=False,
        init_cfg=None,
    ):
        super().__init__(
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            init_cfg=init_cfg,
        )
        self.neck = UnetNeck(
            getattr(UnetNeck, neck_type), conv_cfg, norm_cfg, act_cfg, padding_mode
        )
        models = nn.ModuleList()
        fea_dims = [x[0] for x in self.neck.arch["y"]][::-1]
        nc_out = fea_dims[1:] + [fea_dims[-1]]
        nc_in = [fea_dims[0]] + [a + b for a, b in zip(fea_dims[1:], nc_out[:-1])]
        for inc, outc in zip(nc_in, nc_out):
            layer = nn.Sequential(
                ConvModule(
                    inc,
                    outc,
                    1,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    padding_mode=padding_mode,
                ),
                ConvModule(
                    outc,
                    outc,
                    3,
                    1,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    padding_mode=padding_mode,
                ),
            )
            models.append(layer)
        self.models = models
        self.last_layer = nn.Sequential(
            nn.Conv2d(fea_dims[-1], fea_dims[-1], 1),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(fea_dims[-1], 3, 7),
            nn.Tanh(),
        )

    def forward_model(self, inputs):
        """Forward function.
        inputs: size:big->small,channel: less->many
        """
        inputs = self.neck(inputs)
        inputs = inputs[::-1]
        y = None
        for x, layer in zip(inputs, self.models):
            if y is not None:
                y = resize(
                    y,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                x = torch.cat([x, y], 1)
            y = layer(x)
        y = self.last_layer(y)
        return y
