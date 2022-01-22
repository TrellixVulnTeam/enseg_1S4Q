from typing import List
import torch
import torch.nn as nn
from enseg.ops.wrappers import resize
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer

from ..builder import DECODE_GEN, NECKS, build_neck
from .base import BaseGen
from .modules import ResidualBlockWithDropout
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F

@NECKS.register_module()
class GenNeck(nn.Module):
    # dim; scale factor to img;
    ResNetV1c = {
        "x": [[3, 1], [64, 2], [256, 4], [512, 8], [1024, 8], [2048, 8]],
        "y": [[32, 2], [64, 4], [128, 8], [256, 16], [512, 32], [1024, 64]],
    }
    SwinTransformer = {
        "x": [[3, 1], [128, 4], [256, 8], [512, 16], [1024, 32]],
        "y": [[32, 2], [64, 4], [128, 8], [256, 16], [512, 32]],
    }

    def __init__(
        self, arch, conv_cfg, norm_cfg, act_cfg, padding_mode, accept_origin=True
    ):
        super().__init__()
        self.arch = arch
        self.accept_origin = accept_origin
        if not accept_origin:
            self.arch["x"].pop(0)
            self.arch["y"].pop(0)
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

    def forward(self, features: List[torch.Tensor]):
        if not self.accept_origin:
            features.pop(0)
        return [m(f) for f, m in zip(features, self.models)]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class InvertedBottleneck(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class UnetInvertedBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim
        )  # depthwise conv
        self.norm = LayerNorm(in_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            in_dim, out_dim * 2
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


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
        accept_origin=False,
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
        self.neck = GenNeck(
            getattr(GenNeck, neck_type),
            conv_cfg,
            norm_cfg,
            act_cfg,
            padding_mode,
            accept_origin=accept_origin,
        )
        models = nn.ModuleList()
        fea_dims = [x[0] for x in self.neck.arch["y"]][::-1]
        nc_out = fea_dims[1:] + [fea_dims[-1]]
        nc_in = [fea_dims[0]] + [a + b for a, b in zip(fea_dims[1:], nc_out[:-1])]
        for inc, outc in zip(nc_in, nc_out):
            layer = UnetInvertedBottleneck(inc, outc)
            models.append(layer)
        self.models = models
        self.last_layer = nn.Sequential(
            InvertedBottleneck(fea_dims[-1]),
            ConvModule(
                fea_dims[-1],
                3,
                7,
                padding=3,
                padding_mode=padding_mode,
                norm_cfg=None,
                act_cfg=dict(type="Tanh"),
            ),
        )

    def forward_model(self, inputs):
        """Forward function.
        inputs: size:big->small,channel: less->many
        """
        origin_image = inputs[0]
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

        y = resize(
            y,
            origin_image.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        y = self.last_layer(y)
        return y
