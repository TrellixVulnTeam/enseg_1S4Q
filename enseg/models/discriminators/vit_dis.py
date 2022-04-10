import torch
import torch.nn as nn
from ..builder import DISCRIMINATORS
from enseg.models.backbones.vit import VisionTransformer


@DISCRIMINATORS.register_module()
class ViTDis(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=...,
        act_cfg=...,
        patch_norm=False,
        final_norm=False,
        interpolate_mode="bicubic",
        num_fcs=2,
        norm_eval=False,
        with_cp=False,
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            embed_dims,
            num_layers,
            num_heads,
            mlp_ratio,
            out_indices,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            with_cls_token,
            output_cls_token,
            norm_cfg,
            act_cfg,
            patch_norm,
            final_norm,
            interpolate_mode,
            num_fcs,
            norm_eval,
            with_cp,
            pretrained,
            init_cfg,
        )

    def forward(self, inputs: torch.Tensor, mask):
        B = inputs.shape[0]
        # stole cls_tokens impl from Phil Wang, thanks
        x, hw_shape = self.patch_embed(inputs)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask = torch.cat((torch.zeros_like(cls_tokens), mask), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)
        L=x.shape[-1]
        x = x[~mask.bool()].view(B,-1,L)
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x[:, 1:]
