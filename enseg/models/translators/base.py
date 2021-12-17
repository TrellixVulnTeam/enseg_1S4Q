import torch
from mmcv.runner import BaseModule
from torch import nn
from ..builder import TRANSLATOR, BACKBONES, DECODE_GEN, LOSSES
from enseg.core import add_prefix


@TRANSLATOR.register_module()
class BaseTranslator(BaseModule):
    def __init__(
        self,
        encode=None,
        decode=None,
        encode_decode=None,
        losses_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if encode_decode is not None:
            self.model = TRANSLATOR.build(encode_decode)
        else:
            self.encode = BACKBONES.build(encode)
            self.decode = DECODE_GEN.build(decode)
        if losses_cfg is not None:
            if not isinstance(losses_cfg, (list, tuple)):
                losses_cfg = [losses_cfg]
            self.creterions = [LOSSES.build(loss) for loss in losses_cfg]
            for idx, creterion in enumerate(self.creterions):
                if isinstance(creterion, nn.Module):
                    self.creterions[idx] = creterion.cuda()

    @staticmethod
    def tanh_denormalize(img, norm_cfg):
        # normalize -1~1 img to 0~255, then normalize it by (x-mean)/std
        mean = torch.tensor(
            norm_cfg["mean"], device=img.device, requires_grad=False
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            norm_cfg["std"], device=img.device, requires_grad=False
        ).view(1, 3, 1, 1)
        a = 127.5 / std
        b = (127.5 - mean) / std
        return a * img + b

    def forward(self, img, norm_cfg):
        norm_img = (2 * img - img.min() - img.max()) / (img.max() - img.min())
        if hasattr(self, "model"):
            norm_translated = self.model(norm_img, norm_cfg)
        else:
            norm_translated = self.decode(self.encode(norm_img, norm_cfg))
        translated = self.tanh_denormalize(norm_translated, norm_cfg)
        return translated

    def forward_train(self, img, norm_cfg):
        losses = {}
        generated = self(img, norm_cfg)
        if hasattr(self, "creterions"):
            for creterion in self.creterions:
                losses[f"loss_{creterion.name}"] = creterion(img, generated, norm_cfg)
        return losses, generated
