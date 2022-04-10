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
    def tanh_normalize(img, norm_cfg):
        # normalize -m/s~(255-m)/s img to -1,1
        mean = torch.tensor(
            norm_cfg["mean"], device=img.device, requires_grad=False
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            norm_cfg["std"], device=img.device, requires_grad=False
        ).view(1, 3, 1, 1)
        a = 2 * std / 255
        b = (2 * mean - 255) / 255
        return a * img + b

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

    def forward(self, img, norm_cfg=None):
        if norm_cfg is None:
            norm_cfg = dict(
                mean=[img[:, i].mean() for i in range(3)],
                std=[img[:, i].std() for i in range(3)],
            )
        norm_img = self.tanh_normalize(img, norm_cfg)
        if hasattr(self, "model"):
            norm_translated = self.model(norm_img)
        else:
            norm_translated = self.decode(self.encode(norm_img))
        return self.tanh_denormalize(norm_translated, norm_cfg)

    def forward_train(self, img, norm_cfg=None, ground_truth=None):
        losses = {}
        generated = self(img, norm_cfg)
        if ground_truth is None:
            ground_truth = img
        if hasattr(self, "creterions"):
            for creterion in self.creterions:
                losses[f"loss_{creterion.name}"] = creterion(
                    generated, ground_truth, norm_cfg
                )
        return losses, generated
