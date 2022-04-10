# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class NuclearLoss(nn.Module):
    def __init__(
        self,
        patch_size=16,
        use_softmax=True,
        inverse=False,
        loss_weight=1.0,
        reduction="mean",
        loss_name="loss_nuclear",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_softmax = use_softmax
        self.loss_weight = loss_weight
        self.inverse = inverse
        self.reduction = reduction
        self._loss_name = loss_name

    def patchify(self, imgs: torch.Tensor, shape, p) -> torch.Tensor:
        """
        img: (N,C,H, W)
        x: (N,L, patch_size**2,C)
        """
        n, c, h, w = shape
        ph, pw = h // p, w // p
        x = imgs.reshape((n, c, ph, p, pw, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape((n, ph * pw, p**2, c))
        return x

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        """Forward function."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        # logits: N,C,H,W
        logits = cls_score
        N, C, H, W = logits.shape
        if self.use_softmax:
            logits = F.softmax(logits, 1)
        if self.patch_size != -1:
            patches = self.patchify(logits, logits.shape, self.patch_size)
            # patches: N*L,pp,C
            patches = patches.view(-1, self.patch_size**2, logits.shape[1])
        else:
            patches = torch.einsum("nchw->nhwc", logits)
            patches = patches.view(N, -1, C)
        if not self.inverse:
            mats = torch.sqrt(torch.bmm(patches.transpose(-1, -2), patches))
        else:
            mats = torch.sqrt(torch.bmm(patches, patches.transpose(-1, -2)))
        loss = -sum(torch.trace(mat) for mat in mats)
        if reduction == "mean":
            loss = loss / mats.shape[0]
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
