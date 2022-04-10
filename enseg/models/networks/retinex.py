from collections import namedtuple

import torch
import torch.nn as nn
from enseg.core import add_prefix
from mmcv.runner import build_optimizer
from torch.nn.parallel.distributed import _find_tensors

from ..builder import NETWORKS
from .base_network import BaseNetwork

"""
this class support and only support semantic segmentation
"""


@NETWORKS.register_module()
class Retinex(BaseNetwork):
    def __init__(
        self,
        backbone,
        seg,
        aux=None,
        gen=None,
        dis=None,
        neck=None,
        gan_loss=None,
        pretrained=None,
        train_flow=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone,
            seg,
            aux=aux,
            gen=gen,
            dis=dis,
            neck=neck,
            gan_loss=None,
            pretrained=pretrained,
            train_flow=train_flow or [("s", 10)],
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

    def forward_backbone_train(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        self.featureA = x
        return x

    def forward_retinex(self, img: torch.Tensor):
        BRG_img = torch.stack((img[:, 2], img[:, 0], img[:, 1]), 1).detach()
        GBR_img = torch.stack((img[:, 1], img[:, 2], img[:, 0]), 1).detach()
        feature_RGB = self.featureA
        feature_BRG = self.backbone(BRG_img)
        feature_GBR = self.backbone(GBR_img)
        loss_retinex = torch.tensor(0.0, requires_grad=True)
        creterion = nn.MSELoss()
        alpha = 2
        k = 0.1
        for rgb, brg, gbr in zip(feature_RGB, feature_BRG, feature_GBR):
            loss_retinex = loss_retinex + k * (
                creterion(rgb, brg) + creterion(rgb, gbr) + creterion(brg, gbr)
            )
            k *= alpha
        losses = {"retinex.loss": loss_retinex}
        return losses

    def forward_seg_train(self, img, img_metas, gt_semantic_seg):
        x = self.featureA
        losses = dict()

        loss_decode, seg_logits = self._seg_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_aux:
            loss_aux = self._aux_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return (
            losses,
            {
                "seg/A": img,
                "seg/logits_A": seg_logits,
                "seg/gt_A": gt_semantic_seg,
            },
        )

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # train for network
        total_losses = {}
        dataA = data_batch
        self.forward_backbone_train(dataA["img"])
        self._optim_zero(optimizer, "seg", "aux", "backbone", "neck")
        losses_seg, outputs_seg = self.forward_seg_train(**dataA)
        loss_seg, seg_vars = self._parse_losses(losses_seg)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_seg))
        loss_seg.backward(retain_graph=True)
        total_losses.update(losses_seg)
        losses_retinex = self.forward_retinex(dataA["img"])
        loss_retinex, retinex_vars = self._parse_losses(losses_retinex)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_retinex))
        loss_retinex.backward()
        self._optim_step(optimizer, "seg", "aux", "backbone", "neck")
        total_losses.update(losses_retinex)
        total_loss, total_vars = self._parse_losses(total_losses)

        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(dataA["img_metas"]),
            visual={"img_metasA": dataA["img_metas"], **outputs_seg},
        )
        return outputs

    def _seg_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logits = self.seg.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg, output_pred=True
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses, seg_logits
