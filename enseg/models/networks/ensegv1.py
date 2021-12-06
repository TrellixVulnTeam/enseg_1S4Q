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
class EnsegV1(BaseNetwork):
    def __init__(
        self,
        backbone,
        seg,
        aux=None,
        gen=None,
        dis=None,
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
            gan_loss=None,
            pretrained=pretrained,
            train_flow=train_flow,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        for flow, epoch in train_flow:
            assert flow == "s"

    def forward_backbone_train(self, img):
        x = self.backbone(img)
        self.featureA = x
        return x

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
            {"seg/A": img, "seg/logits_A": seg_logits, "seg/gt_A": gt_semantic_seg,},
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
        work = next(self.train_flow)
        total_losses = dict()
        # train for network
        dataA = data_batch
        self.forward_backbone_train(dataA["img"])
        outputs_seg, outputs_gen, outputs_dis = None, None, None
        if "s" in work:
            self._optim_zero(optimizer, "seg", "aux", "backbone")
            losses_seg, outputs_seg = self.forward_seg_train(**dataA)
            loss_seg, seg_vars = self._parse_losses(losses_seg)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_seg))
            loss_seg.backward(retain_graph="g" in work or "d" in work)
            # print(loss_seg)
            self._optim_step(optimizer, "seg", "aux")
            total_losses.update(losses_seg)
        self._optim_step(optimizer, "backbone")
        self._optim_zero(optimizer, *optimizer.keys())
        total_loss, total_vars = self._parse_losses(total_losses)

        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(dataA["img_metas"]),
            visual={"img_metasA": dataA["img_metas"]},
        )
        for out in [outputs_gen, outputs_dis, outputs_seg]:
            if out is not None:
                outputs["visual"].update(out)
        return outputs
