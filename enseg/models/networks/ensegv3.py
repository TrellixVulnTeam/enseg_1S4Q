from collections import namedtuple
from torch.nn.parallel.distributed import _find_tensors

from .base_network import BaseNetwork
import torch.nn as nn
from ..builder import NETWORKS
from mmcv.runner import build_optimizer
from enseg.core import add_prefix
from enseg.models import builder
import torch

"""
NOTE
Combination of segmentation network and gan

extractor----->night segmentor
        ------>generator------->discriminator
"""


@NETWORKS.register_module()
class EnsegV3(BaseNetwork):
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
            gan_loss=gan_loss,
            pretrained=pretrained,
            train_flow=train_flow,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        self.mce = nn.MSELoss()
        backbone_B = builder.build_backbone(backbone)
        self.backbones = dict(A=self.backbone, B=backbone_B)
        self.backbone_B = backbone_B
        self.feature = {}
        self.all_feature = {}

    def forward_backbone_train(self, img, key="A"):
        x, stem = self.backbones[key](img, True)
        all_feature = [img, stem] + list(x)
        return x, all_feature

    def forward_seg_train(self, img, img_metas, gt_semantic_seg):
        x = self.feature["A"]
        losses = dict()

        loss_decode, seg_logits = self._seg_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_aux:
            loss_aux = self._aux_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return (
            losses,
            {
                "seg/A": img.clone().detach(),
                "seg/logits_A": seg_logits.clone().detach(),
                "seg/gt_A": gt_semantic_seg.clone().detach(),
            },
        )

    def forward_gen_train(self, dataA, dataB):
        fakeB = self.gen(self.all_feature["A"], dataA["img_metas"][0]["img_norm_cfg"])
        feature_fakeB, _ = self.forward_backbone_train(fakeB, "B")
        loss_ce_fakeB, logits_fakeB = self._seg_forward_train(
            feature_fakeB, dataA["img_metas"], dataA["gt_semantic_seg"]
        )
        loss_ce_realB, logits_realB = self._seg_forward_train(
            self.feature["B"], dataB["img_metas"], dataB["gt_semantic_seg"]
        )
        loss_mce = self.mce(logits_fakeB, logits_realB.detach())

        losses = {
            "gen.loss_ce_A": loss_ce_fakeB["decode.loss_ce"],
            "gen.acc_seg_A": loss_ce_fakeB["decode.acc_seg"],
            "gen.loss_ce_B": loss_ce_realB["decode.loss_ce"],
            "gen.acc_seg_B": loss_ce_realB["decode.acc_seg"],
            "gen.loss_mce": loss_mce,
        }
        outputs = {
            "gen/realA": dataA["img"].detach(),
            "gen/fakeB": fakeB.detach(),
            "gen/realB": dataB["img"].detach(),
            "gen/logits_fakeB": logits_fakeB.detach(),
            "gen/logits_realB": logits_realB.detach(),
        }
        return losses, outputs

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        total_losses = dict()
        # train for network
        dataA = data_batch[0]
        dataB = data_batch[1]
        self.feature = {}
        self.all_feature = {}
        self.feature["A"], self.all_feature["A"] = self.forward_backbone_train(
            dataA["img"], "A"
        )
        self.feature["B"], _ = self.forward_backbone_train(dataB["img"], "B")

        self._optim_zero(optimizer, *optimizer.keys())

        losses_seg, outputs_seg = self.forward_seg_train(**dataA)
        total_losses.update(losses_seg)
        losses_gen, outputs_gen = self.forward_gen_train(dataA, dataB)
        total_losses.update(losses_gen)
        total_loss, total_vars = self._parse_losses(total_losses)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(total_loss))
        total_loss.backward()
        self._optim_step(optimizer, *optimizer.keys())

        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(dataA["img_metas"]),
            visual={"img_metasA": dataA["img_metas"], "img_metasB": dataB["img_metas"]},
        )
        for out in [outputs_gen, outputs_seg]:
            if out is not None:
                outputs["visual"].update(out)
        return outputs
