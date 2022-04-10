from collections import namedtuple

from .base_network import BaseNetwork
import torch.nn as nn
from ..builder import NETWORKS
from mmcv.runner import build_optimizer
from enseg.core import add_prefix
import torch

"""
NOTE
Combination of segmentation network and gan

extractor----->night segmentor
        ------>generator------->discriminator
"""


@NETWORKS.register_module()
class EnsegV2(BaseNetwork):
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
        self.creterion_idt = nn.MSELoss()

    def forward_backbone_train(self, img):
        x, stem = self.backbone(img, True)
        self.featureA = x
        self.all_featureA = [img, stem] + list(x)
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
            {
                "seg/A": img.clone().detach(),
                "seg/logits_A": seg_logits.clone().detach(),
                "seg/gt_A": gt_semantic_seg.clone().detach(),
            },
        )

    def forward_gen_train(self, dataA, dataB):
        imgA = dataA["img"]
        fakeB = self.gen(self.all_featureA, dataA["img_metas"][0]["img_norm_cfg"])
        pred_fake = self.dis(fakeB, dataA["gt_semantic_seg"])
        outputs = {
            "gen/realA": imgA.clone().detach(),
            "gen/fakeB": fakeB.clone().detach(),
        }
        losses = dict()
        loss_adv = self.gan_loss(pred_fake, target_is_real=True, is_disc=False)
        losses["gen.loss_adv"] = loss_adv
        return losses, outputs

    def forward_dis_train(self, dataA, dataB, generated):
        if generated is not None:
            fake_img = generated["gen/fakeB"]
        else:
            with torch.no_grad():
                fake_img = self.gen(
                    self.all_featureA, dataA["img_metas"][0]["img_norm_cfg"]
                ).detach()
        fake_gt = dataA["gt_semantic_seg"]
        real_gt = dataB["gt_semantic_seg"]
        real_img = dataB["img"]
        pred_fake = self.dis(fake_img, fake_gt)
        pred_real = self.dis(real_img, real_gt)
        losses = dict()
        losses["dis.loss_adv"] = self.gan_loss(
            pred_fake, target_is_real=False, is_disc=True
        ) + self.gan_loss(pred_real, target_is_real=True, is_disc=True)
        losses["dis.acc"] = self.gan_loss.get_accuracy(pred_real, pred_fake)
        return losses, None
