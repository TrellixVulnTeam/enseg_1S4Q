from collections import namedtuple

import torch
import torch.nn as nn
from enseg.core import add_prefix
from enseg.models import builder
from mmcv.runner import build_optimizer
from mmcv.runner.fp16_utils import auto_fp16
from torch.nn.parallel.distributed import _find_tensors

from ..builder import NETWORKS
from .base_network import BaseNetwork

"""
NOTE
Combination of segmentation network and gan

extractor----->night segmentor
        ------>generator------->discriminator
"""


@NETWORKS.register_module()
class EnsegV4(BaseNetwork):
    def __init__(
        self,
        backbone,
        seg,
        aux=None,
        gen=None,
        dis=None,
        gan_loss=None,
        rec_loss=None,
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
        if rec_loss is not None:
            self.creterion_rec = builder.build_loss(rec_loss)
        backbone_B = builder.build_backbone(backbone)
        self.backbones = dict(A=self.backbone, B=backbone_B)
        self.backbone_B = backbone_B
        self.feature = {}
        self.all_feature = {}

    def forward_backbone_train(self, img, key="A"):
        # x, stem = self.backbones[key](img, True)
        # all_feature = [img, stem] + list(x)
        # return x, all_feature
        x = self.backbones[key](img)
        all_feature = [img] + list(x)
        return x, all_feature

    def forward_seg_train(self, dataA, dataB):
        losses = dict()
        loss_decode_A, seg_logits_A = self._seg_forward_train(
            self.feature["A"], dataA["img_metas"], dataA["gt_semantic_seg"]
        )
        loss_decode_B, seg_logits_B = self._seg_forward_train(
            self.feature["B"], dataB["img_metas"], dataB["gt_semantic_seg"]
        )
        losses.update(loss_decode_A)
        losses.update({k.replace("decode", "aux"): v for k, v in loss_decode_B.items()})

        return (
            losses,
            {
                "segA/A": dataA["img"].detach(),
                "segA/logits_A": seg_logits_A.detach(),
                "segA/gt_A": dataA["gt_semantic_seg"].detach(),
                "segB/B": dataB["img"].detach(),
                "segB/logits_B": seg_logits_B.detach(),
                "segB/gt_B": dataB["gt_semantic_seg"].detach(),
            },
        )

    def forward_gen_train(self, dataA, dataB):
        norm_cfg = dataA["img_metas"][0]["img_norm_cfg"]
        fake_img = self.gen(self.all_feature["A"], norm_cfg)
        rec_img = self.gen(self.all_feature["B"], norm_cfg)
        real_img = dataB["img"]
        pred_fake = self.dis(fake_img, dataA["gt_semantic_seg"])
        loss_rec = self.creterion_rec(rec_img, real_img, norm_cfg)
        loss_adv = self.gan_loss(pred_fake, target_is_real=True, is_disc=False)

        losses = {
            "gen.loss_adv": loss_adv,
            "gen.loss_idt": loss_rec,
        }
        outputs = {
            "gen/realA": dataA["img"].detach(),
            "gen/fakeB": fake_img.detach(),
            "gen/realB": real_img.detach(),
            "gen/recB": rec_img.detach(),
        }
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

    @auto_fp16(apply_to=("img",))
    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        work = next(self.train_flow)
        total_losses = dict()
        # train for network
        dataA = data_batch[0]
        dataB = data_batch[1]
        self.feature = {}
        self.all_feature = {}
        self.feature["A"], self.all_feature["A"] = self.forward_backbone_train(
            dataA["img"], "A"
        )
        self.feature["B"], self.all_feature["B"] = self.forward_backbone_train(
            dataB["img"], "B"
        )

        outputs_seg, outputs_gen, outputs_dis = None, None, None
        if "s" in work:
            self._optim_zero(optimizer, "seg", "aux", "backbone", "backbone_B")
            losses_seg, outputs_seg = self.forward_seg_train(dataA, dataB)
            loss_seg, seg_vars = self._parse_losses(losses_seg)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_seg))
            loss_seg.backward(retain_graph=True)
            # print(loss_seg)
            self._optim_step(optimizer, "seg", "aux")
            total_losses.update(losses_seg)

        # train for generator
        if "g" in work:
            self._optim_zero(optimizer, "gen")
            losses_gen, outputs_gen = self.forward_gen_train(dataA, dataB)
            loss_gen, gen_vars = self._parse_losses(losses_gen)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))
            loss_gen.backward()
            self._optim_step(optimizer, "gen")
            total_losses.update(losses_gen)
        self._optim_step(optimizer, "backbone", "backbone_B")
        # train for discriminator
        if "d" in work:
            self._optim_zero(optimizer, "dis")
            losses_dis, outputs_dis = self.forward_dis_train(dataA, dataB, outputs_gen)
            loss_dis, dis_vars = self._parse_losses(losses_dis)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_dis))
            loss_dis.backward()
            self._optim_step(optimizer, "dis")
            total_losses.update(losses_dis)
        self._optim_zero(optimizer, *optimizer.keys())
        total_loss, total_vars = self._parse_losses(total_losses)
        self.feature = {}
        self.all_feature = {}
        torch.cuda.empty_cache()
        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(dataA["img_metas"]),
            visual={"img_metasA": dataA["img_metas"], "img_metasB": dataB["img_metas"]},
        )
        for out in [outputs_gen, outputs_dis, outputs_seg]:
            if out is not None:
                outputs["visual"].update(out)
        return outputs
