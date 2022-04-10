from typing import Dict

import torch.nn as nn
from enseg.core import add_prefix
from enseg.models import builder
from mmcv.runner import build_optimizer
from mmcv.runner.fp16_utils import auto_fp16
from torch.nn.parallel.distributed import _find_tensors

from ..builder import NETWORKS, TRANSLATOR, build_loss
from ..segmentors import EncoderDecoder
from enseg.core import add_prefix
import torch


@NETWORKS.register_module()
class UGEV2(EncoderDecoder):
    def __init__(
        self,
        seg,
        gen,
        rec,
        pretrained=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        assert isinstance(seg, dict)
        super().__init__(
            seg["encode"],
            seg["decode"],
            seg.get("aux", None),
            seg.get("neck", None),
            pretrained,
            train_cfg,
            test_cfg,
            init_cfg,
        )
        # self._seg_is_fixed = not seg["need_train"]
        self.gen = builder.build_translator(gen)
        self.k_t = 0
        self.lambda_k = 0.001
        self.gamma = 0.5
        if rec is not None:
            self._rec_accept_img = rec.pop("accept_img")
            self.rec = builder.build_translator(rec)
        else:
            self._rec_accept_img = []

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        dataA = data_batch[0]
        dataB = data_batch[1]
        low_img = dataA["img"]
        light_img = dataB["img"]
        img_metasA = dataA["img_metas"]
        img_metasB = dataB["img_metas"]
        gt_semantic_segA = dataA["gt_semantic_seg"]
        gt_semantic_segB = dataB["gt_semantic_seg"]
        norm_cfgA = img_metasA[0]["img_norm_cfg"]
        norm_cfgB = img_metasB[0]["img_norm_cfg"]
        losses_total = dict()
        visual = {
            "photo/light": light_img,
            "photo/low": low_img,
            "img_metasA": img_metasA,
            "img_metas": img_metasA,
            "img_metasB": img_metasB,
            "seg/origin": low_img,
            "seg/gt": gt_semantic_segA,
        }
        # forward
        # for seg
        self._optim_zero(optimizer, "seg", "backbone", "gen")
        losses_gen, enhanced_img = self.gen.forward_train(low_img, norm_cfgA)
        visual["photo/enhanced"] = enhanced_img
        noise_ehanced_img = enhanced_img + torch.randn_like(enhanced_img) * 0.08
        visual["photo/noise_enhanced"] = noise_ehanced_img
        losses_total.update(add_prefix(losses_gen, "enhanced"))
        losses_seg, seg_logits = self._seg_forward_train(
            self.extract_feat(noise_ehanced_img), img_metasA, gt_semantic_segA
        )
        visual["seg/logits"] = seg_logits
        losses_total.update(losses_seg)
        # for regularizer rec
        losses_rec_enhanced, rec_enhanced = self.rec.forward_train(
            enhanced_img, norm_cfgA
        )
        losses_total.update(add_prefix(losses_rec_enhanced, "gen.enhanced_img"))
        visual["photo/rec_enhanced_img"] = rec_enhanced
        loss_total, vars_total = self._parse_losses(losses_total)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_total))
        loss_total.backward()
        self._optim_step(optimizer, "seg", "backbone", "gen")

        # optimizer dis
        self._optim_zero(optimizer, "rec")
        losses_dis = {}
        if "low" in self._rec_accept_img:
            losses_rec_low, rec_low = self.rec.forward_train(low_img, norm_cfgA)
            losses_dis.update(add_prefix(losses_rec_low, "rec.low"))
            visual["photo/rec_low"] = rec_low
        if "light" in self._rec_accept_img:
            losses_rec_light, rec_light = self.rec.forward_train(light_img, norm_cfgA)
            losses_dis.update(add_prefix(losses_rec_light, "rec.light"))
            visual["photo/rec_light"] = rec_light

        losses_rec_enhanced, rec_enhanced = self.rec.forward_train(
            enhanced_img.detach(), norm_cfgA
        )
        loss_real, _ = self._parse_losses(losses_dis)
        loss_fake, _ = self._parse_losses(losses_rec_enhanced)
        for loss in losses_rec_enhanced.keys():
            losses_rec_enhanced[loss] = -self.k_t * losses_rec_enhanced[loss]
        losses_dis.update(add_prefix(losses_rec_enhanced, "rec.enhanced_img"))
        loss_dis, vars_total = self._parse_losses(losses_dis)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_dis))
        loss_dis.backward()
        self.k_t += self.lambda_k * (
            self.gamma * loss_real.detach() - loss_fake.detach()
        )
        self.k_t = max(min(1, self.k_t), 0)
        self._optim_step(optimizer, "rec")
        losses_total.update(losses_dis)
        loss_total, vars_total = self._parse_losses(losses_total)
        outputs = dict(
            loss=loss_total, log_vars=vars_total, num_samples=len(dataA["img_metas"]),
        )
        outputs["visual"] = visual

        for photo_name in outputs["visual"]:
            if photo_name.find("img_metas") != -1:
                continue
            outputs["visual"][photo_name] = outputs["visual"][photo_name][0].detach()
        return outputs

    @staticmethod
    def _optim_zero(optims, *names):
        for name in names:
            if name in optims:
                optims[name].zero_grad()

    @staticmethod
    def _optim_step(optims, *names):
        for name in names:
            if name in optims:
                optims[name].step()

    def _seg_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logits = self.seg.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg, output_pred=True
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses, seg_logits

    def encode_decode(self, img, img_metas):
        img = self.gen(img, img_metas[0]["img_norm_cfg"])
        return super().encode_decode(img, img_metas)
