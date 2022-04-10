from torch import Tensor
from typing import List, Dict
import torch
import torch.nn as nn
from enseg.core import add_prefix
from torch.nn.parallel.distributed import _find_tensors

from enseg.models.builder import build_decode_gen
from ..builder import NETWORKS
from ..decode_gen import Masker, ViTGen
from .base_network import BaseNetwork
from mmcv.runner import auto_fp16, force_fp32
import torch.nn.functional as F
from enseg.models.utils import set_requires_grad


class CriterionRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16_enabled = False

    @force_fp32(apply_to=("pred", "target", "mask"))
    def forward(self, pred, target, mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class CriterionGen(nn.Module):
    def __init__(self, dis: nn.Module):
        super().__init__()
        self.fp16_enabled = False
        self.dis = dis

    @force_fp32(apply_to=("pred", "target", "mask"))
    def forward(self, pred, target, mask):
        pred = self.dis(pred, mask)
        target = self.dis(target, mask)
        loss = F.mse_loss(pred, torch.zeros_like(pred))
        return loss


@NETWORKS.register_module()
class MaeEnseg(BaseNetwork):
    def __init__(
        self,
        backbone,
        seg,
        masker,
        rec,
        gen,
        aux=None,
        dis=None,
        neck=None,
        pretrained=None,
        train_flow=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone,
            seg,
            aux,
            gen,
            dis,
            neck,
            None,
            pretrained,
            train_flow,
            train_cfg,
            test_cfg,
            init_cfg,
        )
        if rec:
            self.rec = build_decode_gen(rec)
        masker = build_decode_gen(masker)
        masker: Masker
        self.masker = masker
        self.criterion_gen = CriterionGen(self.dis)
        self.criterion_rec = CriterionRec()
        self.fp16_enabled = False

    def _get_features(self, img: Tensor) -> Tensor:
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _get_seg_losses(
        self, features: List[Tensor], img_metas, gt_semantic_seg, prefix
    ) -> Tensor:
        losses = dict()
        loss_decode, seg_logits = self.seg.forward_train(
            features, img_metas, gt_semantic_seg, self.train_cfg, output_pred=True
        )
        losses.update(add_prefix(loss_decode, prefix))
        if self.with_aux:
            if isinstance(self.aux, nn.ModuleList):
                for idx, aux_head in enumerate(self.aux):
                    loss_aux = aux_head.forward_train(
                        features, img_metas, gt_semantic_seg, self.train_cfg
                    )
                    losses.update(add_prefix(loss_aux, f"{prefix}_aux_{idx}"))
            else:
                loss_aux = self.aux.forward_train(
                    features, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, "{prefix}_aux"))
        return losses, seg_logits

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        return self.my_train_step(
            data_batch[0]["img"],
            data_batch[0]["img_metas"],
            data_batch[0]["gt_semantic_seg"],
            data_batch[1]["img"],
            data_batch[1]["img_metas"],
            data_batch[1]["gt_semantic_seg"],
            optimizer,
            ddp_reducer,
        )

    @auto_fp16(apply_to=("imgA", "imgB"))
    def my_train_step(
        self, imgA, metaA, gtA, imgB, metaB, gtB, optimizer, ddp_reducer=None
    ):
        work = next(self.train_flow)
        total_losses = {}
        self._optim_zero(optimizer, "seg", "backbone", "gen", "rec")
        set_requires_grad(self.dis, False)
        # if "s" in work:
        featureA = self._get_features(imgA)
        losses_seg, seg_logits = self._get_seg_losses(featureA, metaA, gtA, "seg1")
        total_losses.update(losses_seg)
        # if "r" in work or "g" in work:
        encode_infos = self.masker.encode(self.backbone, imgA)
        # if "r" in work:
        losses_rec, vars_rec = self.masker.decode(
            self.rec, self.criterion_rec, encode_infos
        )
        total_losses.update(losses_rec)
        # if "g" in work:
        encode_infos["gt_img"] = imgB
        losses_gen, vars_gen = self.masker.decode_vit_gan(
            self.gen, self.criterion_gen, encode_infos
        )
        total_losses.update(losses_gen)

        loss_gen, _ = self._parse_losses(total_losses)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))
        loss_gen.backward()
        self._optim_step(optimizer, "seg", "backbone", "gen", "rec")
        # if "d" in work:
        set_requires_grad(self.dis, True)
        self._optim_zero(optimizer, "dis")
        pred = self.dis(vars_gen["pred"].detach().clone(),vars_gen['mask'])
        target = self.dis(vars_gen["target"].detach().clone(),vars_gen['mask'])
        loss_dis = F.mse_loss(pred, torch.zeros_like(pred)) + F.mse_loss(
            target, torch.ones_like(target)
        )
        total_losses["dis.loss"] = loss_dis
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_dis))
        loss_dis.backward()
        self._optim_step(optimizer, "dis")
        total_loss, total_vars = self._parse_losses(total_losses)
        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(metaA),
            visual={
                "img_metas": metaA,
                "segA/input": imgA,
                "segA/gt": gtA,
                "segA/logits": seg_logits,
                "mae/rec": vars_rec["rec_img"],
                "mae/ori": vars_rec["origin_img"],
                "mae/masked": vars_rec["masked_img"],
                "gan/pred": vars_gen["pred"],
                "gan/target": vars_gen["target"],
                "gan/ori": vars_rec["origin_img"],
            },
        )

        for term in outputs["visual"].keys():
            if term.find("img_metas") == -1:
                outputs["visual"][term] = outputs["visual"][term][0].detach()
        return outputs
