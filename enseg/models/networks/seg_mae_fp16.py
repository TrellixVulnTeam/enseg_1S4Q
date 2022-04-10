from collections import namedtuple
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

"""
this class support and only support semantic segmentation
"""


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16_enabled = False

    @force_fp32(apply_to=("pred", "target", "mask"))
    def forward(self, pred, target, mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss





@NETWORKS.register_module()
class SegMAEfp16(BaseNetwork):
    def __init__(
        self,
        backbone,
        seg,
        masker,
        aux=None,
        gen=None,
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
            aux=aux,
            gen=gen,
            neck=neck,
            gan_loss=None,
            pretrained=pretrained,
            train_flow=train_flow or [("s", 10)],
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

        masker = build_decode_gen(masker)
        masker: Masker
        self.masker = masker
        self.criterion = Criterion()
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

    @auto_fp16(apply_to=("img",))
    def my_train_step(self, img, img_metas, gt_semantic_seg):
        # train for network
        work = next(self.train_flow)
        total_losses = {}
        if "1" in work:
            """
            input: img: (N,C,H,W)
            state: features: List[(N,c,h,w)] | List[N,L,D]
            output: seg_logits
            optimizer: backbone,seg
            """
            ## backbone:提取特征
            ## seg:计算分割损失
            features = self._get_features(img)
            losses_seg, seg_logits = self._get_seg_losses(
                features, img_metas, gt_semantic_seg, "seg1"
            )
            total_losses.update(losses_seg)
            # visualize
        if "M" in work:
            """
            input: img: (N,C,H,W), mask_ratio
            state: img_mask: (N,C,H,W) | (N,L,D),L==H*W*mask_ratio
                   last_features_mask
            output: rec_img: (N,C,H,W)
            optimizer: backbone,gen
            """
            ## mask
            ## backbone:提取特征
            ## gen:计算重建损失
            encode_infos = self.masker.encode(self.backbone, img)
            losses_rec, vars_rec = self.masker.decode(self.gen, self.criterion, encode_infos)
            rec_img = vars_rec["rec_img"]
            total_losses.update(losses_rec)
            # visualize
            if "2" in work:
                """
                input: mask_img
                output: seg_logits_mask
                optimizer: backbone,seg
                """
                ## seg:计算分割损失
                # weak TODO: 在生成图中被mask掉的部分，其在计算交叉熵的时候占的权重要低一些，下一阶段应该更低一些
                # NOTE: 在被mask的图片上进行分割大概率不会有提升
                # visualize
            if "3" in work:
                ## backbone:提取特征
                ## seg:计算分割损失
                """
                input: rec_img: (N,C,H,W)
                state: features_rec
                output: seg_logits_rec
                optimizer: backbone,seg,gen
                """

                features_rec = self._get_features(rec_img)
                losses_seg_rec, seg_logits_rec = self._get_seg_losses(
                    features_rec, img_metas, gt_semantic_seg, "seg3"
                )
                total_losses.update(losses_seg_rec)
                # visualize
        total_loss, total_vars = self._parse_losses(total_losses)
        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(img_metas),
            visual={
                "img_metas": img_metas,
                "seg1/input": img,
                "seg1/gt": gt_semantic_seg,
                "seg1/logits": seg_logits,
            },
        )
        if "M" in work:
            visualM = {
                "mae/rec": rec_img,
                "mae/ori": vars_rec["origin_img"],
                "mae/masked": vars_rec["masked_img"],
            }
            outputs["visual"].update(visualM)

        if "3" in work:
            visual3 = {
                "seg3/input": rec_img,
                "seg3/gt": gt_semantic_seg,
                "seg3/logits": seg_logits_rec,
            }
            outputs["visual"].update(visual3)
        for term in outputs["visual"].keys():
            if term.find("img_metas") == -1:
                outputs["visual"][term] = outputs["visual"][term][0].detach()
        return outputs

    def train_step(self, data_batch, optimizer, **kwargs):
        return self.my_train_step(**data_batch)

    def _seg_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logits = self.seg.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg, output_pred=True
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses, seg_logits
