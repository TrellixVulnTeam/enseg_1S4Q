from enseg.core import add_prefix
from enseg.models import builder
from mmcv.runner.fp16_utils import auto_fp16
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import _find_tensors

from ..builder import NETWORKS
from .base_network import BaseNetwork


@NETWORKS.register_module()
class EnsegV6(BaseNetwork):
    def __init__(
        self,
        backbone,
        seg,
        backbone_B=None,
        aux=None,
        gen=None,
        dis=None,
        gan_loss=None,
        rec_loss=None,
        keep_size=False,
        pretrained=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone,
            seg,
            aux=aux,
            gen=None,
            dis=None,
            gan_loss=gan_loss,
            pretrained=pretrained,
            train_flow=None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        self.keep_size = keep_size
        if rec_loss is not None:
            self.creterion_rec = builder.build_loss(rec_loss)
        if backbone_B is None:
            backbone_B = builder.build_backbone(backbone)
        elif backbone_B == "no":
            backbone_B = self.backbone
        else:
            backbone_B = builder.build_backbone(backbone_B)
        self.genA = builder.build_decode_gen(gen)
        self.genB = builder.build_decode_gen(gen)
        self.disA = builder.build_discriminator(dis)
        self.disB = builder.build_discriminator(dis)
        self.backbones = dict(A=self.backbone, B=backbone_B)
        self.backbone_B = backbone_B
        self.feature = {}

    @auto_fp16(apply_to=("img",))
    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        total_losses = dict()
        # train for network
        dataA, dataB = data_batch[0], data_batch[1]
        imgA, imgB = dataA["img"], dataB["img"]
        norm_cfg = dataA["img_metas"][0]["img_norm_cfg"]
        # A2B2A
        self._optim_zero(optimizer, "seg", "backbone", "backbone_B", "genA", "genB")
        featureA = self.backbones["A"](imgA)
        generatedB = self.genB(featureA, imgA, norm_cfg)
        loss_adv_A2B = self.gan_loss(
            self.disB(generatedB), target_is_real=True, is_disc=False
        )
        loss_segA, seg_logits_A = self._seg_forward_train(
            featureA, dataA["img_metas"], dataA["gt_semantic_seg"]
        )
        feature_generatedB = self.backbones["B"](generatedB)
        recoveryA = self.genA(feature_generatedB, generatedB, norm_cfg)
        seg_logits_generatedB = self.seg(feature_generatedB)
        loss_seg_generatedB = F.kl_div(
            seg_logits_generatedB.log_softmax(dim=1),
            seg_logits_A.softmax(dim=1).detach(),
        )
        loss_recA2B2A = self.creterion_rec(recoveryA, imgA, norm_cfg)
        # B2A2B
        featureB = self.backbones["B"](imgB)
        generatedA = self.genA(featureB, imgB, norm_cfg)
        loss_adv_B2A = self.gan_loss(
            self.disA(generatedA), target_is_real=True, is_disc=False
        )
        loss_segB, seg_logits_B = self._seg_forward_train(
            featureB, dataB["img_metas"], dataB["gt_semantic_seg"]
        )
        feature_generatedA = self.backbones["A"](generatedA)
        recoveryB = self.genB(feature_generatedA, generatedA, norm_cfg)
        seg_logits_generatedA = self.seg(feature_generatedA)
        loss_seg_generatedA = F.kl_div(
            seg_logits_generatedA.log_softmax(dim=1),
            seg_logits_B.softmax(dim=1).detach(),
        )
        loss_recB2A2B = self.creterion_rec(recoveryB, imgB, norm_cfg)
        # backward for gen
        losses_gen = {
            "loss_gen.A2B": loss_adv_A2B,
            "loss_rec.A2B2A": loss_recA2B2A,
            "loss_gen.B2A": loss_adv_B2A,
            "loss_rec.B2A2B": loss_recB2A2B,
            "loss_seg.generatedB":loss_seg_generatedB,
            "loss_seg.generatedA":loss_seg_generatedA,
            **add_prefix(loss_segA, "segA"),
            **add_prefix(loss_segB, "segB"),
        }
        loss_gen, _ = self._parse_losses(losses_gen)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))
        loss_gen.backward()
        self._optim_step(optimizer, "seg", "backbone", "backbone_B", "genA", "genB")

        # dis
        self._optim_zero(optimizer, "disA", "disB")
        pred_generatedA = self.disA(generatedA.detach())
        pred_generatedB = self.disA(generatedB.detach())
        pred_realA = self.disB(imgA)
        pred_realB = self.disB(imgB)
        losses_dis = {
            "disA.loss_adv": self.gan_loss(
                pred_generatedA, target_is_real=False, is_disc=True
            )
            + self.gan_loss(pred_realA, target_is_real=True, is_disc=True),
            "disB.loss_adv": self.gan_loss(
                pred_generatedB, target_is_real=False, is_disc=True
            )
            + self.gan_loss(pred_realB, target_is_real=True, is_disc=True),
            "disA.acc": self.gan_loss.get_accuracy(pred_realA, pred_generatedA),
            "disB.acc": self.gan_loss.get_accuracy(pred_realB, pred_generatedB),
        }
        loss_dis, _ = self._parse_losses(losses_dis)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_dis))
        loss_dis.backward()
        self._optim_step(optimizer, "disA", "disB")
        total_losses = {**losses_gen, **losses_dis}
        total_loss, total_vars = self._parse_losses(total_losses)
        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(dataA["img_metas"]),
            visual={
                "img_metasA": dataA["img_metas"],
                "img_metasB": dataB["img_metas"],
                "A2B2A/A": imgA,
                "B2A2B/B": imgB,
                "B2A2B/genA": generatedA,
                "A2B2A/genB": generatedB,
                "A2B2A/recA": recoveryA,
                "B2A2B/recB": recoveryB,
                "segA/A": imgA,
                "segA/logits_A": seg_logits_A,
                "segA/logits_genB": seg_logits_generatedB,
                "segA/gt_A": dataA["gt_semantic_seg"],
                "segB/B": imgB,
                "segB/logits_B": seg_logits_B,
                "segB/logits_genA": seg_logits_generatedA,
                "segB/gt_B": dataB["gt_semantic_seg"],
            },
        )
        for key in outputs["visual"]:
            if key.find("img_metas") == -1:
                outputs["visual"][key] = outputs["visual"][key][0].detach()
        return outputs
