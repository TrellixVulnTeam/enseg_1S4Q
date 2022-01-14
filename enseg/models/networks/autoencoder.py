from typing import Dict
from collections import OrderedDict
import torch.distributed as dist
import mmcv
import torch
import torch.nn as nn
from enseg.core import add_prefix
from enseg.models import builder
from mmcv.runner import BaseModule, auto_fp16, build_optimizer
from mmcv.runner.fp16_utils import auto_fp16
from torch.nn.parallel.distributed import _find_tensors

from ..builder import NETWORKS, TRANSLATOR, build_loss


@NETWORKS.register_module()
class AutoEncoder(BaseModule):
    def __init__(self, rec, noise_std, init_cfg=None):
        super().__init__(init_cfg)
        self._rec_accept_img = rec.pop("accept_img")
        self.rec = builder.build_translator(rec)
        self.noise_std = noise_std

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
        noise_light_img = (
            light_img + torch.randn_like(light_img) * self.noise_std["light"]
        )
        noise_low_img = low_img + torch.randn_like(low_img) * self.noise_std["low"]
        visual = {
            "photo/light": light_img,
            "photo/low": low_img,
            "photo/noise_light": noise_light_img,
            "photo/noise_low": noise_low_img,
            "img_metasA": img_metasA,
            "img_metas": img_metasA,
            "img_metasB": img_metasB,
        }
        # forward
        # optimizer dis
        self._optim_zero(optimizer, "rec")
        losses_dis = {}
        if "low" in self._rec_accept_img:
            losses_rec_low, rec_low = self.rec.forward_train(
                noise_low_img, norm_cfgA, low_img
            )
            losses_dis.update(add_prefix(losses_rec_low, "rec.low"))
            visual["photo/rec_low"] = rec_low
        if "light" in self._rec_accept_img:
            losses_rec_light, rec_light = self.rec.forward_train(
                noise_light_img, norm_cfgA, light_img
            )
            losses_dis.update(add_prefix(losses_rec_light, "rec.light"))
            visual["photo/rec_light"] = rec_light
        loss_dis, vars_total = self._parse_losses(losses_dis)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_dis))
        loss_dis.backward()
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

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
