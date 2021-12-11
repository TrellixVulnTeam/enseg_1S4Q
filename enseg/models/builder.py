# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry("models", parent=MMCV_MODELS)
ATTENTION = Registry("attention", parent=MMCV_ATTENTION)

BACKBONES = MODELS
DECODE_SEG = MODELS
DECODE_GEN = MODELS
DISCRIMINATORS = MODELS
NETWORKS = MODELS
GANLOSSES = MODELS
TRANSLATOR = MODELS

# not using
NECKS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_translator(cfg):
    return TRANSLATOR.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_decode_seg(cfg):
    """Build head."""
    return DECODE_SEG.build(cfg)


def build_decode_gen(cfg):
    """Build head."""
    return DECODE_GEN.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_discriminator(cfg):
    """Build discriminator"""
    return DISCRIMINATORS.build(cfg)


def build_network(cfg):
    return NETWORKS.build(cfg)


def build_segmentor(cfg):
    return SEGMENTORS.build(cfg)


def build_ganloss(cfg):
    return GANLOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            "train_cfg and test_cfg is deprecated, " "please specify them in model",
            UserWarning,
        )
    assert (
        cfg.get("train_cfg") is None or train_cfg is None
    ), "train_cfg specified in both outer field and model field "
    assert (
        cfg.get("test_cfg") is None or test_cfg is None
    ), "test_cfg specified in both outer field and model field "
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )
