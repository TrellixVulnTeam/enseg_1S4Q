# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

_base_ = ["./bs8_normal_swin_h256w512_160k_nightcity.py"]
crop_size = (256, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(
                type="RandomMask",
                prob=0.5,
                ratio=0.5,
                patch_size=16,
                mask_mode="token",
            ),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ]
    ),
)









'''###____pretty_text____###'''



'''
False to Parse'''
