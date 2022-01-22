# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    "../base/models/upernet_convnext.py",
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_80k.py",
]
crop_size = (512, 512)

network = dict(
    backbone=dict(
        type="ConvNeXt",
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/home/wzx/weizhixiang/ensegment/pretrain/convnext/mmlab/convnext_base_22k_224.pth",
        ),
    ),
    seg=dict(in_channels=[128, 256, 512, 1024], num_classes=19,),
    # test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
)


lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8, workers_per_gpu=8)

# runner = dict(type="IterBasedRunnerAmp")

# do not use mmdet version fp16
fp16 = None
