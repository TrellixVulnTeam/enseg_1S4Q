# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    "../base/models/upernet_convnext.py",
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_160k.py",
]
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth"  # noqa
network = dict(
    backbone=dict(
        type="cls_ConvNeXt",
        arch="base",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    seg=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=19,
    ),
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
opt = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
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
optimizer = dict(backbone=opt, seg=opt)






'''###____pretty_text____###'''



'''
False to Parse'''
