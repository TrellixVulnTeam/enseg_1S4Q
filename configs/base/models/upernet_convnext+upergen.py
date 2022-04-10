# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="EnsegV4",
    gan_loss=dict(
        type="GANLoss",
        gan_type="lsgan",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0,
    ),
    rec_loss=dict(
        type="PixelLoss",
        loss_weight=0.1,
        loss_type="Similar",
        loss_params=dict(alpha=0.8),
    ),
    backbone=dict(
        type="ConvNeXt",
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/home/wzx/weizhixiang/ensegment/pretrain/convnext_base_1k_224.model.pth",
        ),
    ),
    seg=dict(
        type="UPerHead",
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    gen=dict(
        type="UPerGen",
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        norm_cfg=norm_cfg,
        align_corners=True,
    ),
    dis=dict(
        type="PatchDiscriminatorWithGT",
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type="IN"),
        init_cfg=dict(type="normal", gain=0.02),
    ),
    # model training and testing settings
    train_flow=[("sgd", 10)],
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
opt = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)
optimizer = dict(backbone=opt, seg=opt, gen=opt, dis=opt,)































'''###____pretty_text____###'''



'''
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='EnsegV4',
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    rec_loss=dict(
        type='PixelLoss',
        loss_weight=0.1,
        loss_type='Similar',
        loss_params=dict(alpha=0.8)),
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/wzx/weizhixiang/ensegment/pretrain/convnext_base_1k_224.model.pth'
        )),
    seg=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    gen=dict(
        type='UPerGen',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True),
    dis=dict(
        type='PatchDiscriminatorWithGT',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    train_flow=[('sgd', 10)],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
opt = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12))
optimizer = dict(
    backbone=dict(
        constructor='LearningRateDecayOptimizerConstructor',
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        paramwise_cfg=dict(
            decay_rate=0.9, decay_type='stage_wise', num_layers=12)),
    seg=dict(
        constructor='LearningRateDecayOptimizerConstructor',
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        paramwise_cfg=dict(
            decay_rate=0.9, decay_type='stage_wise', num_layers=12)),
    gen=dict(
        constructor='LearningRateDecayOptimizerConstructor',
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        paramwise_cfg=dict(
            decay_rate=0.9, decay_type='stage_wise', num_layers=12)),
    dis=dict(
        constructor='LearningRateDecayOptimizerConstructor',
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        paramwise_cfg=dict(
            decay_rate=0.9, decay_type='stage_wise', num_layers=12)))
'''
