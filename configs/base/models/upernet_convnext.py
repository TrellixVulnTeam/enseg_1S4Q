# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="EnsegV1",
    pretrained=None,
    backbone=dict(
        type='cls_ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
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
    # model training and testing settings
    train_flow=[("s", 10)],
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
# opt = dict(
#     constructor="LearningRateDecayOptimizerConstructor",
#     type="AdamW",
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
# )
# optimizer = dict(backbone=opt, seg=opt)































'''###____pretty_text____###'''



'''
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='EnsegV1',
    pretrained=None,
    backbone=dict(
        type='cls_ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
            prefix='backbone.')),
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
    train_flow=[('s', 10)],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
'''
