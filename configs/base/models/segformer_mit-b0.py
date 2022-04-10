# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="EnsegV1",
    pretrained=None,
    backbone=dict(
        type="MixVisionTransformer",
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ),
    seg=dict(
        type="SegformerHead",
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    train_flow=[("s", 10)],
    test_cfg=dict(mode="whole"),
)
optimizer = dict(
    backbone=dict(
        type="AdamW",
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                "pos_block": dict(decay_mult=0.0),
                "norm": dict(decay_mult=0.0),
                "head": dict(lr_mult=10.0),
            }
        ),
    ),
    seg=dict(
        type="AdamW",
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                "pos_block": dict(decay_mult=0.0),
                "norm": dict(decay_mult=0.0),
                "head": dict(lr_mult=10.0),
            }
        ),
    ),
)






























'''###____pretty_text____###'''



'''
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='EnsegV1',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    seg=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    train_flow=[('s', 10)],
    test_cfg=dict(mode='whole'))
optimizer = dict(
    backbone=dict(
        type='AdamW',
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_block=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0),
                head=dict(lr_mult=10.0)))),
    seg=dict(
        type='AdamW',
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_block=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0),
                head=dict(lr_mult=10.0)))))
'''
