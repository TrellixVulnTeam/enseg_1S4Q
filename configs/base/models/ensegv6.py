# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
backbone_norm_cfg = dict(type="LN", requires_grad=True)
network = dict(
    type="EnsegV6",
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
        type="SwinTransformer",
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=backbone_norm_cfg,
    ),
    seg=dict(
        type="UPerHead",
        in_channels=[96, 192, 384, 768],
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
        in_channels=[96, 192, 384, 768],
        channels=512,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        act_cfg=dict(type="LeakyReLU"),
        norm_cfg=norm_cfg,
        padding_mode="zeros",
        align_corners=False,
        keep_size=True,
    ),
    dis=dict(
        type="PatchDiscriminator",
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type="IN"),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
optimizer = dict(
    backbone=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    backbone_B=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    seg=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    genA=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),
    genB=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),
    disA=dict(type="Adam", lr=0.0003, betas=(0.5, 0.999)),
    disB=dict(type="Adam", lr=0.0003, betas=(0.5, 0.999)),
)





'''###____pretty_text____###'''



'''
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
network = dict(
    type='EnsegV6',
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
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True)),
    seg=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
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
        in_channels=[96, 192, 384, 768],
        channels=512,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        act_cfg=dict(type='LeakyReLU'),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        padding_mode='zeros',
        align_corners=False,
        keep_size=True),
    dis=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN')),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer = dict(
    backbone=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    backbone_B=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    seg=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    genA=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999)),
    genB=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999)),
    disA=dict(type='Adam', lr=0.0003, betas=(0.5, 0.999)),
    disB=dict(type='Adam', lr=0.0003, betas=(0.5, 0.999)))
'''
