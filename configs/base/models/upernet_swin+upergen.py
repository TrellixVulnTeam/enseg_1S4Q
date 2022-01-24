# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
backbone_norm_cfg = dict(type="LN", requires_grad=True)
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
    train_flow=[("sgd", 10)],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
opt = dict(
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
optimizer = dict(backbone=opt, seg=opt, gen=opt, dis=opt,)
