# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="UGEV1",
    seg=dict(
        encode=dict(
            type="ResNetV1c",
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style="pytorch",
            contract_dilation=True,
            pretrained="pretrain/cityscape_backbone.pth",
        ),
        decode=dict(
            type="DepthwiseSeparableASPPHead",
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
            init_cfg=dict(
                type="Pretrained", checkpoint="pretrain/cityscape_decode_head.pth"
            ),
        ),
    ),
    gen=dict(
        type="BaseTranslator",
        encode_decode=dict(
            type="UnetGenerator",
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type="BN"),
            use_dropout=True,
            skip=0.8,
            noise_std=0.0,
            init_cfg=dict(type="normal", gain=0.02),
        ),
    ),
    rec=dict(
        type="BaseTranslator",
        encode_decode=dict(
            type="ResnetGenerator",
            in_channels=3,
            out_channels=3,
            base_channels=32,
            norm_cfg=dict(type="IN"),
            use_dropout=False,
            num_blocks=5,
            padding_mode="reflect",
            init_cfg=dict(type="normal", gain=0.02),
        ),
    ),
    loss_rec=dict(loss_weight=1.0, type="L1",),
    loss_regular=dict(type="ZeroDCELoss", CCL_params=dict(weight=0.1)),
    # train_flow=[("s", 10)],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
optimizer = dict(
    backbone=dict(
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
    ),
    seg=dict(
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
    ),
    gen=dict(type="Adam", lr=0.001, betas=(0.5, 0.999)),
    rec=dict(type="SGD", lr=0.0001),
)

