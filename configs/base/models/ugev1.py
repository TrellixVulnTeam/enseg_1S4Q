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
        # need_train=True,
    ),
    gen=dict(
        type="BaseTranslator",
        encode_decode=dict(
            type="UnetGenerator",
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type="IN"),
            use_dropout=True,
            skip=0.0,
            noise_std=0.0,
            init_cfg=dict(type="normal", gain=0.02),
        ),
        losses_cfg=dict(
            type="ZeroDCELoss", loss_weight=10.0, SL_params=dict(weight=2.0)
        ),
    ),
    rec=dict(
        type="BaseTranslator",
        encode_decode=dict(
            type="ResnetGenerator",
            in_channels=3,
            out_channels=3,
            base_channels=64,
            num_down=5,
            norm_cfg=dict(type="IN"),
            use_dropout=False,
            num_blocks=3,
            padding_mode="reflect",
            init_cfg=dict(type="normal", gain=0.02),
        ),
        losses_cfg=dict(type="PixelLoss", loss_weight=1.0, loss_type="L1"),
        accept_img=["low", "light"],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
optimizer = dict(
    backbone=dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0005),
    seg=dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0005),
    gen=dict(type="Adam", lr=0.001, betas=(0.5, 0.999)),
    rec=dict(type="SGD", lr=0.001),
)

