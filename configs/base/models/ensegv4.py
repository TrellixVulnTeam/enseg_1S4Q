# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type="EnsegV4",
    pretrained="open-mmlab://resnet50_v1c",
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
    ),
    seg=dict(
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
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    gen=dict(
        type="UnetGen",
        act_cfg=dict(type="LeakyReLU"),
        norm_cfg=dict(type="IN"),
        padding_mode="reflect",
        init_cfg=dict(type="normal", gain=0.02),
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
optimizer = dict(
    backbone=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    backbone_B=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    seg=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    gen=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),
    dis=dict(type="Adam", lr=0.0003, betas=(0.5, 0.999)),
)

