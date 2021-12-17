# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="EnsegV1",
    pretrained="open-mmlab://resnet50_v1c",
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
        type="NLHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dropout_ratio=0.1,
        reduction=2,
        use_scale=True,
        mode="embedded_gaussian",
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    aux=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    train_flow=[("s", 10)],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
optimizer = dict(
    backbone=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    seg=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
    aux=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
)
