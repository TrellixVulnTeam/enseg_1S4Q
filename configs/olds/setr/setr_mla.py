# model settings
# modify 23
_base_ = ["./setr_naive.py"]
network = dict(
    backbone=dict(
        out_indices=(5, 11, 17, 23),
        with_cls_token=False,
    ),
    neck=dict(
        type="MLANeck",
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        norm_cfg={{_base_.norm_cfg}},
        act_cfg=dict(type="ReLU"),
    ),
    seg=dict(
        _delete_=True,
        type="SETRMLAHead",
        in_channels=(256, 256, 256, 256),
        channels=512,
        in_index=(0, 1, 2, 3),
        dropout_ratio=0,
        mla_channels=128,
        num_classes=19,
        norm_cfg={{_base_.norm_cfg}},
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    aux=[
        dict(
            type="FCNHead",
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
    ],
)





















'''###____pretty_text____###'''



'''
False to Parse'''
