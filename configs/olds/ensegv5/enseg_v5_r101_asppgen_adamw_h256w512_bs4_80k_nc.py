_base_ = [
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ensegv5.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(
        _delete_=True,
        type="ResNetV1c",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg={{_base_.norm_cfg}},
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    seg=dict(
        _delete_=True,
        type="DepthwiseSeparableASPPHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg={{_base_.norm_cfg}},
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    gen=dict(
        _delete_=True,
        type="ASPPGen",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        norm_cfg={{_base_.norm_cfg}},
        act_cfg=dict(type="LeakyReLU"),
        padding_mode="zeros",
        align_corners=False,
    ),
)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
opt = dict(
    _delete_=True,
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)
optimizer = dict(backbone=opt, backbone_B=opt, seg=opt, gen=opt)
lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)






























'''###____pretty_text____###'''



'''
False to Parse'''
