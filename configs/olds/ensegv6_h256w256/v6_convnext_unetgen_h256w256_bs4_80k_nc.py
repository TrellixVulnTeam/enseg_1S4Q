_base_ = [
    "../base/datasets/n_c_h256w256.py",
    "../base/default_runtime.py",
    "../base/models/ensegv6.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    backbone=dict(
        _delete_=True,
        type="ConvNeXt",
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/home/wzx/weizhixiang/ensegment/pretrain/convnext/mmlab/convnext_base_1k_224.pth",
        ),
    ),
    seg=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    gen=dict(type="UnetGen", in_channels=[128, 256, 512, 1024]),
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
adamw = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)
optimizer = dict(
    _delete_=True,
    backbone=adamw,
    backbone_B=adamw,
    seg=adamw,
    genA=adamw,
    genB=adamw,
    disA=adamw,
    disB=adamw,
)
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

# do not use mmdet version fp16
fp16 = None







'''###____pretty_text____###'''



'''
False to Parse'''
