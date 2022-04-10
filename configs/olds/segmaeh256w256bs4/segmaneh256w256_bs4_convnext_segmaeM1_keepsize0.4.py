_base_ = [
    "../base/models/upernet_convnext.py",
    "../base/datasets/nightcity_h256w256.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_80k.py",
]
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth"  # noqa
network = dict(
    type="SegMAEfp16",
    backbone=dict(
        type="cls_ConvNeXt",
        arch="base",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    seg=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=19,
        loss_decode=dict(loss_weight=0.4),
    ),
    gen=dict(
        type="UnetGenMAE",
        in_channels=[128, 256, 512, 1024],
        channels=512,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        act_cfg=dict(type="LeakyReLU"),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        padding_mode="zeros",
        align_corners=False,
        upsample_x2_level=2,
    ),
    masker=dict(
        type="Masker",
        ratio=0.75,
        patch_size=16,
        dim=2,
        with_cls_token=False,
        mode="randn",
        mask_value=1,
        downsample_factor=1,
    ),
    train_flow=(("M1", 100),),
    test_cfg=dict(mode="slide", crop_size=(256, 256), stride=(171, 171)),
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

optimizer_config = dict(type="Fp16OptimizerHook", loss_scale="dynamic")
fp16 = dict()
optimizer = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=0.0006,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)
runner = None
data = dict(samples_per_gpu=4, workers_per_gpu=4)




'''###____pretty_text____###'''



'''
False to Parse'''
