# model settings
_base_ = [
    "../base/models/setr_naive.py",
    "../base/datasets/nightcity_h256w256.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_80k.py",
]
norm_cfg = {{_base_.norm_cfg}}
network = dict(
    type="SegMAEfp16",
    pretrained="/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/jx_vit_large_p16_224-4ee7a4dc.pth",
    backbone=dict(img_size=(256, 256),drop_rate=0.),
    seg=dict(num_convs=4, up_scale=2, kernel_size=3),
    gen=dict(
        type="ViTGen",
        img_size=(256, 256),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
    ),
    masker=dict(
        type="Masker",
        ratio=0.75,
        patch_size=16,
        dim=1,
        with_cls_token=True,
        mode="randn",
        mask_value=1,
        norm_pix_loss=True,
        downsample_factor=1,
    ),
    aux=[
        dict(
            type="SETRUPHead",
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=1,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="SETRUPHead",
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=1,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
        dict(
            type="SETRUPHead",
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=1,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
    ],
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
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)
runner = None
data = dict(samples_per_gpu=4, workers_per_gpu=4)



'''###____pretty_text____###'''



'''
False to Parse'''
