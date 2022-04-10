# model settings
_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_160k.py",
]
network = dict(
    type="SegMAE",
    pretrained="/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/large-mae-nightmae599.pth",
    backbone=dict(
        type="VisionTransformer",
        img_size=(256, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.0,
        norm_cfg=dict(type="LN", eps=1e-06, requires_grad=True),
        with_cls_token=True,
        interpolate_mode="bilinear",
    ),
    gen=dict(
        type="ViTGen",
        img_size=(256, 512),
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
    ),
    seg=dict(
        type="SETRUPHead",
        in_channels=1024,
        channels=256,
        in_index=3,
        num_classes=19,
        dropout_ratio=0,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        num_convs=4,
        up_scale=2,
        kernel_size=3,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    aux=[
        dict(
            type="SETRUPHead",
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            num_convs=2,
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
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            num_convs=2,
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
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        ),
    ],
    train_flow=(("M1", 10),),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=(256, 512), stride=(256, 256)),
)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
adamw = dict(
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-7,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
optimizer = dict(backbone=adamw, seg=adamw, aux=adamw, gen=adamw)
# runner=None
# optimizer=adamw
# # fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# # fp16 placeholder
# fp16 = dict()









'''###____pretty_text____###'''



'''
False to Parse'''
