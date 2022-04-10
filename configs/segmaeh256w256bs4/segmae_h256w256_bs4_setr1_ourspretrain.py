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
    pretrained="/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/large-mae-nightmae599.pth",
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
    train_flow=(("1", 100),),
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
backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='SegMAEfp16',
    pretrained=
    '/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/large-mae-nightmae599.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(256, 256),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bilinear'),
    seg=dict(
        type='SETRUPHead',
        in_channels=1024,
        channels=256,
        in_index=3,
        num_classes=19,
        dropout_ratio=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_convs=4,
        up_scale=2,
        kernel_size=3,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    aux=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_convs=1,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_convs=1,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=19,
            dropout_ratio=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_convs=1,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(171, 171)),
    gen=dict(
        type='ViTGen',
        img_size=(256, 256),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4),
    masker=dict(
        type='Masker',
        ratio=0.75,
        patch_size=16,
        dim=1,
        with_cls_token=True,
        mode='randn',
        mask_value=1,
        downsample_factor=1),
    train_flow=(('1', 100), ))
dataset_type = 'NightCityDataset'
data_root = '/home/wzx/weizhixiang/ensegment/data/enseg/nightcity'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='NightCityDataset',
        data_root='/home/wzx/weizhixiang/ensegment/data/enseg/nightcity',
        img_dir='image/train',
        ann_dir='label/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='NightCityDataset',
        data_root='/home/wzx/weizhixiang/ensegment/data/enseg/nightcity',
        img_dir='image/val',
        ann_dir='label/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='NightCityDataset',
        data_root='/home/wzx/weizhixiang/ensegment/data/enseg/nightcity',
        img_dir='image/test',
        ann_dir='label/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=100,
    hooks=[
        dict(type='CustomizedTextLoggerHook', by_epoch=False),
        dict(type='VisualizationHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
runner = None
find_unused_parameters = False
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
total_iters = 80000
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(
    interval=8000, metric='mIoU', pre_eval=True, save_best='mIoU')
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
fp16 = dict()
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12))
'''
