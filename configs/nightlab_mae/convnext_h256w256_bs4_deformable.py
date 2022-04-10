_base_ = [
    "../base/models/upernet_convnext.py",
    "../base/datasets/nightlab_h256w256.py",
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
        conv_cfg=dict(type="DCN"),
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ],
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
    ),
    masker=dict(
        type="Masker",
        ratio=0.75,
        patch_size=16,
        dim=2,
        with_cls_token=False,
        mode="randn",
        mask_value=1,
        downsample_factor=4,
    ),
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
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='SegMAEfp16',
    pretrained=None,
    backbone=dict(
        type='cls_ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth',
            prefix='backbone.')),
    seg=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ],
        conv_cfg=dict(type='DCN')),
    train_flow=(('1', 100), ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(171, 171)),
    gen=dict(
        type='UnetGenMAE',
        in_channels=[128, 256, 512, 1024],
        channels=512,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        act_cfg=dict(type='LeakyReLU'),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        padding_mode='zeros',
        align_corners=False),
    masker=dict(
        type='Masker',
        ratio=0.75,
        patch_size=16,
        dim=2,
        with_cls_token=False,
        mode='randn',
        mask_value=1,
        downsample_factor=4))
dataset_type = 'NightCityDataset'
data_root = '/home/wzx/weizhixiang/ensegment/data/enseg/nightlab'
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
        data_root='/home/wzx/weizhixiang/ensegment/data/enseg/nightlab',
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
        data_root='/home/wzx/weizhixiang/ensegment/data/enseg/nightlab',
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
        data_root='/home/wzx/weizhixiang/ensegment/data/enseg/nightlab',
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
