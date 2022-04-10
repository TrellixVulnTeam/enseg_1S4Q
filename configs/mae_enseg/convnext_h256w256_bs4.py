_base_ = [
    "../base/models/upernet_convnext.py",
    "../base/datasets/n_c_h256w256.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_80k.py",
]
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth"  # noqa
network = dict(
    type="MaeEnseg",
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
        upsample_x2_level=2,
    ),
    rec=dict(
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
    dis=dict(
        type="ViTDis",
        img_size=256,
        patch_size=16,
        in_channels=3,
        num_heads=3,
        num_layers=12,
        embed_dims=192,
        drop_rate=0.1,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        norm_eval=False,
        interpolate_mode="bicubic",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        ),
    ),
    masker=dict(
        type="Masker",
        ratio=0.5,
        patch_size=16,
        dim=2,
        with_cls_token=False,
        mode="randn",
        mask_value=1,
        downsample_factor=1,
        norm_pix_loss=True,
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
optimizer = dict(backbone=adamw, seg=adamw, gen=adamw, dis=adamw, rec=adamw)
# optimizer_config = dict(type="Fp16OptimizerHook", loss_scale="dynamic")
# fp16 = dict()
# optimizer = dict(
#     constructor="LearningRateDecayOptimizerConstructor",
#     type="AdamW",
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
# )
# runner = None
data = dict(samples_per_gpu=4, workers_per_gpu=4)



'''###____pretty_text____###'''



'''
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='MaeEnseg',
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
        ]),
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
        align_corners=False,
        upsample_x2_level=2),
    rec=dict(
        type='UnetGenMAE',
        in_channels=[128, 256, 512, 1024],
        channels=512,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        act_cfg=dict(type='LeakyReLU'),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        padding_mode='zeros',
        align_corners=False,
        upsample_x2_level=2),
    dis=dict(
        type='ViTDis',
        img_size=256,
        patch_size=16,
        in_channels=3,
        num_heads=3,
        num_layers=12,
        embed_dims=192,
        drop_rate=0.1,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-06),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
        )),
    masker=dict(
        type='Masker',
        ratio=0.5,
        patch_size=16,
        dim=2,
        with_cls_token=False,
        mode='randn',
        mask_value=1,
        downsample_factor=1,
        norm_pix_loss=True))
dataset_type = 'UnpairedDataset'
data_root = '/home/wzx/weizhixiang/ensegment/data/enseg/nightcity'
aux_type = 'CityscapesDataset'
aux_root = '/home/wzx/weizhixiang/ensegment/data/enseg/cityscape'
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
        type='UnpairedDataset',
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
        ],
        aux_dataset=dict(
            type='CityscapesDataset',
            data_root='/home/wzx/weizhixiang/ensegment/data/enseg/cityscape',
            img_dir='image/train',
            ann_dir='label/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(1024, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 256),
                    cat_max_ratio=0.75),
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
            ])),
    val=dict(
        type='UnpairedDataset',
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
        ],
        aux_dataset=dict(
            type='CityscapesDataset',
            data_root='/home/wzx/weizhixiang/ensegment/data/enseg/cityscape',
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
            ])),
    test=dict(
        type='UnpairedDataset',
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
        ],
        aux_dataset=dict(
            type='CityscapesDataset',
            data_root='/home/wzx/weizhixiang/ensegment/data/enseg/cityscape',
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
            ])))
log_config = dict(
    interval=100,
    hooks=[
        dict(type='CustomizedTextLoggerHook', by_epoch=False),
        dict(type='VisualizationHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)
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
adamw = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer = dict(
    backbone=dict(
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_embed=dict(decay_mult=0.0),
                cls_token=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0)))),
    seg=dict(
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_embed=dict(decay_mult=0.0),
                cls_token=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0)))),
    gen=dict(
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_embed=dict(decay_mult=0.0),
                cls_token=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0)))),
    dis=dict(
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_embed=dict(decay_mult=0.0),
                cls_token=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0)))),
    rec=dict(
        type='AdamW',
        lr=6e-05,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys=dict(
                pos_embed=dict(decay_mult=0.0),
                cls_token=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0)))))
'''
