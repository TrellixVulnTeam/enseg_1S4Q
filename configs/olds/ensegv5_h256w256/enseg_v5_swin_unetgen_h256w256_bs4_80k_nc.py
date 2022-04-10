_base_ = [
    "../base/datasets/n_c_h256w256.py",
    "../base/default_runtime.py",
    "../base/models/ensegv5.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    pretrained="/home/wzx/weizhixiang/ensegment/pretrain/swin/mmlab/swin_base_patch4_window7_224.pth",
    backbone=dict(
        pretrain_img_size=224,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
    ),
    seg=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    gen=dict(type="UnetGen", in_channels=[128, 256, 512, 1024]),
)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
opt = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
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
