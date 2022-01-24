_base_ = [
    "../base/models/upernet_swin+upergen.py",
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
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
    # gan_loss=dict(loss_weight=10.0),
    seg=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    gen=dict(in_channels=[128, 256, 512, 1024]),
)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
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

# optimizer = dict(
#     gen=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),
#     dis=dict(type="Adam", lr=0.0003, betas=(0.5, 0.999)),
# )
