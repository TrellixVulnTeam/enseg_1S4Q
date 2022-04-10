_base_ = [
    "../base/datasets/n_c_h256w512.py",
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
    gen=dict(type='UnetGen',in_channels=[128, 256, 512, 1024]),
)
data = dict(samples_per_gpu=4, workers_per_gpu=4)































'''###____pretty_text____###'''



'''
False to Parse'''
