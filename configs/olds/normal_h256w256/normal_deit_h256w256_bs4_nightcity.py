_base_ = [
    "../base/models/upernet_vit-b16_ln_mln.py",
    "../base/datasets/nightcity_h256w256.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    pretrained="/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/deit_base_patch16_224-b5f2ef4d.pth",
    seg=dict(num_classes=19),
    backbone=dict(drop_path_rate=0.1),
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
data = dict(samples_per_gpu=4, workers_per_gpu=4)

























'''###____pretty_text____###'''



'''
False to Parse'''
