_base_ = [
    "../base/models/upernet_vit-b16_ln_mln.py",
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_80k.py",
]

network = dict(
    pretrained="pretrain/deit_base_patch16_224-b5f2ef4d.pth",
    backbone=dict(drop_path_rate=0.1, final_norm=True),
    neck=None,
    seg=dict(num_classes=19),
    aux=dict(num_classes=19),
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






























'''###____pretty_text____###'''



'''
False to Parse'''
