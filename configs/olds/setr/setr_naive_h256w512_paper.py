_base_ = [
    "../base/models/setr_naive.py",
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_160k.py",
]
crop_size=(256,512)
network = dict(
    pretrained="/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/jx_vit_large_p16_224-4ee7a4dc.pth",
    backbone=dict(img_size=(256, 512),drop_rate=0.),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(256, 256)),
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
data = dict(samples_per_gpu=8, workers_per_gpu=8)
opt = dict(
    type="SGD",
    lr=0.002,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(custom_keys={"head": dict(lr_mult=10.0)}),
)
optimizer = dict(backbone=opt, seg=opt, aux=opt)





'''###____pretty_text____###'''



'''
False to Parse'''
