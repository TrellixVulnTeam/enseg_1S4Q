_base_ = [
    "../base/models/setr_naive.py",
    "../base/datasets/nightcity_h224w224.py",
    "../base/default_runtime.py",
    "../base/schedules/schedule_160k.py",
]
network=dict(
    pretrained='/home/wzx/weizhixiang/ensegment/pretrain/vit/mmlab/jx_vit_large_p16_224-4ee7a4dc.pth'
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
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
opt = dict(
    type="AdamW",
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
)
optimizer = dict(backbone=opt, seg=opt,aux=opt)




















'''###____pretty_text____###'''



'''
False to Parse'''
