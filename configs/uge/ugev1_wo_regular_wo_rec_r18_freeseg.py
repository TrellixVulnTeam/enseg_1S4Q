_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ugev1.py",
    "../base/schedules/schedule_80k.py",
]
# fixed seg
network = dict(
    gen=dict(losses_cfg=None),
    rec=None,
    seg=dict(
        encode=dict(depth=18, pretrained="pretrain/cityscape_r18_backbone.pth"),
        decode=dict(
            c1_in_channels=64,
            c1_channels=12,
            in_channels=512,
            channels=128,
            init_cfg=dict(checkpoint="pretrain/cityscape_18_decode_head.pth"),
        ),
    ),
)
optimizer = dict(
    _delete_=True,
    # backbone=dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0005),
    seg=dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0005),
    gen=dict(type="Adam", lr=0.001, betas=(0.5, 0.999)),
    # rec=dict(type="SGD", lr=0.0001),
)