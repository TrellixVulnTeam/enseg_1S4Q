_base_ = [
    "../base/datasets/nightcity_h256w512-pmd.py",
    "../base/default_runtime.py",
    "../base/models/ugev1.py",
    "../base/schedules/schedule_160k.py",
]
# fixed seg
network = dict(
    type="UGEPretrain",
    gen=dict(losses_cfg=None),
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
    rec=dict(
        accept_img=["low", "enhanced", "light"],
        losses_cfg=dict(
            type="PixelLoss",
            loss_weight=1.0,
            loss_type="Similar",
            loss_params=dict(alpha=0.5),
        ),
        init_cfg=dict(type="Pretrained", checkpoint="pretrain/rec_in_n_c_similar"),
    ),
    without_grad=('rec','seg','backbone')
)
optimizer = dict(
    _delete_=True,
    # backbone=dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0005),
    # seg=dict(type="SGD", lr=0.000001, momentum=0.9, weight_decay=0.0005),
    gen=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),
    # rec=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),
)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
lr_config = dict(policy="poly", power=0.9, min_lr=1e-6, by_epoch=False)
