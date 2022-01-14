_base_ = [
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/autoencoder.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    rec=dict(
        losses_cfg=dict(
            type="PixelLoss",
            loss_weight=1.0,
            loss_type="Similar",
            loss_params=dict(alpha=0.5),
        ),
    )
)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
lr_config = dict(policy="poly", power=0.9, min_lr=1e-6, by_epoch=False)
evaluation = None

