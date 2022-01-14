_base_ = [
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/autoencoder.py",
    "../base/schedules/schedule_80k.py",
]
data = dict(samples_per_gpu=8, workers_per_gpu=4)
lr_config = dict(policy="poly", power=0.9, min_lr=1e-6, by_epoch=False)
evaluation = None
network = dict(rec=dict(losses_cfg=dict(type="PixelLoss", loss_type="L2")))
