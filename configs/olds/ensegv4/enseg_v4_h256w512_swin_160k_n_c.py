_base_ = [
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ensegv4_swin.py",
    "../base/schedules/schedule_160k.py",
]

data = dict(samples_per_gpu=4, workers_per_gpu=2)

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
