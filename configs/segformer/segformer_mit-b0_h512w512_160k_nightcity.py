_base_ = [
    "../base/datasets/nightcity_h512w512.py",
    "../base/default_runtime.py",
    "../base/models/segformer_mit-b0.py",
    "../base/schedules/schedule_160k.py",
]

model = dict(pretrained="pretrain/mit_b0.pth", decode_head=dict(num_classes=19))

# optimizer

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

data = dict(samples_per_gpu=8, workers_per_gpu=4)
