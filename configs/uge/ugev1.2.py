_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ugev1.py",
    "../base/schedules/schedule_80k.py"
]

network=dict(
    loss_regular=None,
    rec=None,
)
optimizer = dict(
    _delete_=True,
    backbone=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005),
    seg=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005),
    gen=dict(type="Adam", lr=0.001, betas=(0.5, 0.999)),
    # rec=dict(type="SGD", lr=0.00001),
)