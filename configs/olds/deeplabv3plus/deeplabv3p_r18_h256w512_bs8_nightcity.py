_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/deeplabv3p_r50.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(depth=18),
    train_flow=[("s", 10)],
)
data = dict(samples_per_gpu=8, workers_per_gpu=4)






























'''###____pretty_text____###'''



'''
False to Parse'''
