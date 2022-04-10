_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/deeplabv3plus_r50-d8.py",
    "../base/schedules/schedule_80k.py",
]

network = dict(
    backbone=dict(type="MultiLight", gammas=[0.1, 0.3, 0.5, 0.7, 0.9, 1.3, 1.6]),
)





'''###____pretty_text____###'''



'''
False to Parse'''
