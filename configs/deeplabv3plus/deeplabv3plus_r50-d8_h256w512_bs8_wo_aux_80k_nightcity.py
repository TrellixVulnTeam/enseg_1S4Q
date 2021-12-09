_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/deeplabv3plus_r50-d8_wo_aux.py",
    "../base/schedules/schedule_80k.py"
]

# data = dict(samples_per_gpu=4, workers_per_gpu=2)