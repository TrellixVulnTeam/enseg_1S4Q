_base_ = [
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ensegv4.py",
    "../base/schedules/schedule_80k.py",
]

data = dict(samples_per_gpu=8, workers_per_gpu=8)































'''###____pretty_text____###'''



'''
False to Parse'''
