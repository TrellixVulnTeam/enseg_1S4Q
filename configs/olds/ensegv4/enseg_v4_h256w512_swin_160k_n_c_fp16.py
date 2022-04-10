_base_ = [
    "../base/datasets/n_c_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ensegv4_swin.py",
    "../base/schedules/schedule_160k.py",
]
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()
data = dict(samples_per_gpu=4, workers_per_gpu=2)































'''###____pretty_text____###'''



'''
False to Parse'''
