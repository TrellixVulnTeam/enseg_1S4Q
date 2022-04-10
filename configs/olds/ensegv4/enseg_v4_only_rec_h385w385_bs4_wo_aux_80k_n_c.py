_base_ = [
    "../base/datasets/n_c_h385w385.py",
    "../base/default_runtime.py",
    "../base/models/ensegv4_wo_adv.py",
    "../base/schedules/schedule_80k.py",
]
network = dict(
    gen=dict(
        type="UnetGen",
        act_cfg=dict(type="LeakyReLU"),
        norm_cfg=dict(type="IN"),
        padding_mode="reflect",
        init_cfg=dict(type="normal", gain=0.02),
    ),
    train_flow=[("sg", 10)],
)
optimizer = dict(gen=dict(type="Adam", lr=0.0001, betas=(0.5, 0.999)),)
data = dict(samples_per_gpu=4, workers_per_gpu=2)































'''###____pretty_text____###'''



'''
False to Parse'''
