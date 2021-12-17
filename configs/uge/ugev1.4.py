_base_ = [
    "../base/datasets/nightcity_h256w512.py",
    "../base/default_runtime.py",
    "../base/models/ugev1.py",
    "../base/schedules/schedule_80k.py",
]

network = dict(
    gen=dict(
        encode_decode=dict(
            skip=0,
            # noise_std=0.1,
            noise_std=0.1,
        )
    ),
    loss_rec=dict(loss_weight=10.0, loss_type="L1", rec_low=True),
    loss_regular=None,
)
