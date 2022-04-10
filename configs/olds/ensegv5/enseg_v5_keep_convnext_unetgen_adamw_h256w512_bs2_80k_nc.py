_base_ = ["./enseg_v5_convnext_unetgen_adamw_h256w512_bs4_80k_nc.py"]
network = dict(
    gen=dict(keep_size=True),
    keep_size=True,
    # backbone_B="no"
)

# optimizer = dict(
#     _delete_=True,
#     backbone={{_base_.adamw}},
#     seg={{_base_.adamw}},
#     gen={{_base_.adamw}},
#     dis=dict(type="Adam", lr=0.0003, betas=(0.5, 0.999)),
# )
data = dict(samples_per_gpu=2, workers_per_gpu=2)






























'''###____pretty_text____###'''



'''
False to Parse'''
