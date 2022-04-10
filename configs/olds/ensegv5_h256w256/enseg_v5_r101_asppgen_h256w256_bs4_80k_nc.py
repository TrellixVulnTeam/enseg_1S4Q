_base_ = ["./enseg_v5_r101_upergen_h256w256_bs4_80k_nc.py"]
network = dict(
    gen=dict(
        _delete_=True,
        type="ASPPGen",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        norm_cfg={{_base_.norm_cfg}},
        act_cfg=dict(type="LeakyReLU"),
        padding_mode="zeros",
        align_corners=False,
    )
)






'''###____pretty_text____###'''



'''
False to Parse'''
