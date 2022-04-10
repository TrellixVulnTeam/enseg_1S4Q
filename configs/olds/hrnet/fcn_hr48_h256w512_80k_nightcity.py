_base_ = "./fcn_hr18_h256w512_80k_nightcity.py"
network = dict(
    pretrained="open-mmlab://msra/hrnetv2_w48",
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)),
        )
    ),
    seg=dict(in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])),
)






























'''###____pretty_text____###'''



'''
False to Parse'''
