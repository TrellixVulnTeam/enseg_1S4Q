_base_ = "./deeplabv3plus_r50-d8.py"
network = dict(
    pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(depth=18),
    seg=dict(c1_in_channels=64, c1_channels=12, in_channels=512, channels=128,),
    aux=dict(in_channels=256, channels=64),
)
