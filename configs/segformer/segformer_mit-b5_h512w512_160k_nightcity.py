_base_ = ['./segformer_mit-b0_h512w512_160k_nightcity.py']
# model settings
network = dict(
    pretrained="pretrain/mit_b5.pth",
    backbone=dict(embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    seg=dict(in_channels=[64, 128, 320, 512]),
)
