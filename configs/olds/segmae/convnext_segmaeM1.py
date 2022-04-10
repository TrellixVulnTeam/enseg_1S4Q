_base_ = ["../normal_bs8_160k_h256w512/cls_convnext_h256w512_160k_nightcity.py"]
network = dict(
    type="SegMAEfp16",
    gen=dict(
        type="UnetGenMAE",
        in_channels=[128, 256, 512, 1024],
        channels=512,
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        act_cfg=dict(type="LeakyReLU"),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        padding_mode="zeros",
        align_corners=False,
    ),
    masker=dict(
        type="Masker",
        ratio=0.75,
        patch_size=16,
        dim=2,
        with_cls_token=False,
        mode="randn",
        mask_value=1,
    ),
    train_flow=(("M1", 100),),
)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale="dynamic")
fp16 = dict()
optimizer = dict(
    _delete_=True,
    constructor="LearningRateDecayOptimizerConstructor",
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)
runner = None





'''###____pretty_text____###'''



'''
False to Parse'''
