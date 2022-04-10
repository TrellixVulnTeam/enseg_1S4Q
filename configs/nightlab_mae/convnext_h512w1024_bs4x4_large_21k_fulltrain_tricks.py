_base_ = ["./convnext_h512w1024_bs4x4_base_21k_fulltrain_tricks.py"]
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth"  # noqa
network = dict(
    backbone=dict(
        arch="large",
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    seg=dict(
        in_channels=[192, 384, 768, 1536],
    ),
)
