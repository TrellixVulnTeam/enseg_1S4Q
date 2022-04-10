# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="AutoEncoder",
    rec=dict(
        type="BaseTranslator",
        encode_decode=dict(
            type="ResnetGenerator",
            in_channels=3,
            out_channels=3,
            base_channels=64,
            num_down=5,
            norm_cfg=dict(type="IN"),
            use_dropout=False,
            num_blocks=3,
            padding_mode="reflect",
            init_cfg=dict(type="normal", gain=0.02),
        ),
        losses_cfg=dict(type="PixelLoss", loss_weight=1.0, loss_type="Similar"),
        accept_img=["low", "light"],
    ),
    noise_std=dict(light=0.2, low=0.05,),
)
optimizer = dict(rec=dict(type="Adam", lr=0.001, betas=(0.5, 0.999)),)































'''###____pretty_text____###'''



'''
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='AutoEncoder',
    rec=dict(
        type='BaseTranslator',
        encode_decode=dict(
            type='ResnetGenerator',
            in_channels=3,
            out_channels=3,
            base_channels=64,
            num_down=5,
            norm_cfg=dict(type='IN'),
            use_dropout=False,
            num_blocks=3,
            padding_mode='reflect',
            init_cfg=dict(type='normal', gain=0.02)),
        losses_cfg=dict(
            type='PixelLoss', loss_weight=1.0, loss_type='Similar'),
        accept_img=['low', 'light']),
    noise_std=dict(light=0.2, low=0.05))
optimizer = dict(rec=dict(type='Adam', lr=0.001, betas=(0.5, 0.999)))
'''
