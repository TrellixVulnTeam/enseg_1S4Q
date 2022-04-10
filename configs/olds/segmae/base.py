# model settings
backbone_norm_cfg = dict(type="LN", eps=1e-6, requires_grad=True)
norm_cfg = dict(type="SyncBN", requires_grad=True)
network = dict(
    type="EnsegV1",
    pretrained="pretrain/jx_vit_base_p16_224-80ecf9dd.pth",
    backbone=dict(
        type="VisionTransformer",
        img_size=(256, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode="bilinear",
    ),
    gen=dict(
        type="ViTGen",
        img_size=(256, 512),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
    ),
    masker=dict(
        type="Masker",
        ratio=0.75,
        patch_size=16,
        dim=1,
        with_cls_token=True,
        mode="randn",
        mask_value=1,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)








'''###____pretty_text____###'''



'''
backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
network = dict(
    type='EnsegV1',
    pretrained='pretrain/jx_vit_base_p16_224-80ecf9dd.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(256, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bilinear'),
    gen=dict(
        type='ViTGen',
        img_size=(256, 512),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4),
    masker=dict(
        type='Masker',
        ratio=0.75,
        patch_size=16,
        dim=1,
        with_cls_token=True,
        mode='randn',
        mask_value=1),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
'''
