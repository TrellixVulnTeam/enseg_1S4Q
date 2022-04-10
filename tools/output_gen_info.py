import mmcv
import argparse
import torch
from mmcv.cnn.utils import revert_sync_batchnorm

# import enseg.models import build_decode_gen,build_backbone
from enseg.models import builder
from enseg.models import build_backbone, build_decode_gen
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config file path",
        default="/home/wzx/weizhixiang/ensegment/configs/ensegv5/enseg_v5_swin_unetgen_adamw_h256w512_bs4_80k_nc.py",
    )
    parser.add_argument(
        "--shape", type=int, help="input tensor shape", default=[256, 512], nargs="+"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    backbone = config["network"]["backbone"]
    gen = config["network"]["gen"]

    print(f"Backbone is {backbone.type}")
    print(f"input tensor shape is {[1,3,*args.shape]}")
    # print(builder.DECODE_GEN.module_dict.keys())
    gen = builder.build_decode_gen(gen)
    gen = revert_sync_batchnorm(gen)

    backbone = build_backbone(backbone)
    backbone = revert_sync_batchnorm(backbone)

    x = torch.randn([2, 3, *args.shape])
    features = backbone(x)
    output = gen(features)
    print(f"{x.shape} -> {output.shape}")


if __name__ == "__main__":
    main()
