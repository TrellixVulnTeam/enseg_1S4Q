import mmcv
import argparse
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from enseg.models.builder import build_backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--shape", type=int, help="input tensor shape", default=[256, 256], nargs="+"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    if "network" in config:
        backbone = config["network"]["backbone"]
    else:
        backbone = config["backbone"]

    print(f"Backbone is {backbone.type}")
    print(f"input tensor shape is {[1,3,*args.shape]}")
    backbone = build_backbone(backbone)
    backbone = revert_sync_batchnorm(backbone)
    model = backbone.cuda()
    x = torch.randn([1, 3, *args.shape]).cuda()
    features = model(x)
    for idx, f in enumerate(features):
        print(f"* {idx}:{f.shape}")


if __name__ == "__main__":
    main()
