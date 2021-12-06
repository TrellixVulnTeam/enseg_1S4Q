from mmcv.utils import Config
import torch.nn as nn
from enseg.models.builder import build_decode_seg

cfg = Config.fromfile(
    "/home/wzx/weizhixiang/exp_enseg/configs/base/models/deeplabv3plus_r50-d8.py"
)
from enseg.models import build_backbone

resnet = build_backbone(cfg["network"]["backbone"])
seg = build_decode_seg(cfg["network"]["seg"])
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("graph/seg")
writer.add_graph(nn.Sequential(resnet, seg), torch.randn([1, 3, 385, 385]))
writer.close()

