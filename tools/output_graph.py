from mmcv.utils import Config
import torch.nn as nn
from enseg.models.builder import build_decode_seg

cfg = Config.fromfile(
    "/home/wzx/weizhixiang/ensegment/configs/upergen/upernet_convnext_base+upergen_h256w512_80k_nightcity.py"
)
from enseg.models.builder import build_backbone
from enseg.models.builder import build_decode_seg
import torch
from torch.utils.tensorboard import SummaryWriter
from mmcv.cnn.utils import revert_sync_batchnorm

writer = SummaryWriter("graph/backgen")
x = torch.randn([1, 3, 256, 512]).cuda()
cfg = cfg.network
gen = build_decode_seg(dict(type="BackGen",seg_dim=19)).cuda()
backbone = build_backbone(cfg.backbone).cuda()
seg = build_backbone(cfg.seg).cuda()

backbone.eval()
seg.eval()
gen.eval()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
        self.seg = seg
        self.gen = gen

    def forward(self, x):
        y = self.backbone(x)
        y = self.seg(y)
        y = self.gen(y, x)
        return y


writer.add_graph(Model(), x)
writer.close()

