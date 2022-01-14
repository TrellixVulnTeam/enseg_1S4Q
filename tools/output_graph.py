from mmcv.utils import Config
import torch.nn as nn
from enseg.models.builder import build_decode_seg

cfg = Config.fromfile(
    "/home/wzx/weizhixiang/ensegment/configs/base/models/upernet_swin.py"
)
from enseg.models.builder import build_backbone
from enseg.models.builder import build_decode_seg
import torch
from torch.utils.tensorboard import SummaryWriter
from mmcv.cnn.utils import revert_sync_batchnorm

writer = SummaryWriter("graph/swin")
x = torch.randn([1, 3, 256, 512]).cuda()
cfg=cfg.network
backbone = build_backbone(cfg.backbone).cuda()
seg = build_decode_seg(cfg.seg).cuda()

backbone.eval()
seg.eval()
feature = backbone(x)
seg_logits = seg(feature)
writer.add_graph(nn.Sequential(backbone,seg), x)
# writer.add_graph(seg, feature)
writer.close()

