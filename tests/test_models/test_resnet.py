from mmcv.utils import Config

cfg = Config.fromfile(
    "/home/wzx/weizhixiang/exp_enseg/configs/base/models/deeplabv3plus_r50-d8.py"
)
from enseg.models import build_backbone
resnet=build_backbone(cfg['network']['backbone'])
import torch
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('graph')
writer.add_graph(resnet,torch.randn([1,3,385,385]))
writer.close()

