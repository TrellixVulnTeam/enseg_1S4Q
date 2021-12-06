import torch.nn as nn
import mmcv
from enseg.models.builder import NETWORKS


@NETWORKS.register_module()
class DemoNetwork(nn.Module):
    def __init__(self, backbone, seg, aux, gen, dis=None):
        super().__init__()
        self.model = nn.ModuleDict(
            dict(backbone=backbone, seg=seg, aux=aux, gen=gen, dis=dis)
        )

    def forward(self, in_img):
        with mmcv.Timer(   
            print_tmpl="Forward total takes {:.1f} seconds" + "\n" + "*" * 15
        ):
            with mmcv.Timer():
                feature = self.model["backbone"](in_img)
                print("features", [f.shape for f in feature])
            with mmcv.Timer():
                seg_img = self.model["seg"](feature)
                print("segmap", seg_img.shape)
            with mmcv.Timer():
                gen_img = self.model["gen"](list(feature) + [in_img])
                print("generated", gen_img.shape)
            with mmcv.Timer():
                aux_img = self.model["aux"](feature)
                print("aux_map", aux_img.shape)
            with mmcv.Timer():
                score_img = self.model["dis"](gen_img)
                print("score_map", score_img.shape)
        return seg_img

    def forward_step(self, in_img):
        self.forward(in_img)
