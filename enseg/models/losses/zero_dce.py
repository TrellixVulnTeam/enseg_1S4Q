from torch.functional import norm
from ..builder import LOSSES
import torch
import torch.nn as nn


@LOSSES.register_module()
class ZeroDCELoss(nn.Module):
    name = "ZeroDCELoss"

    def __init__(self, loss_weight=1.0, CCL_params={}, ECL_params={}, SL_params={}):
        # Predefined filters for calculating spatial loss
        super().__init__()
        self.weights_spatial = torch.stack(
            [
                torch.tensor(w, dtype=torch.float32).repeat(3, 1, 1)
                for w in [
                    [[[0, -1 / 3, 0], [0, 1 / 3, 0], [0, 0, 0]]],
                    [[[0, 0, 0], [0, 1 / 3, 0], [0, -1 / 3, 0]]],
                    [[[0, 0, 0], [-1 / 3, 1 / 3, 0], [0, 0, 0]]],
                    [[[0, 0, 0], [0, 1 / 3, -1 / 3], [0, 0, 0]]],
                ]
            ],
            dim=0,
        ).squeeze(0).cuda()
        self.SL_pool = nn.AvgPool2d(4, stride=4)
        self.SL_weight = SL_params.get("weight", 0.1)
        local_size = ECL_params.get("local_size", 16)
        self.ECL_pool = nn.AvgPool2d(local_size, stride=local_size)
        self.ECL_level = ECL_params.get("E", 0.6)
        self.ECL_weight = ECL_params.get("weight", 1.0)
        self.CCL_weight = CCL_params.get("weight", 1.0)

        self.loss_weight = loss_weight

    def spatial_loss(self, i, o):
        i = self.SL_pool(i)
        o = self.SL_pool(o)
        d_i = nn.functional.conv2d(i, self.weights_spatial, padding=1, stride=1)
        d_o = nn.functional.conv2d(o, self.weights_spatial, padding=1, stride=1)
        d = torch.square(torch.abs(d_o) - torch.abs(d_i))
        s = torch.sum(d, dim=1)
        l_spa = torch.mean(s)
        return l_spa

    def exposure_control_loss(self, x, norm_cfg):
        x = self.ECL_pool(x)
        E = ((self.ECL_level * 255 - norm_cfg["mean"]) / norm_cfg["std"]).mean().item()
        x = torch.abs(x - E * torch.ones_like(x))
        x = torch.mean(x)
        return x

    def color_constancy_loss(self, x):
        avg_intensity_channel = torch.mean(x, dim=(2, 3))
        avg_intensity_channel_rolled = torch.roll(avg_intensity_channel, 1, 1)
        d_j = torch.square(
            torch.abs(avg_intensity_channel - avg_intensity_channel_rolled)
        )
        l_col = torch.mean(torch.sum(d_j, dim=1))
        return l_col

    def __call__(self, img, generated, norm_cfg):
        return (
            self.color_constancy_loss(generated) * self.CCL_weight
            + self.exposure_control_loss(generated, norm_cfg) * self.ECL_weight
            + self.spatial_loss(img, generated) * self.SL_weight
        ) * self.loss_weight

