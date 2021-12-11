from ..builder import LOSSES
import torch
import torch.nn as nn


@LOSSES.register_module()
class ZeroDCELoss:
    def __init__(self, loss_weight=1.0, CCL_params={}, ECL_params={}):
        local_size = ECL_params.get("local_size", 16)
        self.ECL_pool = nn.AvgPool2d(local_size, stride=local_size)
        self.ECL_level = ECL_params.get("E", 0.6)
        self.ECL_weight = ECL_params.get("weight", 1.0)
        self.CCL_weight = CCL_params.get("weight", 1.0)
        self.loss_weight = loss_weight

    def exposure_control_loss(self, x):
        x = self.ECL_pool(x)
        x = torch.abs(x - self.ECL_level * torch.ones_like(x))
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

    def __call__(self, x):
        return (
            self.color_constancy_loss(x) * self.CCL_weight
            + self.exposure_control_loss(x) * self.ECL_weight
        ) * self.loss_weight

