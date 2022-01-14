from torch.functional import norm
from ..builder import LOSSES
import torch
import torch.nn as nn
import pytorch_msssim

# Multi-Frame GAN Image Enhancementfor Stereo Visual Odometry in Low Light.pdf
class SimilarLoss(nn.Module):
    def __init__(self, alpha=0.6):
        """相似度损失

        Args:
            alpha (float)): 采用ssim的比例，为0时不采用ssim
        """
        super(SimilarLoss, self).__init__()
        self.alpha = alpha
        self.ssim = pytorch_msssim.SSIM()
        self.creterion = nn.L1Loss()

    def forward(self, img1, img2):
        ndim = img1.ndim
        if ndim == 5:
            B, N, C, H, W = img1.shape
            img1 = img1.view(-1, C, H, W)
            img2 = img2.view(-1, C, H, W)
        elif ndim == 3:
            img1.unsqueeze_(0)
            img2.unsqueeze_(0)
        ssim_result = self.alpha * (1 - self.ssim(img1, img2))
        L1_result = (1 - self.alpha) * self.creterion(img1, img2)
        return ssim_result + L1_result


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = pytorch_msssim.SSIM()

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


@LOSSES.register_module()
class PixelLoss:
    def __init__(self, loss_weight=1.0, loss_type="L1", loss_params={}):
        relation = dict(
            L1=nn.L1Loss, L2=nn.MSELoss, SSIM=SSIMLoss, Similar=SimilarLoss,
        )
        self.name = f"PixelLoss_{loss_type}"
        self.creterion = relation[loss_type](**loss_params)
        self.loss_weight = loss_weight

    def __call__(self, generated, ground_truth, norm_cfg):
        return self.creterion(ground_truth, generated) * self.loss_weight
