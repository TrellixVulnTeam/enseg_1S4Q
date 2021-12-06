from torch.autograd.grad_mode import F
from .patch import PatchDiscriminator
from .patch_with_gt import PatchDiscriminatorWithGT

__all__ = ["PatchDiscriminator", "PatchDiscriminatorWithGT"]

