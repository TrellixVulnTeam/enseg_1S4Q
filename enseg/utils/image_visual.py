from PIL import Image
import cityscapesscripts.helpers.labels as CSLabels
import numpy as np
import torch

"""
Only cityscape format is supported!!!
"""

CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


def trainId2labelId(map_in_trainId: object):
    """Convert trainId to id for cityscapes."""
    map_in_trainId = np.array(map_in_trainId)
    map_in_labelId = np.array(map_in_trainId)
    for trainId, label in CSLabels.trainId2label.items():
        map_in_labelId[map_in_trainId == trainId] = label.id

    return map_in_labelId


def segmap2colormap(segmap: torch.Tensor) -> torch.Tensor:
    if segmap.ndim == 3:
        segmap=segmap.squeeze(0)
    H, W = segmap.shape
    colormap = torch.zeros([H, W, 3], dtype=torch.uint8, device=segmap.device)
    for trainId, color in enumerate(PALETTE):
        colormap[segmap[:, :] == trainId] = torch.tensor(
            color, dtype=torch.uint8, device=segmap.device
        )
    colormap = colormap.permute([2, 0, 1])
    return colormap


def de_normalize(img: torch.Tensor, img_norm_cfg, div_by_255=False) -> torch.Tensor:
    C = img.shape[0]
    std = img_norm_cfg["std"]
    mean = img_norm_cfg["mean"]
    img = (
        img * torch.tensor(std, device=img.device).view(C, 1, 1)
        + torch.tensor(mean, device=img.device).view(C, 1, 1)
    ).clip_(0, 255)
    if div_by_255:
        img = img / 255.0
    return img
