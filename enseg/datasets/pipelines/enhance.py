import numpy as np
from ..builder import PIPELINES


class EqualizeHist:
    def __init__(self, tgt_img: np.ndarray, limit=0.02):
        """
        Get color map from given RGB image
        :param tgt_img: target image
        :param limit: limit of density
        """
        self._limit = limit
        self._color_map = [self.get_color_map(tgt_img[:, :, i]) for i in range(3)]

    def get_color_map(self, img: np.ndarray):
        assert img.dtype == np.uint8
        # get shape
        h, w = img.shape
        num_pixels = h * w
        # get hist
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        limit_pixels = int(num_pixels * self._limit)
        # get number of overflow and clip
        num_overflow = np.sum(np.clip(hist - limit_pixels, a_min=0, a_max=None))
        hist = np.clip(hist, a_min=0, a_max=limit_pixels)
        # add
        hist += np.round(num_overflow / 256.0).astype(np.int)
        # get cdf
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
        # return
        return cdf

    def __call__(self, img: np.ndarray):
        chs = [self._color_map[i][img[:, :, i]] for i in range(3)]
        return np.stack(chs, axis=-1)


import torch, mmcv


@PIPELINES.register_module()
class MCIE(object):
    def __init__(self, clip_limit=0.2):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit

    def __call__(self, results):
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results["origin"] = torch.tensor(mmcv.rgb2bgr(results["img"].copy())).permute(
            2, 0, 1
        )
        equ_hist = EqualizeHist(results["img"], self.clip_limit)
        results["img"] = equ_hist(results["img"])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(clip_limit={self.clip_limit}"
        return repr_str
