import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomMask(object):
    def __init__(
        self, prob=0.5, ratio=0.75, patch_size=16, mask_mode="token", mask_value=1
    ) -> None:
        self.prob = prob
        self.ratio = ratio
        self.patch_size = patch_size
        self.mask_mode = mask_mode
        self.mask_value = mask_value

    def get_hadamard(self, n: int) -> np.ndarray:
        k = np.log2(n)
        h = np.array([[1, 1], [1, -1]])
        res = h
        for i in range(int(k - 1)):
            res = np.kron(res, h)
        return res

    def patchify(self, img: np.ndarray, h, w) -> np.ndarray:
        """
        img: (H, W,3)
        x: (L, patch_size**2 *3)
        """
        p = self.patch_size
        assert h % p == 0, f"{h},{p}"
        assert w % p == 0, f"{w},{p}"
        ph, pw = h // p, w // p
        x = img.reshape((ph, p, pw, p, 3))
        x = np.einsum("hpwqc->hwpqc", x)
        x = x.reshape((ph * pw, p**2 * 3))
        return x

    def unpatchify(self, x, h, w):
        """
        x: (L, patch_size**2 *3)
        imgs: (H, W,3)
        """
        p = self.patch_size
        ph, pw = h // p, w // p

        x = x.reshape((ph, pw, p, p, 3))
        x = np.einsum("hwpqc->hpwqc", x)
        imgs = x.reshape((ph * p, pw * p, 3))
        return imgs

    def __call__(self, results):
        if "mask" not in results:
            results["mask"] = np.random.rand() < self.prob
        if results["mask"]:
            results["mask_ratio"] = self.ratio
            results["mask_mode"] = self.mask_mode
            results["mask_value"] = self.mask_value
            img: np.ndarray
            patches: np.ndarray
            img = results["img"]
            h, w = img.shape[:2]
            patches = self.patchify(img, h, w)
            L = patches.shape[0]
            mask = np.random.rand(L) < self.ratio
            mask = mask.reshape(L, 1)
            if self.mask_mode == "token":
                fill_patch = (
                    self.get_hadamard(self.patch_size)
                    .reshape([self.patch_size**2])
                    .repeat(3)
                    / 2
                    + 0.5
                )
            elif self.mask_mode == "constant":
                fill_patch = self.mask_value * np.ones(patches.shape[-1])
            elif self.mask_mode == "randn":
                fill_patch = np.random.randn(*patches.shape)
            # mask_patch:0.0~1.0 mean0.5 std 0.5->(x-0.5)/0.5*s+m=(x-0.5)*2s+m=2sx-s+m
            k = 2 * patches.std(1).reshape(L, 1)
            b = (patches.std(1) + patches.mean(1)).reshape(L, 1)
            denormalized_fill_patch = k * fill_patch + b
            masked_patches = patches * (~mask) + denormalized_fill_patch * mask
            results["img"] = self.unpatchify(masked_patches, h, w).astype(img.dtype)
        return results

