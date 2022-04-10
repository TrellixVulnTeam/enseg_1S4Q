from typing import Callable, Tuple, List, Dict
from attr import has
import torch
from torch import Tensor
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from enseg.ops.wrappers import resize
from ..builder import DECODE_GEN
import torch.nn.functional as F


@DECODE_GEN.register_module()
class Masker(nn.Module):
    mode_choices = ("randn", "token", "constant")

    def __init__(
        self,
        ratio,
        patch_size=16,
        dim=1,
        with_cls_token=False,
        mode="randn",
        mask_value=1,
        norm_pix_loss=False,
        downsample_factor=1,
    ) -> None:
        # 可以考虑用sequence decoder去处理2D模型产生的特征图
        """patch-mask image like mae
        1d w shuffle:
            encode: img->[backbone.embed_mae]->seq->[shuffle_mask_seq]->masked seq->[backbone]
            decode: [decoder.embed_mae]->[recovery_mask_seq]->[recovery_shuffle_seq]->[decoder]->[unpatchify]->img
            loss: [decoder.embed_mae]->[recovery_mask_seq]->[recovery_shuffle_seq]->[decoder]->[loss1d]
        2d w/o shuffle:
            encode: img->[patchify]->[shuffle_mask_seq]->[recovery_mask_seq]->[recovery_shuffle_seq]->[unpatchify]->masked img->[backbone]
            decode: masked img->[decoder]->img->[loss2d]
        # abandon: 训练一个理解shuffle数据的backbone至少在分割任务上看不出来意义
        # 1d+2d w shuffle:
        #     encode: img->[patchify]->[shuffle_mask_seq]->[recovery_mask_seq]->[unpatchify]->masked shuffle img->[backbone]
        #     decode: [decoder.embed_mae]->[recovery_mask_seq]->[recovery_shuffle_seq]->[decoder]->[unpatchify]->img
        #     loss: [decoder.embed_mae]->[recovery_mask_seq]->[recovery_shuffle_seq]->[decoder]->[loss1d]
        Args:
            ratio (float): mask ratio
            patch_size (int, optional): patch size. Defaults to 16.
            shuffle (bool, optional): whether to shuffle patch sequence. Defaults to True.
            dim (int, optional): using 1d patch or 2d patch. Defaults to 1.
            mode (str, optional): what to fill in 2d patch. Defaults to 'randn'.
            mask_value (int, optional): what number to fill in constant patch. Defaults to 1.
        """
        super().__init__()
        self.fp16_enabled = False
        self.ratio = ratio
        self.p = patch_size
        self.dim = dim
        self.with_cls_token = with_cls_token
        self.norm_pix_loss = norm_pix_loss
        self.downsample_factor = downsample_factor

        if with_cls_token and dim == 2:
            raise NotImplementedError(f"with_cls_token {with_cls_token} and dim {dim}")
        if self.dim == 2:
            self.mode = mode
            assert mode in self.mode_choices, f"Not implement for {mode}"
            if self.mode == "constant":
                self.fill_patch = mask_value * torch.ones([patch_size**2])
            if self.mode == "token":
                self.fill_patch = self._get_hadamard(patch_size).view([patch_size**2])

    @torch.no_grad()
    def get_masked_img(self, im: Tensor, mask: Tensor) -> Tensor:
        """get masked img for visualize use ori im and mask

        Args:
            im (Tensor): origin img
            mask (Tensor): mask

        Returns:
            Tensor: masked img
        """
        mask = mask.unsqueeze(-1).repeat(1, 1, self.p**2 * im.shape[1])
        mask = self.unpatchify(mask, im.shape)
        return im * (1 - mask)

    def encode(
        self, encoder: nn.Module, im: Tensor
    ) -> Tuple[List[Tensor], Tensor, Tensor]:
        """mask and encode
        require for encoder: has function
            1d:[embed_mae=lambda im:seq]
            1d:[forward_mae |forward =lambda x:feature]
        1d: img->[backbone.embed_mae]->seq->[shuffle_mask_seq]->masked seq->[backbone]
        2d: img->[patchify]->[shuffle_mask_seq]->[recovery_mask_seq]->[recovery_shuffle_seq]->[unpatchify]->masked img->[backbone]
        Args:
            x (Tensor)[N,L,D]: input patch sequence

        Returns:
            features(List[Tensor]): masked patch sequence,l<L
            mask(Tensor)[N,L]: binary mask, 0 is keep, 1 is remove
            ids_restore(Tenosr)[N,L]: id for restore. using self.unshuffle to use it
        """
        if self.dim == 1:
            seq = encoder.embed_mae(im)
            x, mask, ids_restore = self.shuffle_mask_seq(seq)
        else:
            with torch.no_grad():
                x, mask, ids_restore = self.shuffle_mask_seq(self.patchify(im))
                fill_seq = self._get_fill_seq(im.shape, im.device)
                shuffle_seq = self.recovery_mask_seq(x, fill_seq)
                x = self.unpatchify(
                    self.recovery_shufflle_seq(shuffle_seq, ids_restore), im.shape
                ).detach()
        feature: Tensor
        if hasattr(encoder, "forward_mae"):
            feature = encoder.forward_mae(x)
        else:
            feature = encoder(x)
        return dict(
            feature=feature, mask=mask, ids_restore=ids_restore, masked_img=x, gt_img=im
        )

    # @force_fp32(apply_to=("x_gt", "mask"))

    def decode(
        self,
        decoder: nn.Module,
        criterion: Callable,
        encode_infos: Dict[str, Tensor],
        need_recovery_img=True,
    ) -> Tuple[Tensor, Tensor]:
        """decode
        require for decoder: has function
            1d:[embed_mae=lambda x_masked:embed_x_masked,fill_token]
        1d: [decoder.embed_mae]->[recovery_mask_seq]->[recovery_shuffle_seq]->[decoder]->[unpatchify]->img,loss
        2d: masked img->[decoder]->img,loss2d
        """
        feature_masked = encode_infos["feature"]
        mask = encode_infos["mask"]
        x_gt = encode_infos["gt_img"]
        img_shape = x_gt.shape
        N = x_gt.shape[0]
        ps = self.p // self.downsample_factor
        if self.dim == 1:
            x, fill_token = decoder.embed_mae(feature_masked)
            fill_seq_shape = self._get_fill_seq_shape(img_shape)
            fill_token = fill_token.view([1, 1, -1]).repeat(*fill_seq_shape[:2], 1)
            x_shuffle = self.recovery_mask_seq(x, fill_token)
            x = self.recovery_shufflle_seq(x_shuffle, encode_infos["ids_restore"])
            pred = decoder(x)
            with torch.no_grad():
                target = self.patchify(x_gt, p=ps)
        else:
            x = decoder(feature_masked)
            pred = self.patchify(x, p=ps)
            with torch.no_grad():
                if x_gt.numel() > x.numel():
                    x_gt = resize(
                        x_gt, x.shape[-2:], align_corners=False, mode="bilinear"
                    ).detach()
                target = self.patchify(x_gt, p=ps)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = criterion(pred, target, mask)
        losses = {"mae.loss": loss}
        vars = {"origin_img": self.unpatchify(target, x_gt.shape, p=ps)}
        if need_recovery_img:
            masked_target = target * (1 - mask).unsqueeze(-1)
            vars["rec_img"] = self.unpatchify(
                pred * mask.unsqueeze(-1) + masked_target, x_gt.shape, p=ps
            )
            if self.dim == 1:
                vars["masked_img"] = self.unpatchify(masked_target, x_gt.shape, p=ps)
            else:
                vars["masked_img"] = encode_infos["masked_img"]
        return losses, vars

    def decode_vit_gan(
        self,
        decoder: nn.Module,
        criterion: Callable,
        encode_infos: Dict[str, Tensor],
        need_recovery_img=True,
    ) -> Tuple[Tensor, Tensor]:
        """decode
        require for decoder: has function
            1d:[embed_mae=lambda x_masked:embed_x_masked,fill_token]
        1d: [decoder.embed_mae]->[recovery_mask_seq]->[recovery_shuffle_seq]->[decoder]->[unpatchify]->img,loss
        2d: masked img->[decoder]->img,loss2d
        """
        feature_masked = encode_infos["feature"]
        mask = encode_infos["mask"]
        x_gt = encode_infos["gt_img"]
        img_shape = x_gt.shape
        N = x_gt.shape[0]
        target = x_gt
        ps = self.p // self.downsample_factor
        if self.dim == 1:
            x, fill_token = decoder.embed_mae(feature_masked)
            fill_seq_shape = self._get_fill_seq_shape(img_shape)
            fill_token = fill_token.view([1, 1, -1]).repeat(*fill_seq_shape[:2], 1)
            x_shuffle = self.recovery_mask_seq(x, fill_token)
            x = self.recovery_shufflle_seq(x_shuffle, encode_infos["ids_restore"])
            pred = decoder(x)
            pred = self.unpatchify(pred, x_gt.shape, p=ps)
        else:
            pred = decoder(feature_masked)
        if self.norm_pix_loss:
            mean = target.view(N, -1).mean(dim=-1).view(-1, 1, 1, 1)
            var = target.view(N, -1).var(dim=-1).view(-1, 1, 1, 1)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = criterion(pred, target, mask)
        losses = {"gen.loss": loss}
        vars = {"pred": pred, "target": target, "mask": mask}
        return losses, vars

    def _get_hadamard(self, n: int) -> Tensor:
        k = torch.log2(n)
        h = torch.tensor([[1, 1], [1, -1]])
        res = h
        for i in range(int(k - 1)):
            res = torch.kron(res, h)
        return res

    def _get_masked_L(self, L: int) -> int:
        return int(L * (1 - self.ratio))

    def _get_fill_seq_shape(self, img_shape: torch.Size) -> torch.Size:
        N, C, H, W = img_shape
        D = self.p**2 * C
        L = H * W // (self.p**2)
        masked_L = self._get_masked_L(L)
        fill_seq_shape = [N, L - masked_L, D]
        return fill_seq_shape

    def _get_fill_seq(self, img_shape, device) -> Tensor:
        fill_seq_shape = self._get_fill_seq_shape(img_shape)
        if self.mode == "randn":
            fill_seq = torch.randn(fill_seq_shape, device=device)
        elif self.mode == "token" or self.mode == "constant":
            self.fill_patch = self.fill_patch.to(device)
            fill_seq = self.fill_patch.repeat(
                fill_seq_shape.numel() // self.fill_patch.numel()
            ).view(fill_seq_shape)
        else:
            raise NotImplementedError(str(self.mode))
        return fill_seq

    def patchify(self, imgs: Tensor, shape=None, p=-1) -> Tensor:
        """patchify img to patch sequence

        Args:
            imgs (Tensor)[N,C,H,W]: input image
            shape (torch.Size): shape of imgs

        Returns:
            Tensor[N,L,p**2*C]: patch sequence
        """
        if shape is None:
            shape = imgs.shape
        n, c, h, w = shape
        if p == -1:
            p = self.p
        ph, pw = h // p, w // p
        x = imgs.view((n, c, ph, p, pw, p))
        x: Tensor
        x = torch.einsum("nchpwq->nhwcpq", x).contiguous()
        x = x.view((n, ph * pw, c * p**2))
        return x

    def unpatchify(self, x: Tensor, shape: torch.Size, p=-1) -> Tensor:
        """unpatchify patch sequence to img

        Args:
            x (Tensor)[N,L,p**2*C]: patch sequence
            shape (torch.Size): shape of img

        Returns:
            Tensor[NCHW]: img
        """
        n, c, h, w = shape
        if p == -1:
            p = self.p
        ph, pw = h // p, w // p
        x = x.view((n, ph, pw, c, p, p))
        x = torch.einsum("nhwcpq->nchpwq", x).contiguous()
        imgs = x.view((n, c, h, w))
        return imgs

    def shuffle_mask_seq(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """mae mask: Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffling is done by argsort random noise.
        Args:
            x (Tensor)[N,L,D]: input patch sequence

        Returns:
            Tensor:
            x_masked(Tensor)[N,l,D]: masked patch sequence,l<L
            mask(Tensor)[N,L]: binary mask, 0 is keep, 1 is remove
            ids_restore(Tenosr)[N,L]: id for restore. using self.unshuffle to use it
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = self._get_masked_L(L)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def recovery_mask_seq(self, masked_seq: Tensor, fill_seq: Tensor) -> Tensor:
        """fix masked_seq with fill_seq

        Args:
            masked_seq (Tensor[N,l1,D]): masked seq
            fill_seq (Tensor[N,l2,D]): seq to use for fix masked seq

        Returns:
            Tensor[N,L,D]: recovery sequence,L=l1+l2
        """
        return torch.cat([masked_seq, fill_seq], dim=1)  # fix

    def recovery_shufflle_seq(self, shuffle_seq: Tensor, ids_restore: Tensor) -> Tensor:
        """unshuffle

        Args:
            masked_seq (Tensor[N,l1,D]): masked seq
            ids_restore (Tensor[N,L,D]): ids for unshuffle

        Returns:
            Tensor[N,L,D]: recovery sequence,L=l1+l2
        """
        ids_restore = ids_restore.unsqueeze(-1).repeat(1, 1, shuffle_seq.shape[-1])
        if self.with_cls_token:
            x_ = torch.gather(
                shuffle_seq[:, 1:, :],
                dim=1,
                index=ids_restore,
            )  # unshuffle
            return torch.cat([shuffle_seq[:, :1, :], x_], dim=1)  # append cls token
        else:
            return torch.gather(
                shuffle_seq,
                dim=1,
                index=ids_restore,
            )


if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt

    fig = 1

    def plot(im: Tensor):
        global fig
        plt.figure(fig)
        fig += 1
        im = (im - im.min()) / (im.max() - im.min())
        if im.ndim == 4:
            plt.imshow(im[0].detach().permute(1, 2, 0).numpy())
        else:
            plt.plot(im[0].detach().mean(dim=-1).numpy())

    from timm.models.vision_transformer import PatchEmbed, Block

    def criterion(pred, target, mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    class Encoder1D(nn.Module):
        def __init__(self, embed_mae):
            super().__init__()
            self.embed_mae = masker.patchify
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_mae))

        def forward(self, x):
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            return torch.cat((cls_tokens, x), dim=1)

    class Decoder1D(nn.Module):
        def __init__(self, embed_mae):
            super().__init__()
            self.mask_token = nn.Parameter(torch.zeros([embed_mae]))

        def embed_mae(self, x) -> Tensor:
            return x, self.mask_token

        def forward(self, x):
            return x[:, 1:, :]

    def test1d(ori, masker: Masker):
        encoder = Encoder1D(masker.p * masker.p * ori.shape[1])
        decoder = Decoder1D(masker.p * masker.p * ori.shape[1])
        for param in encoder.parameters():
            assert param.grad is None
        for param in decoder.parameters():
            assert param.grad is None
        x_masked, mask, ids_restore = masker.encode(encoder, ori)
        result, vars = masker.decode(
            decoder, criterion, ori, x_masked, mask, ids_restore, ori.shape
        )
        loss, rec = result["mae.loss"], vars["rec_img"]
        loss.backward()
        assert ori.shape == rec.shape
        result, vars = masker.decode(
            decoder, criterion, ori, x_masked, 1 - mask, ids_restore, ori.shape
        )
        loss, rec = result["mae.loss"], vars["rec_img"]
        assert loss == 0
        for param in encoder.parameters():
            assert param.grad is not None
        for param in decoder.parameters():
            assert param.grad is not None

    class Encoder2D(nn.Module):
        def forward(self, x):
            return x

    class Decoder2D(nn.Module):
        def forward(self, x):
            return x

    def test2d(ori, masker):
        encoder = Encoder2D()
        decoder = Decoder2D()
        for param in encoder.parameters():
            assert param.grad is None
        for param in decoder.parameters():
            assert param.grad is None
        x_masked, mask, ids_restore = masker.encode(encoder, ori)
        result, vars = masker.decode(
            decoder, criterion, ori, x_masked, mask, ids_restore, ori.shape
        )
        loss, rec = result["mae.loss"], vars["rec_img"]
        loss.backward()
        assert ori.shape == rec.shape
        result, vars = masker.decode(
            decoder, criterion, ori, x_masked, 1 - mask, ids_restore, ori.shape
        )
        loss, rec = result["mae.loss"], vars["rec_img"]
        assert loss == 0
        for param in encoder.parameters():
            assert param.grad is not None
        for param in decoder.parameters():
            assert param.grad is not None

    A = 100 * torch.sin(
        torch.linspace(0, 10, 224 * 224).view([1, 1, 224, 224]).repeat([2, 3, 1, 1])
    )
    B = torch.randn(4, 19, 256, 256)
    C = torch.randn(4, 19, 256, 512)
    for i in [A, B, C]:
        masker = Masker(0.5, 16, 1, True, "randn", 1)
        test1d(i, masker)
        masker = Masker(0.5, 16, 2, False, "randn", 1)
        test2d(i, masker)
