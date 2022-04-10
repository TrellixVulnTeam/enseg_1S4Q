from typing import Iterable
from enseg.core.utils.misc import add_prefix
from enseg.models.builder import (
    build_backbone,
    build_decode_gen,
    build_decode_seg,
    build_discriminator,
    build_ganloss,
)
from enseg.models.segmentors import EncoderDecoder
from torch.nn.parallel.distributed import _find_tensors


class BaseNetwork(EncoderDecoder):
    """Base class for segmentors."""

    def __init__(
        self,
        backbone,
        seg,
        aux=None,
        gen=None,
        dis=None,
        neck=None,
        gan_loss=None,
        pretrained=None,
        train_flow=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ):
        super(BaseNetwork, self).__init__(
            backbone,
            seg,
            aux,
            neck,
            pretrained,
            train_cfg,
            test_cfg,
            init_cfg,
        )
        self.train_flow = self.get_train_flow(train_flow)
        self.fp16_enabled = False        
        if gen:
            self.gen = build_decode_gen(gen)
        if dis:
            self.dis = build_discriminator(dis)
        if gan_loss is not None:
            self.gan_loss = build_ganloss(gan_loss)
        else:
            self.gan_loss = None

    @property
    def with_seg(self):
        """bool: whether the network has seg"""
        return hasattr(self, "seg") and self.seg is not None

    @property
    def with_aux(self):
        """bool: whether the netwrok has aux"""
        return hasattr(self, "aux") and self.aux is not None

    @property
    def with_gen(self):
        """bool: whether the netwrok has gen"""
        return hasattr(self, "gen") and self.gen is not None

    @property
    def with_dis(self):
        """bool: whether the netwrok has dis"""
        return hasattr(self, "dis") and self.dis is not None

    # placeholder
    def forward_gen_train(self, dataA, dataB):
        return None, None

    # placeholder
    def forward_dis_train(self, dataA, dataB, generated):
        return None, None

    def forward_backbone_train(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        self.featureA = x
        return x

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        work = next(self.train_flow)
        total_losses = dict()
        # train for network
        dataA = data_batch[0]
        dataB = data_batch[1]
        self.forward_backbone_train(dataA["img"])
        outputs_seg, outputs_gen, outputs_dis = None, None, None
        if "s" in work:
            self._optim_zero(optimizer, "seg", "aux", "backbone")
            losses_seg, outputs_seg = self.forward_seg_train(**dataA)
            loss_seg, seg_vars = self._parse_losses(losses_seg)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_seg))
            loss_seg.backward(retain_graph="g" in work or "d" in work)
            # print(loss_seg)
            self._optim_step(optimizer, "seg", "aux")
            total_losses.update(losses_seg)

        # train for generator
        if "g" in work:
            self._optim_zero(optimizer, "gen")
            losses_gen, outputs_gen = self.forward_gen_train(dataA, dataB)
            loss_gen, gen_vars = self._parse_losses(losses_gen)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))
            loss_gen.backward()
            self._optim_step(optimizer, "gen")
            total_losses.update(losses_gen)
        self._optim_step(optimizer, "backbone")
        # train for discriminator
        if "d" in work:
            self._optim_zero(optimizer, "dis")
            losses_dis, outputs_dis = self.forward_dis_train(dataA, dataB, outputs_gen)
            loss_dis, dis_vars = self._parse_losses(losses_dis)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_dis))
            loss_dis.backward()
            self._optim_step(optimizer, "dis")
            total_losses.update(losses_dis)
        self._optim_zero(optimizer, *optimizer.keys())
        total_loss, total_vars = self._parse_losses(total_losses)

        outputs = dict(
            loss=total_loss,
            log_vars=total_vars,
            num_samples=len(dataA["img_metas"]),
            visual={"img_metasA": dataA["img_metas"], "img_metasB": dataB["img_metas"]},
        )
        for out in [outputs_gen, outputs_dis, outputs_seg]:
            if out is not None:
                outputs["visual"].update(out)
        return outputs

    @staticmethod
    def _optim_zero(optims, *names, strict=False):
        if isinstance(optims, Iterable):
            for name in names:
                if strict or name in optims:
                    optims[name].zero_grad()

    @staticmethod
    def _optim_step(optims, *names, strict=False):
        if isinstance(optims, Iterable):
            for name in names:
                if strict or name in optims:
                    optims[name].step()

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self.forward_seg(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"])
        )

        return outputs

    @staticmethod
    def get_train_flow(train_flow):
        while True:
            for train, epoch in train_flow:
                for e in range(epoch):
                    yield train

    def _seg_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logits = self.seg.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg, output_pred=True
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses, seg_logits
