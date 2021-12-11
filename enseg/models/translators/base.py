import torch
from mmcv.runner import BaseModule
from ..builder import TRANSLATOR, BACKBONES, DECODE_GEN


@TRANSLATOR.register_module()
class BaseTranslator(BaseModule):
    def __init__(self, encode=None, decode=None, encode_decode=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if encode_decode is not None:
            self.model = TRANSLATOR.build(encode_decode)
        else:
            self.encode = BACKBONES.build(encode)
            self.decode = DECODE_GEN.build(decode)

    def forward(self, *args, **kwargs):
        if hasattr(self, "model"):
            return self.model(*args,**kwargs)
        else:
            return self.decode(self.encode(*args,**kwargs))
