# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import datetime
from collections import OrderedDict

import torch

import mmcv
from mmcv.runner import HOOKS
from mmcv.runner import TextLoggerHook


@HOOKS.register_module()
class CustomizedTextLoggerHook(TextLoggerHook):
    """Customized Text Logger hook.

    This logger prints out both lr and layer_0_lr.
        
    """

    def __init__(
        self,
        by_epoch=True,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        interval_exp_name=1000,
        out_dir=None,
        out_suffix=...,
        keep_local=True,
        file_client_args=None,
    ):
        super().__init__(
            by_epoch,
            interval,
            ignore_last,
            reset_flag,
            interval_exp_name,
            out_dir,
            out_suffix,
            keep_local,
            file_client_args,
        )
        self.first_flag = True

    def after_train_iter(self, runner):
        if self.first_flag:
            runner.logger.info("The first iteration runs smoothly.")
            self.first_flag = False
        super().after_train_iter(runner)
