"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import override

from .linear_warmup_cosine_annealing import PerParamGroupLinearWarmupCosineAnnealingLR

log = getLogger(__name__)


class PerParamGroupLinearWarmupCosineAnnealingRLPLR(
    PerParamGroupLinearWarmupCosineAnnealingLR
):
    def __init__(
        self,
        optimizer: Optimizer,
        param_group_settings: list[dict[str, Any]] | dict[str, Any],
        rlp_settings: dict,
        max_epochs: int,
        last_epoch: int = -1,
    ) -> None:
        self.rlp_start_epoch = max_epochs
        self.rlp = ReduceLROnPlateau(optimizer, **rlp_settings)
        self.global_step: int | None = None

        super().__init__(
            optimizer,
            param_group_settings,
            last_epoch,
        )

        for settings in self.param_group_settings:
            should_restart = settings.get("should_restart", False)
            assert (
                not should_restart
            ), "If you want to use RLP, set should_restart=False."

            max_epochs_setting = settings.get("max_epochs", None)
            if max_epochs_setting is not None:
                assert (
                    max_epochs_setting == max_epochs
                ), f"max_epochs must be {max_epochs}"
            else:
                settings["max_epochs"] = max_epochs

    def on_new_step(self, global_step: int):
        self.global_step = global_step
        log.debug(f"global_step: {self.global_step}")

    def is_in_rlp_stage(self, global_step: int | None = None):
        if global_step is None:
            global_step = self.global_step
        if global_step is None:
            global_step = self.last_epoch
        return global_step >= self.rlp_start_epoch

    def rlp_step(self, metric: float | torch.Tensor):
        return self.rlp.step(metric)

    @override
    def step(self, metrics=None, epoch=None):
        assert metrics is None, f"metrics must be None but got {metrics}"
        if self.is_in_rlp_stage():
            return

        return super().step(epoch=epoch)
