"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import warnings
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        should_restart: bool = True,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.should_restart = should_restart

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if not self.should_restart and self.last_epoch >= self.max_epochs:
            return [self.eta_min] * len(self.base_lrs)

        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class PerParamGroupLinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        param_group_settings: list[dict[str, Any]] | dict[str, Any],
        last_epoch: int = -1,
    ) -> None:
        if isinstance(param_group_settings, dict):
            param_group_settings = [param_group_settings] * len(optimizer.param_groups)

        if len(param_group_settings) != len(optimizer.param_groups):
            raise ValueError(
                "Number of elements in param_group_settings must match the number of parameter groups in the optimizer."
            )

        self.param_group_settings = param_group_settings
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        new_lrs = []
        for group, settings in zip(
            self.optimizer.param_groups, self.param_group_settings
        ):
            warmup_epochs = settings["warmup_epochs"]
            max_epochs = settings["max_epochs"]
            warmup_start_lr = settings.get("warmup_start_lr", 0.0)
            eta_min = settings.get("eta_min", 0.0)
            should_restart = settings.get("should_restart", True)

            if self.last_epoch == 0:
                new_lrs.append(warmup_start_lr)
            elif self.last_epoch < warmup_epochs:
                new_lr = group["lr"] + (group["initial_lr"] - warmup_start_lr) / (
                    warmup_epochs - 1
                )
                new_lrs.append(new_lr)
            elif self.last_epoch == warmup_epochs:
                new_lrs.append(group["initial_lr"])
            elif not should_restart and self.last_epoch >= max_epochs:
                new_lrs.append(eta_min)
            else:
                new_lr = eta_min + 0.5 * (group["initial_lr"] - eta_min) * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - warmup_epochs)
                        / (max_epochs - warmup_epochs)
                    )
                )
                new_lrs.append(new_lr)

        return new_lrs


def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn
