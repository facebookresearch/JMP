"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from typing_extensions import override

log = getLogger(__name__)


def print_bad_gradients(
    module: LightningModule,
    nonfinite_grads: bool = True,
    none_grads: bool = False,
):
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            if none_grads:
                log.critical(f"Parameter {name} ({param.shape}) has None gradients")
            continue

        if not nonfinite_grads or torch.isfinite(param.grad.float()).all():
            continue

        has_nan = torch.isnan(param.grad.float()).any()
        has_inf = torch.isinf(param.grad.float()).any()
        kinds = [
            "NaN" if has_nan else None,
            "Inf" if has_inf else None,
        ]
        kinds = " and ".join(prop for prop in kinds if prop is not None)
        log.critical(f"{name} ({param.shape}) has {kinds} gradients")


class PrintBadGradientsCallback(Callback):
    def __init__(
        self,
        *,
        nonfinite_grads: bool = True,
        none_grads: bool = False,
    ):
        super().__init__()

        self._nonfinite_grads = nonfinite_grads
        self._none_grads = none_grads

    @override
    def on_after_backward(self, _trainer: Trainer, module: LightningModule):
        print_bad_gradients(
            module,
            nonfinite_grads=self._nonfinite_grads,
            none_grads=self._none_grads,
        )
