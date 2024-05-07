"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LambdaCallback
from torch.optim import Optimizer
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin

log = getLogger(__name__)


def grad_norm(
    module: nn.Module,
    norm_type: float | int | str,
    group_separator: str = "/",
    grad: bool = True,
) -> dict[str, float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(
            f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}"
        )

    if grad:
        norms = {
            f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type)
            for name, p in module.named_parameters()
            if p.grad is not None
        }
        if norms:
            total_norm = torch.tensor(list(norms.values())).norm(norm_type)
            norms[f"grad_{norm_type}_norm_total"] = total_norm
    else:
        norms = {
            f"param_{norm_type}_norm{group_separator}{name}": p.data.norm(norm_type)
            for name, p in module.named_parameters()
            if p.grad is not None
        }
        if norms:
            total_norm = torch.tensor(list(norms.values())).norm(norm_type)
            norms[f"param_{norm_type}_norm_total"] = total_norm

    return norms


def _to_norm_type(log_grad_norm_per_param: float | str | Literal[True]):
    norm_type = 2.0
    if log_grad_norm_per_param is not True:
        norm_type = log_grad_norm_per_param
    return norm_type


def _skipped_steps_on_before_optimizer_step(
    trainer: Trainer,
    pl_module: LightningModule,
    optimizer: Optimizer,
) -> None:
    if not isinstance(pl_module, OptimizerModuleMixin):
        raise TypeError(f"Expected OptimizerModuleMixin, got {type(pl_module)}")

    if (
        config := cast(
            BaseConfig, pl_module.hparams
        ).trainer.optimizer.gradient_skipping
    ) is None or not config.enabled:
        return

    # Skip the step if the global step is less than the start_after_n_steps
    # This is because we want to let AMP adjust the loss scale before we start
    if (
        config.start_after_n_steps is not None
        and pl_module.global_step < config.start_after_n_steps
    ):
        return

    norm = pl_module.compute_parameter_norm(optimizer, config.norm_type)
    # If the norm is NaN/Inf, we don't want to skip the step
    # beacuse AMP checks for NaN/Inf grads to adjust the loss scale.
    if torch.isfinite(norm).all() and (norm > config.threshold).any():
        optimizer.zero_grad()
        log.warning(
            f"Skipping step at global step {pl_module.global_step} with norm {norm:.2f} > {config.threshold:.2f}"
        )
        pl_module.grad_skipped_steps(1)
    else:
        pl_module.grad_skipped_steps(0)

    pl_module.log(
        "train/grad_skipped_steps",
        pl_module.grad_skipped_steps,
        on_step=True,
        on_epoch=False,
    )
    pl_module._perform_norm_logging(optimizer, prefix="train/post_skip_")


class OptimizerModuleMixin(mixin_base_type(CallbackModuleMixin)):
    grad_skipped_steps: torchmetrics.SumMetric

    @override
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        def _grad_skip_callback():
            nonlocal self

            if (
                config := cast(
                    BaseConfig, self.hparams
                ).trainer.optimizer.gradient_skipping
            ) is not None and config.enabled:
                self.grad_skipped_steps = torchmetrics.SumMetric()

            return LambdaCallback(
                on_before_optimizer_step=_skipped_steps_on_before_optimizer_step
            )

        self.register_callback(_grad_skip_callback)

    def compute_parameter_norm(
        self,
        optimizer: Optimizer | None = None,
        p: float | str = 2.0,
        grad: bool = True,
    ) -> torch.Tensor:
        if optimizer is not None:
            tensors = [
                cast(torch.Tensor, p.grad if grad else p)
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
        else:
            tensors = [
                p.grad if grad else p for p in self.parameters() if p.grad is not None
            ]

        if not tensors:
            return torch.tensor(0.0, device=self.device)

        return torch.norm(torch.stack([torch.norm(g, p=p) for g in tensors]), p=p)

    def _perform_norm_logging(self, optimizer: Optimizer, prefix: str):
        config = cast(BaseConfig, self.hparams)

        # Gradient norm logging
        if log_grad_norm := config.trainer.optimizer.log_grad_norm:
            norm = self.compute_parameter_norm(
                optimizer, _to_norm_type(log_grad_norm), grad=True
            )
            self.log(f"{prefix}grad_norm", norm, on_step=True, on_epoch=False)
        if log_grad_norm_per_param := config.trainer.optimizer.log_grad_norm_per_param:
            norm_type = _to_norm_type(log_grad_norm_per_param)
            self.log_dict(
                {
                    f"{prefix}{k}": v
                    for k, v in grad_norm(self, norm_type, grad=True).items()
                }
            )

        # Parameter norm logging
        if log_param_norm := config.trainer.optimizer.log_param_norm:
            norm = self.compute_parameter_norm(
                optimizer, _to_norm_type(log_param_norm), grad=False
            )
            self.log(f"{prefix}param_norm", norm, on_step=True, on_epoch=False)
        if (
            log_param_norm_per_param
            := config.trainer.optimizer.log_param_norm_per_param
        ):
            norm_type = _to_norm_type(log_param_norm_per_param)
            self.log_dict(
                {
                    f"{prefix}{k}": v
                    for k, v in grad_norm(self, norm_type, grad=False).items()
                }
            )

    @override
    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ):
        self._perform_norm_logging(optimizer, prefix="train/")

        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )
