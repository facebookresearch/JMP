"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

import torch.nn as nn
import torch.optim as optim
from jmp.lightning import Field, TypedConfig

log = getLogger(__name__)


class OutputConfig(TypedConfig):
    num_mlps: int = 5
    """Number of MLPs in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""


class AdamWConfig(TypedConfig):
    name: Literal["adamw"] = "adamw"

    lr: float
    """Learning rate for the optimizer."""

    weight_decay: float = 1.0e-2
    """Weight decay (L2 penalty) for the optimizer."""

    betas: tuple[float, float] = (0.9, 0.999)
    """
    Betas for the optimizer:
    (beta1, beta2) are the coefficients used for computing running averages of
    gradient and its square.
    """

    eps: float = 1e-8
    """Term added to the denominator to improve numerical stability."""

    amsgrad: bool = False
    """Whether to use the AMSGrad variant of this algorithm."""


@dataclass(frozen=True)
class _OptimizerParamGroupConfig:
    cls: type[optim.Optimizer]
    param_group_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {})


OptimizerConfig: TypeAlias = Annotated[AdamWConfig, Field(discriminator="name")]


class EmbeddingConfig(TypedConfig):
    num_elements: int
    embedding_size: int


def _create_dict_from_config(
    config: OptimizerConfig,
    params: Iterable[nn.Parameter],
    name: str | None = None,
):
    # This is a hack to get type hints for the kwargs
    # of the module while actually returning a dict.
    from torch.optim import AdamW

    AdamWKwargs = AdamW

    if not TYPE_CHECKING:
        AdamWKwargs = dict

    if config.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {config.lr}")

    kwargs = cast(
        dict,
        AdamWKwargs(
            params=params,
            lr=config.lr,
            amsgrad=config.amsgrad,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        ),
    )
    if name is not None:
        kwargs["name"] = name
    return _OptimizerParamGroupConfig(AdamW, param_group_kwargs=kwargs)


def optimizer_from_config(
    param_groups: list[tuple[OptimizerConfig, Iterable[nn.Parameter]]]
    | list[tuple[OptimizerConfig, Iterable[nn.Parameter], str | None]],
    *,
    base: "OptimizerConfig | None" = None,
):
    configs = [
        _create_dict_from_config(
            param_group[0],
            param_group[1],
            name=param_group[2] if len(param_group) == 3 else None,
        )
        for param_group in param_groups
    ]
    optimizer_cls_list = [c.cls for c in configs]
    assert len(set(optimizer_cls_list)) == 1, "All optimizers must be of the same type"
    optimizer_cls = optimizer_cls_list[0]

    optimizer_kwargs_list = [c.optimizer_kwargs for c in configs]
    assert (
        len(set(map(str, optimizer_kwargs_list))) == 1
    ), "All optimizers must have the same kwargs"
    optimizer_kwargs = optimizer_kwargs_list[0]

    base_kwargs = {}
    if base is not None:
        base_config = _create_dict_from_config(base, [])
        assert (
            base_config.cls == optimizer_cls
        ), "Base optimizer must be of the same type"
        _ = base_config.param_group_kwargs.pop("params", None)
        base_kwargs.update(base_config.param_group_kwargs)

    param_groups_configs = [c.param_group_kwargs for c in configs]
    optimizer = optimizer_cls(
        params=param_groups_configs,
        **optimizer_kwargs,
        **base_kwargs,
    )
    # detailed log about the optimizer configuration
    param_groups_logs: list[str] = []
    for i, c in enumerate(param_groups_configs):
        c = copy.deepcopy(c)
        params = c.pop("params", None)
        n_params = len(params) if params is not None else 0
        total_param_size = sum(p.numel() for p in params) if params is not None else 0
        param_groups_logs.append(
            f"Param group {i}:\n"
            f"    Params: {n_params}\n"
            f"    Total param size: {total_param_size}\n"
            f"    Other kwargs: {c}"
        )
    param_groups_log = "\n".join(param_groups_logs)
    log.critical(
        f"Optimizer: {optimizer_cls.__name__}\n"
        f"Optimizer kwargs: {optimizer_kwargs}\n"
        f"Base kwargs: {base_kwargs}\n"
        f"Param groups: {param_groups_log}"
    )
    return optimizer
