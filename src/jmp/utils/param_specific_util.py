"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from typing import Any, cast

from jmp.tasks.finetune.base import (
    FinetuneConfigBase,
    ParamSpecificOptimizerConfig,
    WarmupCosRLPConfig,
)
from typing_extensions import TypeVar


def PARAMETER_PATTERNS(num_blocks: int):
    return {
        "embedding": ["embedding.*"],
        "additional_embedding": ["additional_embedding.*"],
        "bases": ["backbone.bases.*"],
        # "all_int_blocks": ["backbone.int_blocks.*"],
        **{
            f"int_blocks_{i}": [f"backbone.int_blocks.{i}.*"] for i in range(num_blocks)
        },
        # "all_out_blocks": ["backbone.out_blocks.*"],
        **{
            f"out_blocks_{i}": [f"backbone.out_blocks.{i}.*"]
            for i in range(num_blocks + 1)
        },
        **{
            f"blocks_{i}": [
                f"backbone.int_blocks.{i}.*",
                f"backbone.out_blocks.{i+1}.*",
                *(["backbone.out_blocks.0.*"] if i == 0 else []),
            ]
            for i in range(num_blocks)
        },
        "out_mlp_E": ["backbone.out_mlp.E.*"],
    }


TConfig = TypeVar("TConfig", infer_variance=True)


def make_parameter_specific_optimizer_config(
    config: FinetuneConfigBase,
    num_blocks: int,
    max_lr_scales: dict[str, float],
):
    base_lr = config.optimizer.lr

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] = []
    max_lr_scales = cast(dict[str, Any], max_lr_scales)
    for name, lr_scale in max_lr_scales.items():
        assert isinstance(lr_scale, float), f"max_lr_scales[{name}] must be float"

        optimizer = copy.deepcopy(config.optimizer)
        optimizer.lr = base_lr * lr_scale

        lrs = None
        match config.lr_scheduler:
            case WarmupCosRLPConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.min_lr_factor = lrs.min_lr_factor / lr_scale
                lrs.min_lr_factor = max(0.01, min(0.99, lrs.min_lr_factor))
            case _:
                raise ValueError(
                    "You must set config.lr_scheduler to WarmupCosRLPConfig to use parameter specific optimizers."
                )

        assert (
            (parameter_patterns := PARAMETER_PATTERNS(num_blocks).get(name)) is not None
        ), f"PARAMETER_PATTERNS[{name}] is None. You must set PARAMETER_PATTERNS[{name}]"
        parameter_specific_optimizers.append(
            ParamSpecificOptimizerConfig(
                paremeter_patterns=parameter_patterns,
                optimizer=optimizer,
                lr_scheduler=lrs,
            )
        )

    return parameter_specific_optimizers
