"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import fnmatch
from logging import getLogger
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from typing_extensions import NotRequired

log = getLogger(__name__)


class _LightningCheckpoint(TypedDict):
    optimizer_states: NotRequired[list[Any]]
    state_dict: dict[str, torch.Tensor]


def _get_parameter_list_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    ignore_scale_factor: bool,
    param_dict_prefixes: list[list[str]] | None = None,
):
    if param_dict_prefixes:
        buffer = list(state_dict.keys())
        ordered_parameters: list[str] = []
        for prefixes in param_dict_prefixes:
            to_remove = set[str]()
            for key in buffer:
                if not any(fnmatch.fnmatch(key, prefix) for prefix in prefixes):
                    continue

                ordered_parameters.append(key)
                to_remove.add(key)  # we don't want to remove while iterating

            log.critical(f"Processed {len(to_remove)} keys under prefixes: {prefixes}")
            for key in to_remove:
                buffer.remove(key)

        if buffer:
            ordered_parameters.extend(buffer)
    else:
        ordered_parameters = list(state_dict.keys())

    parameters_list: list[str] = []
    for k in ordered_parameters:
        if k.endswith("rbf.offset") or k.endswith("rbf.temps"):
            continue
        if ".seq_energy_pre." in k:
            continue
        if k.startswith("task_steps"):
            continue
        if ignore_scale_factor and k.endswith("scale_factor"):
            continue
        parameters_list.append(k)
    return parameters_list


def retreive_state_dict_for_finetuning(
    ckpt_path: str | Path,
    load_emas: bool = True,
    ignore_scale_factor: bool = True,
    param_dict_prefixes: list[list[str]] | None = None,
):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(ckpt, dict), type(ckpt)

    ckpt = cast(_LightningCheckpoint, ckpt)
    state_dict = retreive_ft_state_dict_from_loaded_ckpt(
        ckpt, load_emas, ignore_scale_factor, param_dict_prefixes
    )
    log.critical(f"Loaded state dict from {ckpt_path}")
    return state_dict


def retreive_ft_state_dict_from_loaded_ckpt(
    ckpt: _LightningCheckpoint,
    load_emas: bool = True,
    ignore_scale_factor: bool = True,
    param_dict_prefixes: list[list[str]] | None = None,
):
    state_dict = ckpt["state_dict"].copy()
    if load_emas:
        optimizer_states = ckpt.get("optimizer_states")
        assert optimizer_states is not None, "optimizer_states must be present"

        assert len(optimizer_states) == 1, f"{len(optimizer_states)=} != 1"
        optimizer_state = optimizer_states[0]
        assert isinstance(optimizer_state, dict), type(optimizer_state)
        assert (ema := optimizer_state.get("ema")) is not None, "ema must be present"
        assert isinstance(ema, tuple), type(ema)

        ema = cast(list[torch.Tensor], list(ema))
        parameters_list = _get_parameter_list_from_state_dict(
            state_dict,
            ignore_scale_factor,
            param_dict_prefixes,
        )

        assert len(ema) == len(
            parameters_list
        ), f"{len(ema)=} != {len(parameters_list)=}"

        for i, (param, ema) in enumerate(zip(parameters_list, ema)):
            existing_param = state_dict[param]
            assert (
                existing_param.shape == ema.shape
            ), f"{existing_param.shape=} != {ema.shape=} for {param=} at index {i}"
            assert (
                existing_param.dtype == ema.dtype
            ), f"{existing_param.dtype=} != {ema.dtype=} for {param=} at index {i}"
            state_dict[param] = ema
            log.info(f"Loaded EMA for {param}")

        log.critical(f"Loaded {len(parameters_list)} EMA parameters")

    return state_dict


def filter_state_dict(state_dict: dict[str, torch.Tensor], prefix: str):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
