"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import fnmatch
from collections import defaultdict
from collections.abc import Mapping
from logging import getLogger

import torch
import torch.nn as nn
from lightning_fabric.utilities import rank_zero_warn

log = getLogger(__name__)


def _report_incompat_keys(
    model: nn.Module,
    missing_keys: list[str],
    unexpected_keys: list[str],
    strict: bool = True,
):
    error_msgs = []
    if missing_keys:
        error_msgs.insert(
            0,
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            ),
        )
    if unexpected_keys:
        error_msgs.insert(
            0,
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            ),
        )

    if len(error_msgs) > 0:
        error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
        )
        if strict:
            raise RuntimeError(error_msg)
        else:
            rank_zero_warn(error_msg)


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    ignored_key_patterns: list[str] | None = None,
    ignored_missing_keys: list[str] | None = None,
    ignored_unexpected_keys: list[str] | None = None,
    strict: bool = True,
):
    if ignored_key_patterns:
        updated_state_dict: dict[str, torch.Tensor] = {}
        matching_patterns = defaultdict[str, list[str]](lambda: [])
        for k, v in state_dict.items():
            if (
                matched_pattern := next(
                    (
                        pattern
                        for pattern in ignored_key_patterns
                        if fnmatch.fnmatch(k, pattern)
                    ),
                    None,
                )
            ) is not None:
                matching_patterns[matched_pattern].append(k)
                continue

            updated_state_dict[k] = v
        state_dict = updated_state_dict

        for pattern, matched_keys in matching_patterns.items():
            log.critical(
                f"{pattern=} matched keys {matched_keys}, "
                "which were ignored during loading."
            )

    missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

    if ignored_key_patterns:
        missing_keys = [
            k
            for k in missing_keys
            if not any(fnmatch.fnmatch(k, pattern) for pattern in ignored_key_patterns)
        ]
    if ignored_missing_keys:
        missing_keys = [k for k in missing_keys if k not in ignored_missing_keys]

    if ignored_unexpected_keys:
        unexpected_keys = [
            k for k in unexpected_keys if k not in ignored_unexpected_keys
        ]

    _report_incompat_keys(module, missing_keys, unexpected_keys, strict=strict)
