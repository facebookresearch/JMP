"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch_geometric.data.data import BaseData


def atomref_transform(
    data: BaseData,
    refs: dict[str, torch.Tensor],
    keep_raw: bool = False,
):
    z: torch.Tensor = data.atomic_numbers
    for target, coeffs in refs.items():
        value = getattr(data, target)
        if keep_raw:
            setattr(data, f"{target}_raw", value.clone())
        value = value - coeffs[z].sum()
        setattr(data, target, value)

    return data
