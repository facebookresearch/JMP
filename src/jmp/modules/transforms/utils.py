"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable

from torch_geometric.data.data import BaseData

Transform = Callable[[BaseData], BaseData]


def compose(transforms: list[Transform]):
    def composed(data: BaseData):
        for transform in transforms:
            data = transform(data)
        return data

    return composed
