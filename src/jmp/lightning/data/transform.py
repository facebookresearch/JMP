"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from typing import Any, Callable, cast

import wrapt
from typing_extensions import TypeVar, override

TDataset = TypeVar("TDataset", infer_variance=True)


def transform(
    dataset: TDataset,
    transform: Callable[[Any], Any],
    *,
    deepcopy: bool = False,
) -> TDataset:
    class _TransformedDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx):
            nonlocal deepcopy, transform

            data = self.__wrapped__.__getitem__(idx)
            if deepcopy:
                data = copy.deepcopy(data)
            data = transform(data)
            return data

    return cast(TDataset, _TransformedDataset(dataset))


def transform_with_index(
    dataset: TDataset,
    transform: Callable[[Any, int], Any],
    *,
    deepcopy: bool = False,
) -> TDataset:
    class _TransformedWithIndexDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal deepcopy, transform

            data = self.__wrapped__.__getitem__(idx)
            if deepcopy:
                data = copy.deepcopy(data)
            data = transform(data, idx)
            return data

    return cast(TDataset, _TransformedWithIndexDataset(dataset))


__all__ = ["transform"]
