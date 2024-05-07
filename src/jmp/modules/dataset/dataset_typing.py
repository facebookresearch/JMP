"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar


@runtime_checkable
class DatasetProtocol(Protocol):
    @property
    def atoms_metadata(self) -> np.ndarray: ...

    def __getitem__(self, index: int, /) -> BaseData: ...

    def __len__(self) -> int: ...


TDataset = TypeVar("TDataset", bound=DatasetProtocol, infer_variance=True)
