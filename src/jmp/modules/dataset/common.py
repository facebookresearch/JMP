"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import assert_never

import numpy as np
import torch
from jmp.lightning import TypedConfig

from . import dataset_transform as DT
from .dataset_typing import TDataset


class DatasetSampleNConfig(TypedConfig):
    sample_n: int
    """Number of samples to take from the dataset"""

    seed: int
    """Seed for the random number generator used to sample the dataset"""


class DatasetAtomRefConfig(TypedConfig):
    refs: dict[str, dict[int, float] | list[float] | np.ndarray | torch.Tensor]
    """
    Reference values for each property.

    For each property, the references can be provided as:
    - A dictionary with the atom index as the key and the reference value as the value
    - A list with the reference values, `(max_atomic_number,)`
    - A numpy array with the reference values, `(max_atomic_number,)`
    - A torch tensor with the reference values, `(max_atomic_number,)`
    """


def _atomref_to_tensor(
    value: dict[int, float] | list[float] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    match value:
        case dict():
            max_atomic_number = max(value.keys())
            tensor = torch.zeros(max_atomic_number + 1)
            for key, val in value.items():
                tensor[key] = val
            return tensor
        case list():
            return torch.tensor(value)
        case np.ndarray():
            return torch.tensor(value)
        case torch.Tensor():
            return value
        case _:
            assert_never(value)


class CommonDatasetConfig(TypedConfig):
    sample_n: DatasetSampleNConfig | None = None
    """Sample n samples from the dataset"""

    atom_ref: DatasetAtomRefConfig | None = None
    """Configuration for referencing methods for atoms"""


def wrap_common_dataset(dataset: TDataset, config: CommonDatasetConfig) -> TDataset:
    if (sample_n := config.sample_n) is not None:
        dataset = DT.sample_n_transform(
            dataset,
            n=sample_n.sample_n,
            seed=sample_n.seed,
        )

    if (atom_ref := config.atom_ref) is not None:
        # Covnert the refs to a dict of tensors
        refs = {key: _atomref_to_tensor(value) for key, value in atom_ref.refs.items()}
        dataset = DT.atomref_transform(dataset, refs)

    return dataset
