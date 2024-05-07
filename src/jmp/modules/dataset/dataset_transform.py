"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from collections import abc
from collections.abc import Callable
from functools import cache, partial
from logging import getLogger
from typing import Any, cast

import numpy as np
import torch
import wrapt
from typing_extensions import override

from .. import transforms as T
from .dataset_typing import TDataset

log = getLogger(__name__)


def transform(
    dataset: TDataset,
    transform: Callable[[Any], Any],
    copy_data: bool = True,
) -> TDataset:
    """
    Applies a transformation/mapping function to all elements of the dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        transform (Callable): The transformation function.
        copy_data (bool, optional): Whether to copy the data before transforming. Defaults to True.
    """

    class _TransformedDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx):
            nonlocal copy_data, transform

            assert transform is not None, "Transform must be defined."
            data = self.__wrapped__.__getitem__(idx)
            if copy_data:
                data = copy.deepcopy(data)
            data = transform(data)
            return data

    return cast(TDataset, _TransformedDataset(dataset))


def atomref_transform(
    dataset: TDataset,
    refs: dict[str, torch.Tensor],
    keep_raw: bool = False,
) -> TDataset:
    """
    Subtracts the atomrefs from the target properties of the dataset. For a data sample x and atomref property p,
    the transformed property is `x[p] = x[p] - atomref[x.atomic_numbers].sum()`.

    This is primarily used to normalize energies using a "linear referencing" scheme.

    Args:
        dataset (Dataset): The dataset to transform.
        refs (dict[str, torch.Tensor]): The atomrefs to subtract from the target properties.
        keep_raw (bool, optional): Whether to keep the original properties, renamed as `{target}_raw`. Defaults to False.
    """
    # Convert the refs to tensors
    refs_dict: dict[str, torch.Tensor] = {}
    for k, v in refs.items():
        if isinstance(v, list):
            v = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float()
        elif not torch.is_tensor(v):
            raise TypeError(f"Invalid type for {k} in atomrefs: {type(v)}")
        refs_dict[k] = v

    return transform(
        dataset,
        partial(T.atomref_transform, refs=refs_dict, keep_raw=keep_raw),
        copy_data=False,
    )


def expand_dataset(dataset: TDataset, n: int) -> TDataset:
    """
    Expands the dataset to have `n` elements by repeating the elements of the dataset as many times as necessary.

    Args:
        dataset (Dataset): The dataset to expand.
        n (int): The desired length of the dataset.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"expand_dataset ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    og_size = len(dataset)
    if og_size > n:
        raise ValueError(
            f"expand_dataset ({n}) must be greater than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__}"
        )

    class _ExpandedDataset(wrapt.ObjectProxy):
        @override
        def __len__(self):
            nonlocal n
            return n

        @override
        def __getitem__(self, index: int):
            nonlocal n, og_size
            if index < 0 or index >= n:
                raise IndexError(
                    f"Index {index} is out of bounds for dataset of size {n}."
                )
            return self.__wrapped__.__getitem__(index % og_size)

        @cache
        def _atoms_metadata_cached(self):
            """
            We want to retrieve the atoms metadata for the expanded dataset.
            This includes repeating the atoms metadata for the elemens that are repeated.
            """

            # the out metadata shape should be (n,)
            nonlocal n, og_size

            metadata = self.__wrapped__.atoms_metadata
            metadata = np.resize(metadata, (n,))
            log.debug(
                f"Expanded the atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    dataset = cast(TDataset, _ExpandedDataset(dataset))
    log.info(f"Expanded dataset {dataset.__class__.__name__} from {og_size} to {n}.")
    return dataset


def first_n_transform(dataset: TDataset, *, n: int) -> TDataset:
    """
    Returns a new dataset that contains the first `n` elements of the original dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to keep.
    """
    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"first_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) < n:
        raise ValueError(
            f"first_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    class _FirstNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(idx)

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the first n elements."""
            nonlocal n

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[:n]

            log.info(
                f"Retrieved the first {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _FirstNDataset(dataset))


def sample_n_transform(dataset: TDataset, *, n: int, seed: int) -> TDataset:
    """
    Similar to first_n_transform, but samples n elements randomly from the dataset.

    Args:
        dataset (Dataset): The dataset to transform.
        n (int): The number of elements to sample.
        seed (int): The random seed to use for sampling.
    """

    if not isinstance(dataset, abc.Sized):
        raise TypeError(
            f"sample_n ({n}) must be used with a dataset that is an instance of abc.Sized "
            f"for {dataset.__class__.__qualname__} "
        )

    if len(dataset) < n:
        raise ValueError(
            f"sample_n ({n}) must be less than or equal to the length of the dataset "
            f"({len(dataset)}) for {dataset.__class__.__qualname__} "
        )

    sampled_indices = np.random.default_rng(seed).choice(len(dataset), n, replace=False)

    class _SampleNDataset(wrapt.ObjectProxy):
        @override
        def __getitem__(self, idx: int):
            nonlocal n, sampled_indices

            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Index {idx} is out of bounds for dataset of size {n}."
                )

            return self.__wrapped__.__getitem__(sampled_indices[idx])

        @override
        def __len__(self):
            nonlocal n
            return n

        @cache
        def _atoms_metadata_cached(self):
            """We only want to retrieve the atoms metadata for the sampled n elements."""
            nonlocal n, sampled_indices

            metadata = self.__wrapped__.atoms_metadata
            og_size = len(metadata)
            metadata = metadata[sampled_indices]

            log.info(
                f"Retrieved the sampled {n} atoms metadata for {self.__class__.__name__} ({og_size} => {len(metadata)})."
            )
            return metadata

        @property
        def atoms_metadata(self):
            return self._atoms_metadata_cached()

    return cast(TDataset, _SampleNDataset(dataset))
