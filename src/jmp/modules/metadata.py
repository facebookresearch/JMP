"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


@runtime_checkable
class _DatasetWrapper(Protocol):
    @property
    def dataset(self) -> Dataset: ...


@runtime_checkable
class _HasMetadataPath(Protocol):
    @property
    def metadata_path(self) -> Path: ...


@runtime_checkable
class _HasMetadataProperty(Protocol):
    @property
    def atoms_metadata(self) -> torch.Tensor | np.ndarray: ...


def _dataset_repr(dataset):
    if isinstance(dataset, _DatasetWrapper):
        name = dataset.__class__.__qualname__
        inner_repr = _dataset_repr(dataset.dataset)
        return f"{name}({inner_repr})"

    return dataset.__class__.__qualname__


def _get_metadata_property(dataset):
    if isinstance(dataset, _HasMetadataProperty):
        return (
            dataset.atoms_metadata.numpy()
            if isinstance(dataset.atoms_metadata, torch.Tensor)
            else dataset.atoms_metadata
        )

    return None


def _get_metadata_path(dataset):
    if isinstance(dataset, _HasMetadataPath):
        if not dataset.metadata_path.exists():
            return None
        return np.load(dataset.metadata_path)["natoms"]

    return None


def get_metadata(
    dataset,
    *,
    strict: bool,
):
    fns = [_get_metadata_property, _get_metadata_path]
    metadata = next((fn(dataset) for fn in fns if fn(dataset) is not None), None)
    if metadata is None:
        if strict:
            raise RuntimeError(f"Failed to load metadata for {_dataset_repr(dataset)}")
        else:
            logging.warning(f"Failed to load metadata for {_dataset_repr(dataset)}")
    return metadata


def post_create_dataset(dataset: Dataset, *, strict: bool):
    if isinstance(dataset, (ConcatDataset)):
        metadata = [get_metadata(d, strict=strict) for d in dataset.datasets]
        all_set = True
        for dataset_idx, (m, dataset_) in enumerate(zip(metadata, dataset.datasets)):
            if m is not None:
                logging.debug(
                    f"Loaded metadata of size ({len(m)}) for {_dataset_repr(dataset_)}"
                )
                continue

            all_set = False
            error_msg = (
                f"Failed to load metadata for {dataset_idx=}: {_dataset_repr(dataset_)}"
            )
            if strict:
                raise RuntimeError(error_msg)
            else:
                logging.warning(error_msg)
        if all_set and False:
            setattr(dataset, "atoms_metadata", np.concatenate(metadata))
    else:
        metadata = get_metadata(dataset, strict=strict)
        if strict and metadata is None:
            raise RuntimeError(f"Failed to load metadata for {_dataset_repr(dataset)}")
