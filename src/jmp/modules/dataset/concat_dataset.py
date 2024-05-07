"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import cache, partial
from logging import getLogger
from typing import Generic, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jmp.lightning import TypedConfig
from torch.utils.data import ConcatDataset
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ..metadata import post_create_dataset
from . import dataset_transform as DT
from .dataset_transform import expand_dataset
from .dataset_typing import DatasetProtocol, TDataset

log = getLogger(__name__)


def _update_graph_value(data: BaseData, key: str, onehot: torch.Tensor):
    assert (value := getattr(data, key, None)) is not None, f"{key} must be defined."
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float)

    value = cast(torch.Tensor, value)
    value = rearrange(value.view(-1), "1 -> 1 1") * onehot
    setattr(data, key, value)


def _update_node_value(data: BaseData, key: str, onehot: torch.Tensor):
    assert (value := getattr(data, key, None)) is not None, f"{key} must be defined."
    assert torch.is_tensor(value), f"{key} must be a tensor."
    value = cast(torch.Tensor, value)
    value = rearrange(value, "n ... -> n ... 1") * onehot
    setattr(data, key, value)


def _update_task_idx_transform(
    data: BaseData,
    *,
    task_idx: int,
    num_tasks: int,
    taskify_keys_graph: list[str] = ["y"],
    taskify_keys_node: list[str] = ["force"],
    use_onehot: bool = True,
):
    data.task_idx = torch.tensor(task_idx, dtype=torch.long)
    # set one-hot vector
    onehot: torch.Tensor = F.one_hot(
        data.task_idx, num_classes=num_tasks
    ).bool()  # (t,)
    taskify_onehot = onehot if use_onehot else torch.ones_like(onehot, dtype=torch.bool)

    # set task boolean mask
    data.task_mask = rearrange(onehot, "t -> 1 t")

    # update graph-level attrs to be a one-hot vector * attr
    for key in taskify_keys_graph:
        _update_graph_value(data, key, taskify_onehot)
        if f"{key}_norm_mean" in data:
            _update_graph_value(data, f"{key}_norm_mean", taskify_onehot)
        if f"{key}_norm_std" in data:
            _update_graph_value(data, f"{key}_norm_std", taskify_onehot)

    # update node-level attrs to be a one-hot vector * attr
    for key in taskify_keys_node:
        _update_node_value(data, key, taskify_onehot)
        if f"{key}_norm_mean" in data:
            _update_node_value(data, f"{key}_norm_mean", taskify_onehot)
        if f"{key}_norm_std" in data:
            _update_node_value(data, f"{key}_norm_std", taskify_onehot)

    return data


class _MTConcatDataset(ConcatDataset[BaseData], Generic[TDataset]):
    """
    Small wrapper around `ConcatDataset` which handles the concatenation of
    `atoms_metadata` properly. `atoms_metadata` stores the number of atoms in
    each molecule in the dataset and is used for balancing the batches
    during runtime without having to load the entire molecule into memory.
    """

    datasets: list[TDataset]

    @override
    def __init__(
        self,
        datasets: list[TDataset],
        *,
        taskify_keys_graph: list[str],
        taskify_keys_node: list[str],
        num_tasks: int,
        task_idxs: list[int],
        taskify_use_onehot: bool = True,
    ) -> None:
        datasets = list(datasets)

        datasets = [
            DT.transform(
                dataset,
                partial(
                    _update_task_idx_transform,
                    taskify_keys_graph=taskify_keys_graph,
                    taskify_keys_node=taskify_keys_node,
                    num_tasks=num_tasks,
                    task_idx=task_idx,
                    use_onehot=taskify_use_onehot,
                ),
            )
            for task_idx, dataset in zip(task_idxs, datasets)
        ]
        super().__init__(datasets)

        for dataset in self.datasets:
            if not isinstance(dataset, DatasetProtocol):
                raise TypeError(
                    f"Expected dataset to be an instance of DatasetProtocol, "
                    f"but got {dataset.__class__.__qualname__}."
                )

            if len(dataset.atoms_metadata) != len(dataset):
                raise ValueError(
                    f"Expected atoms_metadata to have the same length as the dataset, "
                    f"but got {len(dataset.atoms_metadata)=} and {len(dataset)=}."
                )

    @property
    @cache
    def atoms_metadata(self) -> np.ndarray:
        return np.concatenate([d.atoms_metadata for d in self.datasets])

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]


class MTDatasetConfig(TypedConfig):
    balanced: bool | None = None
    strict: bool = True

    taskify_keys_graph: list[str] = ["y", "y_scale", "force_scale"]
    """Converts the graph-level attributes to a one-hot vector * attr."""
    taskify_keys_node: list[str] = ["force"]
    """Converts the node-level attributes to a one-hot vector * attr."""
    taskify_use_onehot: bool = True
    """If True, the one-hot vector is used. If False, a vector of ones is used. (Should be True for most cases.)"""

    sample_type: Literal["uniform", "temperature"] | None = None
    """
    The type of sampling to use for the datasets.
    If `None`, the value of `balanced` will be used to determine the sampling type:
    - If `balanced` is `True`, `sample_type` will be set to "uniform".
    - If `balanced` is `False`, `sample_type` will be set to "temperature".
    """
    sample_temperature: float | None = 1.0
    """
    The temperature to use for temperature sampling.
    If `None`, the temperature will be set to 1.0.
    """


def _uniform_sampling(dataset_sizes: list[int]):
    return [1.0] * len(dataset_sizes)


def _temperature_sampling(dataset_sizes: list[int], temp: float):
    total_size = sum(dataset_sizes)  # 3.25 mil
    return [(size / total_size) ** (1.0 / temp) for size in dataset_sizes]


def merged_dataset(dataset_sizes_list: list[int], ratios_list: list[float]):
    dataset_sizes = np.array(dataset_sizes_list)
    ratios = np.array(ratios_list)

    # Calculate the target size of the final dataset
    target_size = sum(dataset_sizes) / sum(ratios)

    # Calculate the minimum expansion factor for each dataset
    expansion_factors = target_size * ratios / dataset_sizes

    # Make sure that the expansion factors are all at least 1.0
    expansion_factors = expansion_factors / np.min(expansion_factors)

    # Calculate the number of samples to take from each dataset
    samples_per_dataset = np.ceil(
        dataset_sizes * (expansion_factors / np.min(expansion_factors))
    ).astype(int)

    samples_per_dataset = cast(list[int], samples_per_dataset.tolist())
    return samples_per_dataset


class MTSampledDataset(_MTConcatDataset[TDataset], Generic[TDataset]):
    """
    Takes a list of datasets, and scales the loss weights of each dataset by
    the number of graphs and nodes in the dataset. This is useful for combining
    datasets with different numbers of graphs and nodes.
    """

    @override
    def __init__(
        self,
        datasets: list[TDataset],
        config: MTDatasetConfig,
        *,
        num_tasks: int | None = None,
        task_idxs: list[int] | None = None,
        ignore_balancing: bool = False,
    ) -> None:
        if num_tasks is None:
            num_tasks = len(datasets)

        if task_idxs is None:
            task_idxs = list(range(num_tasks))

        sample_type = config.sample_type
        sample_temperature = config.sample_temperature

        if sample_type is None:
            if config.balanced is None:
                raise ValueError(
                    "Either `sample_type` or `balanced` must be specified in `MTSampledDataset.__init__`."
                )

            if config.balanced:
                sample_type = "uniform"
            else:
                sample_type = "temperature"
                sample_temperature = 1.0

            log.critical(
                f"Using {sample_type=} and {sample_temperature=} because "
                f"`sample_type` is None and `balanced` is {config.balanced}."
            )

        assert (
            sample_type
            in {
                "uniform",
                "temperature",
            }
        ), f"{config.sample_type=} must be one of 'balanced', 'uniform', or 'temperature'."

        if ignore_balancing:
            log.critical(
                "Ignoring balancing because `ignore_balancing` is True in `MTSampledDataset.__init__`."
            )
            sample_type = "temperature"
            sample_temperature = 1.0

        match sample_type:
            case "uniform":
                # we use uniform sampling (i.e., we sample each dataset with equal probability)
                ratios = _uniform_sampling([len(d) for d in datasets])
            case "temperature":
                assert (
                    sample_temperature is not None
                ), "sample_temperature must be specified if sample_type is 'temperature'."
                ratios = _temperature_sampling(
                    [len(d) for d in datasets], sample_temperature
                )
            case _:
                raise ValueError(f"{sample_type=} is not a valid sampling type.")

        ratios = [r / sum(ratios) for r in ratios]
        log.info(f"Using {ratios=} for {sample_type=}.")

        expanded_dataset_sizes = merged_dataset([len(d) for d in datasets], ratios)
        datasets = [
            expand_dataset(dataset, n=n)
            for dataset, n in zip(datasets, expanded_dataset_sizes)
        ]

        super().__init__(
            datasets,
            taskify_keys_graph=config.taskify_keys_graph,
            taskify_keys_node=config.taskify_keys_node,
            num_tasks=num_tasks,
            task_idxs=task_idxs,
            taskify_use_onehot=config.taskify_use_onehot,
        )

        post_create_dataset(self, strict=config.strict)

    def representative_batch_for_testing(self, *, n: int = 1, start_index: int = 0):
        data_list = [
            cast(BaseData, dataset[index])
            for dataset in self.datasets
            for index in range(start_index, start_index + n)
        ]
        return data_list
