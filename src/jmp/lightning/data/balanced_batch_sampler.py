"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import heapq
from functools import cached_property
from logging import getLogger
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed
from lightning_fabric.utilities.distributed import _DatasetSamplerWrapper
from torch.utils.data import BatchSampler, Dataset, DistributedSampler
from typing_extensions import override

log = getLogger(__name__)


def _all_gather(tensor: torch.Tensor, device: torch.device | None = None):
    gathered = [
        torch.zeros_like(tensor, device=device)
        for _ in range(torch.distributed.get_world_size())
    ]
    _ = torch.distributed.all_gather(gathered, tensor)
    return gathered


# @numba.njit
def _balanced_partition(sizes: np.ndarray, num_parts: int):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)  # Sort in descending order
    heap = []
    for idx in sort_idx[:num_parts]:
        heap.append((sizes[idx], [idx]))
    heapq.heapify(heap)
    for idx in sort_idx[num_parts:]:
        smallest_part = heapq.heappop(heap)
        new_size = smallest_part[0] + sizes[idx]
        new_idx = smallest_part[1] + [idx]
        heapq.heappush(heap, (new_size, new_idx))
    idx_balanced = [part[1] for part in heap]
    return idx_balanced


@runtime_checkable
class DatasetWithSizes(Protocol):
    def data_sizes(self, indices: list[int]) -> np.ndarray: ...


class BalancedBatchSampler(BatchSampler):
    @staticmethod
    def _ensure_supported(dataset: Any):
        if not isinstance(dataset, Dataset):
            raise ValueError(
                "BalancedBatchSampler requires a dataset that implements `__getitem__`"
            )

        if not isinstance(dataset, DatasetWithSizes):
            raise ValueError(
                "BalancedBatchSampler requires a dataset that implements `data_sizes`"
            )

        log.critical(f"BalancedBatchSampler: Resolved dataset to {type(dataset)}")
        return dataset

    @staticmethod
    def _unwrap_dataset(dataset: Dataset) -> Dataset:
        if isinstance(dataset, _DatasetSamplerWrapper):
            if (data_source := getattr(dataset._sampler, "data_source", None)) is None:
                raise ValueError("Could not unwrap dataset from _DatasetSamplerWrapper")
            return data_source
        return dataset

    @property
    def distributed_sampler(self):
        if not isinstance(self.sampler, DistributedSampler):
            raise ValueError(
                f"Sampler must be a DistributedSampler, got {type(self.sampler)}"
            )
        return self.sampler

    @cached_property
    def dataset(self):
        return self._ensure_supported(
            self._unwrap_dataset(self.distributed_sampler.dataset)
        )

    def __init__(
        self,
        sampler: DistributedSampler,
        *,
        batch_size: int,
        device: torch.device,
        drop_last: bool = False,
    ):
        super().__init__(sampler, batch_size, drop_last=drop_last)

        self._device = device

        log.info(
            f"Created BalancedBatchSampler with {sampler=}, {batch_size=}, {drop_last=}"
        )

    @staticmethod
    def _dist_enabled():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @override
    def __iter__(self):
        if not self._dist_enabled():
            yield from super().__iter__()
            return

        for batch_idx in super().__iter__():
            sizes = self.dataset.data_sizes(batch_idx)
            idx_sizes = torch.stack(
                [
                    torch.tensor(batch_idx, device=self._device),
                    torch.tensor(sizes, device=self._device),
                ]
            )
            idx_sizes_all = _all_gather(idx_sizes, device=self._device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = _balanced_partition(
                sizes_all.numpy(), num_parts=self.distributed_sampler.num_replicas
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[local_idx_balanced[self.distributed_sampler.rank]].tolist()
