"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from dataclasses import replace
from typing import final

import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...datasets.finetune_pdbbind import PDBBindTask
from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import FinetuneConfigBase, FinetuneModelBase, FinetunePDBBindDatasetConfig
from .metrics import MetricPair


class PDBBindConfig(FinetuneConfigBase):
    pbdbind_task: PDBBindTask

    graph_scalar_targets: list[str] = ["y"]
    node_vector_targets: list[str] = []

    cutoff: float = 12.0
    max_neighbors: int = 30
    pbc: bool = False

    @override
    def __post_init__(self):
        super().__post_init__()

        all_datasets: list[FinetunePDBBindDatasetConfig] = []
        if self.train_dataset is not None:
            assert isinstance(
                self.train_dataset, FinetunePDBBindDatasetConfig
            ), "dataset config must be of type FinetunePDBBindDatasetConfig"
            all_datasets.append(self.train_dataset)

        if self.val_dataset is not None:
            assert isinstance(
                self.val_dataset, FinetunePDBBindDatasetConfig
            ), "dataset config must be of type FinetunePDBBindDatasetConfig"
            all_datasets.append(self.val_dataset)

        if self.test_dataset is not None:
            assert isinstance(
                self.test_dataset, FinetunePDBBindDatasetConfig
            ), "dataset config must be of type FinetunePDBBindDatasetConfig"
            all_datasets.append(self.test_dataset)

        # Make sure all datasets have the same task
        for dataset in all_datasets:
            assert (
                dataset.task == self.pbdbind_task
            ), "All datasets must have the same task"


@final
class PDBBindModel(FinetuneModelBase[PDBBindConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return PDBBindConfig

    @override
    def metric_prefix(self) -> str:
        return f"pdbbind/{self.config.pbdbind_task}"

    @override
    def metrics_provider(
        self,
        prop: str,
        batch: Batch,
        preds: dict[str, torch.Tensor],
    ) -> MetricPair | None:
        """
        For PDBbind, the moleculenet dataset already normalizes the properties when it gives it to us.
        Therefore, we need to change the logic for unnormalizing the properties for metrics.
        """
        if (pair := super().metrics_provider(prop, batch, preds)) is None:
            return None

        # Get the mean and std of the property from the dataset and unnormalize for metrics
        mean = getattr(
            batch,
            f"{prop}_mean",
            torch.tensor(
                0.0,
                device=pair.ground_truth.device,
                dtype=pair.ground_truth.dtype,
            ),
        )
        std = getattr(
            batch,
            f"{prop}_std",
            torch.tensor(
                1.0,
                device=pair.ground_truth.device,
                dtype=pair.ground_truth.dtype,
            ),
        )

        pair = replace(
            pair,
            ground_truth=(pair.ground_truth * std) + mean,
            predicted=(pair.predicted * std) + mean,
        )
        return pair

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        data = copy.deepcopy(data)
        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y, dtype=torch.float)
        data.y = data.y.float().view(-1)
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.pos = data.pos.float()
        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(self.config.cutoff),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                self.config.max_neighbors
            ),
            pbc=self.config.pbc,
        )

        return data
