"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Literal, TypeAlias, final

import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from . import base

MatbenchDataset: TypeAlias = Literal[
    "jdft2d",
    "phonons",
    "dielectric",
    "log_gvrh",
    "log_kvrh",
    "perovskites",
    "mp_gap",
    "mp_e_form",
    "mp_is_metal",
]
MatbenchFold: TypeAlias = Literal[0, 1, 2, 3, 4]


class MatbenchConfig(base.FinetuneConfigBase):
    dataset: MatbenchDataset
    graph_scalar_targets: list[str] = ["y"]
    node_vector_targets: list[str] = []

    graph_scalar_reduction_default: Literal["sum", "mean", "max"] = "mean"

    fold: MatbenchFold = 0
    mp_e_form_dev: bool = True

    conditional_max_neighbors: bool = False


@final
class MatbenchModel(base.FinetuneModelBase[MatbenchConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return MatbenchConfig

    @override
    def metric_prefix(self) -> str:
        return f"matbench/{self.config.dataset}"

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)

        if self.config.dataset == "mp_is_metal":
            data.y = data.y.bool()

        data.atomic_numbers = data.atomic_numbers.long()
        assert data.num_nodes is not None
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.pos = data.pos.float()

        max_neighbors = 30
        if self.config.conditional_max_neighbors:
            if data.natoms > 300:
                max_neighbors = 5
            elif data.natoms > 200:
                max_neighbors = 10
            else:
                max_neighbors = 30

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
            pbc=True,
        )
        return data
