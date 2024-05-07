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
from .energy_forces_base import EnergyForcesConfigBase, EnergyForcesModelBase

MD22Molecule: TypeAlias = Literal[
    "Ac-Ala3-NHMe",
    "DHA",
    "stachyose",
    "AT-AT",
    "AT-AT-CG-CG",
    "buckyball-catcher",
    "double-walled_nanotube",
]


class MD22Config(EnergyForcesConfigBase):
    molecule: MD22Molecule

    graph_scalar_targets: list[str] = ["y"]
    node_vector_targets: list[str] = ["force"]

    graph_scalar_loss_coefficients: dict[str, float] = {"y": 1.0}
    node_vector_loss_coefficients: dict[str, float] = {"force": 100.0}


@final
class MD22Model(EnergyForcesModelBase[MD22Config]):
    @classmethod
    @override
    def config_cls(cls):
        return MD22Config

    @override
    def metric_prefix(self) -> str:
        return f"md22/{self.config.molecule}"

    @override
    def generate_graphs_transform(self, data: BaseData):
        return self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
        )

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        data.y = data.pop("y").view(-1).float()
        data.atomic_numbers = data.pop("atomic_numbers").long()
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.cell = (torch.eye(3) * 1000.0).unsqueeze(dim=0)
        return data
