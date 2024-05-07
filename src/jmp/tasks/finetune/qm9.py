"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable
from typing import Annotated, Literal, TypeAlias, assert_never, final

import torch
import torch.nn as nn
from ase.data import atomic_masses
from einops import rearrange
from jmp.lightning import Field, TypedConfig
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import FinetuneConfigBase, FinetuneModelBase, OutputHeadInput

QM9Target: TypeAlias = Literal[
    "mu",  # dipole_moment
    "alpha",  # isotropic_polarizability
    "eps_HOMO",  # hOMO
    "eps_LUMO",  # lumo
    "delta_eps",  # homo_lumo_gap
    "R_2_Abs",  # electronicspatial_extent
    "ZPVE",  # zpve
    "U_0",  # energy_U0
    "U",  # energy_U
    "H",  # enthalpy_H
    "G",  # free_energy
    "c_v",  # heat_capacity
    "U_0_ATOM",  # atomization_energy_U0
    "U_ATOM",  # atomization_energy_U
    "H_ATOM",  # atomization_enthalpy_H
    "G_ATOM",  # atomization_free_energy
    "A",  # rotational_constant_A
    "B",  # rotational_constant_B
    "C",  # rotational_constant_C
]


class DefaultOutputHeadConfig(TypedConfig):
    name: Literal["default"] = "default"


class SpatialExtentConfig(TypedConfig):
    name: Literal["spatial_extent"] = "spatial_extent"


OutputHeadConfig: TypeAlias = Annotated[
    DefaultOutputHeadConfig | SpatialExtentConfig,
    Field(discriminator="name"),
]


class QM9Config(FinetuneConfigBase):
    graph_scalar_targets: list[str] = []
    node_vector_targets: list[str] = []

    graph_scalar_reduction: dict[str, Literal["sum", "mean", "max"]] = {
        "mu": "sum",
        "alpha": "sum",
        "eps_HOMO": "sum",
        "eps_LUMO": "sum",
        "delta_eps": "sum",
        "R_2_Abs": "sum",
        "ZPVE": "sum",
        "U_0": "sum",
        "U": "sum",
        "H": "sum",
        "G": "sum",
        "c_v": "sum",
    }

    output_head: OutputHeadConfig = DefaultOutputHeadConfig()

    max_neighbors: int = 30


class SpatialExtentOutputHead(nn.Module):
    @override
    def __init__(self, atomic_masses: Callable[[], torch.Tensor], reduction: str):
        super().__init__()

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default
        assert reduction == "sum", f"reduction must be sum, got {self.reduction=}"

        dim = self.config.backbone.emb_size_atom
        scalar_mlp_layers: list[nn.Module] = []
        for _ in range(self.config.output.num_mlps):
            scalar_mlp_layers.append(nn.Linear(dim, dim, bias=False))
            scalar_mlp_layers.append(self.config.activation_cls())
        scalar_mlp_layers.append(nn.Linear(dim, 1, bias=False))
        self.scalar_mlp = nn.Sequential(*scalar_mlp_layers)

        self.atomic_masses = atomic_masses
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput):
        data = input["data"]
        backbone_output = input["backbone_output"]
        x = self.scalar_mlp(backbone_output["energy"])  # n 1

        batch_size = int(torch.max(data.batch).item() + 1)

        # Get the center of mass
        masses = self.atomic_masses()[data.atomic_numbers]  # n
        center_of_mass = scatter(
            masses.unsqueeze(-1) * data.pos,  # n 3
            data.batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        ) / scatter(
            masses.unsqueeze(-1),
            data.batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )  # b 3

        # Get the squared norm of each position vector
        pos_norm_sq = (
            torch.linalg.vector_norm(
                data.pos - center_of_mass[data.batch],
                dim=-1,
                keepdim=True,
                ord=2,
            )
            ** 2
        )  # n 1
        x = x * pos_norm_sq  # n 1

        # Apply the reduction
        x = scatter(
            x,
            data.batch,
            dim=0,
            dim_size=batch_size,
            reduce=self.reduction,
        )  # (bsz, 1)

        x = rearrange(x, "b 1 -> b")
        return x


@final
class QM9Model(FinetuneModelBase[QM9Config]):
    targets: list[QM9Target] = [
        "mu",
        "alpha",
        "eps_HOMO",
        "eps_LUMO",
        "delta_eps",
        "R_2_Abs",
        "ZPVE",
        "U_0",
        "U",
        "H",
        "G",
        "c_v",
        "U_0_ATOM",
        "U_ATOM",
        "H_ATOM",
        "G_ATOM",
        "A",
        "B",
        "C",
    ]

    atomic_masses: torch.Tensor

    @override
    def __init__(self, hparams: QM9Config):
        super().__init__(hparams)

        self.register_buffer(
            "atomic_masses",
            torch.from_numpy(atomic_masses).float(),
            persistent=False,
        )

    @override
    def validate_config(self, config: QM9Config):
        super().validate_config(config)

        for key in config.graph_scalar_targets:
            assert key in self.targets, f"{key} is not a valid QM9 target"

    @classmethod
    @override
    def config_cls(cls):
        return QM9Config

    @override
    def metric_prefix(self) -> str:
        return "qm9"

    @override
    def construct_graph_scalar_output_head(self, target: str):
        reduction = self.config.graph_scalar_reduction.get(
            target, self.config.graph_scalar_reduction_default
        )
        match self.config.output_head:
            case SpatialExtentConfig():
                # This is only supported for R_2_Abs
                assert (
                    target == "R_2_Abs"
                ), f"{target} is not supported for spatial extent"

                return SpatialExtentOutputHead(lambda: self.atomic_masses, reduction)
            case DefaultOutputHeadConfig():
                return super().construct_graph_scalar_output_head(target)
            case _:
                assert_never(self.config.output_head)

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(8.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
        )

        return data
