"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import ExitStack
from logging import getLogger
from typing import Any, Generic, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce
from jmp.lightning import TypedConfig
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypeVar, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...models.gemnet.layers.force_scaler import ForceScaler
from ...modules.dataset import dataset_transform as DT
from .base import FinetuneConfigBase, FinetuneModelBase

log = getLogger(__name__)


class PretrainOutputHeadConfig(TypedConfig):
    enabled: bool = False

    num_pretrain_heads: int = 4
    energy_reduction: str = "sum"

    direct_forces: bool = False
    gradient_forces: bool = False

    combine_strategy: str = "mean"


class EnergyForcesConfigBase(FinetuneConfigBase):
    graph_scalar_targets: list[str] = ["y"]
    node_vector_targets: list[str] = ["force"]

    graph_scalar_loss_coefficients: dict[str, float] = {"y": 1.0}
    node_vector_loss_coefficients: dict[str, float] = {"force": 100.0}

    gradient_forces: bool = False
    model_type: Literal["energy", "forces", "energy_forces"] = "energy_forces"

    pretrain_output_head: PretrainOutputHeadConfig = PretrainOutputHeadConfig()

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.gradient_forces:
            assert (
                not self.trainer.inference_mode
            ), "Gradient forces requires trainer.inference_mode = False"


TConfig = TypeVar("TConfig", bound=EnergyForcesConfigBase, infer_variance=True)


class EnergyForcesModelBase(
    FinetuneModelBase[TConfig], nn.Module, ABC, Generic[TConfig]
):
    @override
    def validate_config(self, config: TConfig):
        super().validate_config(config)

        assert config.model_type in ("energy", "forces", "energy_forces"), (
            f"{config.model_type=} must be one of these values: "
            "energy, forces, energy_forces"
        )

        # if config.gradient_forces:
        #     assert (
        #         not config.trainer.inference_mode
        #     ), f"config.trainer.inference_mode must be False when {config.gradient_forces=}"

    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)

        if self.config.gradient_forces:
            self.force_scaler = ForceScaler()

    @override
    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        output: Mapping[str, Any] | None = None,
        strict: bool = True,
    ):
        super().load_backbone_state_dict(
            backbone=backbone, embedding=embedding, strict=strict
        )

        if self.config.pretrain_output_head.enabled:
            assert (
                output is not None
            ), "output must be provided when pretrain_output_head is enabled"
            self._load_pretrain_output_state_dict(output)

    def construct_energy_head(self):
        def dims(
            emb_size: int,
            *,
            num_targets: int = self.config.backbone.num_targets,
            num_mlps: int = self.config.output.num_mlps,
        ):
            return ([emb_size] * num_mlps) + [num_targets]

        self.out_energy = self.mlp(
            dims(self.config.backbone.emb_size_atom),
            activation=self.config.activation_cls,
            bias=False,
        )

    @override
    def construct_output_heads(self):
        def dims(
            emb_size: int,
            *,
            num_targets: int = self.config.backbone.num_targets,
            num_mlps: int = self.config.output.num_mlps,
        ):
            return ([emb_size] * num_mlps) + [num_targets]

        self.out_energy = None
        if (
            self.config.model_type in ("energy", "energy_forces")
            or self.config.gradient_forces
        ):
            self.construct_energy_head()

        self.out_forces = None
        if (
            self.config.model_type in ("forces", "energy_forces")
            and not self.config.gradient_forces
        ):
            self.out_forces = self.mlp(
                dims(self.config.backbone.emb_size_edge),
                activation=self.config.activation_cls,
            )

    @override
    def outhead_parameters(self):
        head_params: list[nn.Parameter] = []
        if self.out_energy is not None:
            head_params.extend(self.out_energy.parameters())
        if self.out_forces is not None:
            head_params.extend(self.out_forces.parameters())
        return head_params

    def combine_outputs(
        self,
        energy_list: list[torch.Tensor],
        forces_list: list[torch.Tensor],
    ):
        energy, _ = pack(energy_list, "b *")  # (bsz, T)
        forces, _ = pack(forces_list, "n p *")  # (N, 3, T)

        match self.config.pretrain_output_head.combine_strategy:
            case "mean":
                energy = reduce(energy, "b T -> b", "mean")
                forces = reduce(forces, "n p T -> n p", "mean")
            case _:
                raise ValueError(
                    f"Unknown combine strategy: {self.config.pretrain_output_head.combine_strategy}"
                )

        return energy, forces

    @override
    def forward(self, data: BaseData):
        preds: dict[str, torch.Tensor] = {}
        with ExitStack() as stack:
            if self.config.gradient_forces or (
                self.config.pretrain_output_head.enabled
                and self.config.pretrain_output_head.gradient_forces
            ):
                stack.enter_context(torch.inference_mode(mode=False))
                stack.enter_context(torch.enable_grad())

                data.pos.requires_grad_(True)
                data = self.generate_graphs_transform(data)

            atomic_numbers = data.atomic_numbers - 1
            h = self.embedding(atomic_numbers)
            out: GOCBackboneOutput = self.backbone(data, h=h)

            n_molecules = int(torch.max(data.batch).item() + 1)
            n_atoms = data.atomic_numbers.shape[0]

            if self.out_energy is not None:
                output = self.out_energy(out["energy"])  # (n_atoms, 1)

                # TODO: set reduce to config
                output = scatter(
                    output,
                    data.batch,
                    dim=0,
                    dim_size=n_molecules,
                    reduce="sum",
                )
                preds["y"] = rearrange(output, "b 1 -> b")

            if self.out_forces is not None:
                output = self.out_forces(out["forces"])
                output = output * out["V_st"]
                output = scatter(
                    output, out["idx_t"], dim=0, dim_size=n_atoms, reduce="sum"
                )
                preds["force"] = output

            if self.config.gradient_forces:
                assert "force" not in preds, f"force already in preds: {preds.keys()}"
                assert (
                    energy := preds.get("y")
                ) is not None, f"energy not in preds: {preds.keys()}"
                preds["force"] = self.force_scaler.calc_forces_and_update(
                    energy, data.pos
                )

            if self.config.pretrain_output_head.enabled:
                pretrain_energies, pretrain_forces = self.pretrain_output(
                    data, out
                )  # (bsz, T), (N, 3, T)

                pretrain_energies = cast(list[torch.Tensor], pretrain_energies)
                pretrain_forces = cast(list[torch.Tensor], pretrain_forces)

                gradient_forces: list[torch.Tensor] = []
                if self.config.pretrain_output_head.gradient_forces:
                    for energy in pretrain_energies:
                        # energy: (bsz)
                        forces = self.force_scaler.calc_forces_and_update(
                            energy, data.pos
                        )  # (N, 3)
                        gradient_forces.append(forces)

                all_energies = [preds["y"]] + pretrain_energies
                all_forces = [preds["force"]] + pretrain_forces + gradient_forces

                preds["y"], preds["force"] = self.combine_outputs(
                    all_energies, all_forces
                )

        return preds

    @override
    def compute_losses(self, batch: BaseData, preds: dict[str, torch.Tensor]):
        losses: list[torch.Tensor] = []

        if self.config.model_type in ("energy", "energy_forces"):
            loss = F.l1_loss(preds["y"], batch["y"])
            self.log("y_loss", loss)

            coef = self.config.graph_scalar_loss_coefficients.get(
                "y", self.config.graph_scalar_loss_coefficient_default
            )
            loss = coef * loss
            self.log("y_loss_scaled", loss)
            losses.append(loss)

        if self.config.model_type in ("forces", "energy_forces"):
            assert preds["force"].shape[-1] == 3, f"{preds['force'].shape=}"

            loss = F.pairwise_distance(preds["force"], batch["force"], p=2.0).mean()
            self.log("force_loss", loss)

            coef = self.config.node_vector_loss_coefficients.get(
                "force", self.config.node_vector_loss_coefficient_default
            )
            loss = coef * loss
            self.log("force_loss_scaled", loss)

            losses.append(loss)

        loss = sum(losses)
        self.log("loss", loss)

        return loss

    @abstractmethod
    def generate_graphs_transform(self, data: BaseData) -> BaseData: ...

    def _generate_graphs_transform(self, data: BaseData):
        if self.config.gradient_forces:
            # We need to compute the graphs in the forward method
            # so that we can compute the forces using the energy
            # and the positions.
            return data
        return self.generate_graphs_transform(data)

    @override
    def train_dataset(self):
        if (dataset := super().train_dataset()) is None:
            return None

        dataset = DT.transform(dataset, transform=self._generate_graphs_transform)
        return dataset

    @override
    def val_dataset(self):
        if (dataset := super().val_dataset()) is None:
            return None

        dataset = DT.transform(dataset, transform=self._generate_graphs_transform)
        return dataset

    @override
    def test_dataset(self):
        if (dataset := super().test_dataset()) is None:
            return None

        dataset = DT.transform(dataset, transform=self._generate_graphs_transform)
        return dataset
