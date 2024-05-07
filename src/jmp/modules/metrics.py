"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable
from functools import partial
from typing import TypedDict

import torch
import torch.nn as nn
import torchmetrics
from jmp.lightning.util.typed import TypedModuleList
from torch_geometric.data import Batch
from typing_extensions import NotRequired, override

from .transforms.normalize import denormalize_batch
from .transforms.units import VALID_UNITS, Unit, _determine_factor


class MetricConfig(TypedDict):
    idx: int
    additional_units: NotRequired[list[str]]


def _transform(x: torch.Tensor, *, from_: Unit, to: Unit):
    factor = _determine_factor(from_, to)
    return x * factor


class FMTaskMetrics(nn.Module):
    @override
    def __init__(
        self,
        name: str,
        config: MetricConfig,
        num_tasks: int,
        free_atoms_only: bool = True,
    ):
        super().__init__()

        self.name = name
        self.config = config
        self.num_tasks = num_tasks
        self.free_atoms_only = free_atoms_only

        self.energy_mae = torchmetrics.MeanAbsoluteError()
        self.forces_mae = torchmetrics.MeanAbsoluteError()

        if units := self.config.get("additional_units", []):
            for unit in units:
                if unit not in VALID_UNITS:
                    raise ValueError(
                        f"Invalid unit: {unit}. Valid units: {VALID_UNITS}"
                    )
            self.energy_mae_additional = TypedModuleList(
                [torchmetrics.MeanAbsoluteError() for _ in units]
            )
            self.forces_mae_additional = TypedModuleList(
                [torchmetrics.MeanAbsoluteError() for _ in units]
            )

    @override
    def forward(self, batch: Batch, energy: torch.Tensor, forces: torch.Tensor):
        metrics: dict[str, torchmetrics.Metric] = {}

        self._energy_mae(batch, energy, self.energy_mae)
        self._forces_mae(batch, forces, self.forces_mae)

        metrics["energy_mae"] = self.energy_mae
        metrics["forces_mae"] = self.forces_mae

        if additional := self.config.get("additional_units", []):
            for unit, energy_metric, forces_metric in zip(
                additional, self.energy_mae_additional, self.forces_mae_additional
            ):
                assert (
                    unit in VALID_UNITS
                ), f"Invalid unit: {unit}. Valid units: {VALID_UNITS}"
                sanitized_unit = unit.replace("/", "_")
                self._energy_mae(
                    batch,
                    energy,
                    energy_metric,
                    transform=partial(_transform, from_="eV", to=unit),
                )
                self._forces_mae(
                    batch,
                    forces,
                    forces_metric,
                    transform=partial(_transform, from_="eV", to=unit),
                )

                metrics[f"energy_mae_{sanitized_unit}"] = energy_metric
                metrics[f"forces_mae_{sanitized_unit}"] = forces_metric

        return {f"{self.name}/{name}": metric for name, metric in metrics.items()}

    def _forces_mae(
        self,
        batch: Batch,
        forces: torch.Tensor,
        forces_mae: torchmetrics.MeanAbsoluteError,
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        task_idx = self.config["idx"]

        forces_mask = batch.task_mask[:, task_idx]  # (b,)
        forces_mask = forces_mask[batch.batch]  # (n,)
        if self.free_atoms_only:
            forces_mask = forces_mask & ~batch.fixed
        forces_target = batch.force[..., task_idx][forces_mask]
        forces_pred = forces[..., task_idx][forces_mask]
        if transform is not None:
            forces_target = transform(forces_target)
            forces_pred = transform(forces_pred)

        forces_mae(forces_pred, forces_target)

    def _energy_mae(
        self,
        batch: Batch,
        energy: torch.Tensor,
        energy_mae: torchmetrics.MeanAbsoluteError,
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        task_idx = self.config["idx"]

        energy_mask = batch.task_mask[:, task_idx]  # (b,)
        energy_target = batch.y[..., task_idx][energy_mask]  # (b,)
        energy_pred = energy[..., task_idx][energy_mask]  # (b,)
        if transform is not None:
            energy_target = transform(energy_target)
            energy_pred = transform(energy_pred)

        energy_mae(energy_pred, energy_target)


class FMMetrics(nn.Module):
    @override
    def __init__(
        self,
        tasks: dict[str, MetricConfig],
        *,
        denormalize: bool,
        free_atoms_only: bool = True,
    ):
        super().__init__()

        self.denormalize = denormalize
        self.task_metrics = TypedModuleList(
            [
                FMTaskMetrics(
                    name, config, num_tasks=len(tasks), free_atoms_only=free_atoms_only
                )
                for name, config in tasks.items()
            ]
        )

    @override
    def forward(self, batch: Batch, energy: torch.Tensor, forces: torch.Tensor):
        if self.denormalize:
            batch, d = denormalize_batch(batch, {"y": energy, "force": forces})
            energy, forces = d["y"], d["force"]

        metrics: dict[str, torchmetrics.Metric] = {}
        for task_metrics in self.task_metrics:
            metrics.update(task_metrics(batch, energy, forces))
        return metrics
