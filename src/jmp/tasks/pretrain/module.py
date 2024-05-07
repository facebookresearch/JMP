"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from collections.abc import Callable
from functools import cache, partial
from logging import getLogger
from typing import Annotated, Generic, Literal, TypeAlias, assert_never, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce
from jmp.lightning import Base, BaseConfig, Field, LightningModuleBase, TypedConfig
from jmp.lightning.data.balanced_batch_sampler import BalancedBatchSampler
from jmp.lightning.util.typed import TypedModuleDict, TypedModuleList
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.utils import dropout_edge
from torch_scatter import scatter
from torchmetrics import SumMetric
from typing_extensions import TypeVar, override

from ...datasets.pretrain_lmdb import PretrainDatasetConfig as PretrainDatasetConfigBase
from ...datasets.pretrain_lmdb import PretrainLmdbDataset
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import BackboneConfig
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.dataset.concat_dataset import MTDatasetConfig, MTSampledDataset
from ...modules.ema import EMAConfig
from ...modules.metrics import FMMetrics
from ...modules.scheduler.linear_warmup_cosine_annealing import (
    LinearWarmupCosineAnnealingLR,
)
from ...modules.transforms.normalize import NormalizationConfig
from ...utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)

log = getLogger(__name__)


class LinearWarmupCosineAnnealingSchedulerConfig(TypedConfig):
    name: Literal["linear_warmup_cosine_annealing"] = "linear_warmup_cosine_annealing"

    warmup_steps: int = 0
    max_steps: int | None = None
    max_epochs: int | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1


LRSchedulerConfig: TypeAlias = Annotated[
    LinearWarmupCosineAnnealingSchedulerConfig, Field(discriminator="name")
]


class PretrainDatasetConfig(PretrainDatasetConfigBase, CommonDatasetConfig):
    pass


class TaskConfig(TypedConfig):
    name: str
    """Name of the task."""

    train_dataset: PretrainDatasetConfig
    """Train dataset configuration."""

    val_dataset: PretrainDatasetConfig
    """Validation dataset configuration."""

    node_energy_reduction: Literal["sum", "mean"] = "sum"
    """How to reduce the node energy scalar contributions (to get the total energy)."""

    additional_units: list[str] = []
    """Additional units to log for this task."""

    energy_loss_scale: float = 1.0
    """Scale factor for the energy loss."""
    force_loss_scale: float = 1.0
    """Scale factor for the force loss."""

    normalization: dict[str, NormalizationConfig] | None = None
    """
    Normalization to apply to the target values.
    Each key is the name of the target value
    and the value is a dict with the mean and std.
    """


class PretrainConfig(BaseConfig):
    optimizer: OptimizerConfig
    """Optimizer to use."""
    lr_scheduler: LRSchedulerConfig | None = None
    """Learning rate scheduler configuration. If None, no learning rate scheduler is used."""

    activation: Literal[
        "scaled_silu",
        "scaled_swish",
        "silu",
        "swish",
    ] = "scaled_silu"
    """Activation function to use."""

    dropout: float | None = None
    """The dropout rate to use in GemNet."""
    edge_dropout: float | None = None
    """The percentage of edges to drop. If None, no edges are dropped."""

    embedding: EmbeddingConfig = EmbeddingConfig(
        num_elements=BackboneConfig.base().num_elements,
        embedding_size=BackboneConfig.base().emb_size_atom,
    )
    """Configuration for the embedding layer."""
    backbone: BackboneConfig = BackboneConfig.base()
    """Configuration for the backbone."""
    output: OutputConfig = OutputConfig(num_mlps=5)
    """Configuration for the output head."""

    batch_size: int
    """Batch size to use."""
    eval_batch_size: int | None = None
    """Batch size to use for evaluation. If None, use the same as batch_size."""
    num_workers: int
    """Number of workers to use for data loading."""
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""

    shuffle_train: bool = True
    """Should we shuffle the training dataset?"""

    shuffle_val: bool = False
    """Should we shuffle the validation dataset?"""

    @property
    def activation_cls(self):
        match self.activation:
            case "scaled_silu" | "scaled_swish":
                return ScaledSiLU
            case "silu" | "swish":
                return nn.SiLU
            case None:
                return nn.Identity
            case _:
                raise NotImplementedError(
                    f"Activation {self.activation} is not implemented"
                )

    log_task_losses: bool = True
    """Log the loss for each task."""
    log_task_steps_and_epochs: bool = True
    """Log the number of steps and epochs for each task."""

    tasks: list[TaskConfig]
    """List of datasets/tasks to train on."""
    mt_dataset: MTDatasetConfig = MTDatasetConfig(
        balanced=True,
        strict=True,
    )
    """Configuration for the multi-task dataset."""

    exclude_keys: list[str] = [
        "id",  # only oc20,oc22 have this
        "fid",  # only oc20,oc22 have this
        "cell_offsets",  # only oc20 has this
        "edge_index",  # only oc20 has this
        "absolute_idx",  # only ani has this
        "target_pos",  # only ani has this
        "ref_energy",  # only ani/geom have this
        "pbc",  # only ani/transition1x have this
        "oc22",  # only oc22 has this
        "name",
    ]
    """Keys to exclude when creating a batch from a data list."""

    train_on_free_atoms_only: bool = False
    """Train only on free atoms."""

    eval_on_free_atoms_only: bool = True
    """Evaluate only on free atoms."""

    energy_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the energy loss. "sum" or "mean"."""
    force_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the force loss. "sum" or "mean"."""

    structurewise_loss_reduction: bool = True
    """Use the proposed structurewise loss (from the paper) reduction for the force loss."""

    ema: EMAConfig | None = None
    """Configuration for the exponential moving average."""

    @override
    def __post_init__(self):
        super().__post_init__()

        self.trainer.use_distributed_sampler = False

        self.backbone.dropout = self.dropout
        self.backbone.edge_dropout = self.edge_dropout


class Embedding(Base[PretrainConfig], nn.Module):
    @override
    def __init__(self, hparams: PretrainConfig):
        super().__init__(hparams)

        self.atom_embedding = nn.Embedding(
            num_embeddings=self.config.embedding.num_elements,
            embedding_dim=self.config.embedding.embedding_size,
        )

    @override
    def forward(self, data: BaseData):
        atomic_numbers = data.atomic_numbers - 1
        x = self.atom_embedding(atomic_numbers)
        return x


class Output(Base[PretrainConfig], nn.Module):
    @override
    def __init__(self, hparams: PretrainConfig):
        super().__init__(hparams)

        def dims(
            emb_size: int,
            *,
            num_targets: int = self.config.backbone.num_targets,
            num_mlps: int = self.config.output.num_mlps,
        ):
            return ([emb_size] * num_mlps) + [num_targets]

        self.out_energy = TypedModuleList(
            [
                self.mlp(
                    dims(self.config.backbone.emb_size_atom),
                    activation=self.config.activation_cls,
                )
                for _ in self.config.tasks
            ]
        )
        self.out_forces = TypedModuleList(
            [
                self.mlp(
                    dims(self.config.backbone.emb_size_edge),
                    activation=self.config.activation_cls,
                )
                for _ in self.config.tasks
            ]
        )

    @override
    def forward(self, data: BaseData, backbone_out: GOCBackboneOutput):
        energy = backbone_out["energy"]
        forces = backbone_out["forces"]
        V_st = backbone_out["V_st"]
        idx_t = backbone_out["idx_t"]

        batch: torch.Tensor = data.batch
        n_molecules = int(torch.max(batch).item() + 1)
        n_atoms = data.atomic_numbers.shape[0]

        energy_list: list[torch.Tensor] = []
        forces_list: list[torch.Tensor] = []

        for energy_mlp, forces_mlp, task in zip(
            self.out_energy, self.out_forces, self.config.tasks
        ):
            E_t = energy_mlp(energy)  # (n_atoms, 1)
            E_t = scatter(
                E_t,
                batch,
                dim=0,
                dim_size=n_molecules,
                reduce=task.node_energy_reduction,
            )
            energy_list.append(E_t)  # (bsz, 1)

            F_st = forces_mlp(forces)  # (n_edges, 1)
            F_st = F_st * V_st  # (n_edges, 3)
            F_t = scatter(F_st, idx_t, dim=0, dim_size=n_atoms, reduce="sum")
            forces_list.append(F_t)  # (n_atoms, 3)

        E, _ = pack(energy_list, "bsz *")
        F, _ = pack(forces_list, "n_atoms p *")

        return E, F


TConfig = TypeVar(
    "TConfig", bound=PretrainConfig, default=PretrainConfig, infer_variance=True
)


class PretrainModel(LightningModuleBase[TConfig], Generic[TConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return PretrainConfig

    @staticmethod
    def _model_validate_config(config: TConfig):
        assert (
            config.activation.lower() == config.backbone.activation.lower()
        ), f"{config.activation=} != {config.backbone.activation=}"

        assert (
            config.embedding.num_elements == config.backbone.num_elements
        ), f"{config.embedding.num_elements=} != {config.backbone.num_elements=}"
        assert (
            config.embedding.embedding_size == config.backbone.emb_size_atom
        ), f"{config.embedding.embedding_size=} != {config.backbone.emb_size_atom=}"

    def _construct_backbone(self):
        backbone = GemNetOCBackbone(self.config.backbone, **dict(self.config.backbone))
        return backbone

    @override
    def __init__(self, hparams: TConfig):
        self._model_validate_config(hparams)
        super().__init__(hparams)

        # Set up callbacks
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        # Set up the model
        self.embedding = Embedding(self.config)
        self.backbone = self._construct_backbone()
        self.output = Output(self.config)

        # Set up the metrics
        self.train_metrics = FMMetrics(
            {
                task.name: {"idx": idx, "additional_units": task.additional_units}
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
        )
        self.val_metrics = FMMetrics(
            {
                task.name: {"idx": idx, "additional_units": task.additional_units}
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
        )

        # GemNet-OC re-uses some parameters at every layer.
        # We need to make sure that these parameters' gradients are
        # downscaled by the number of layers so that the gradients
        # are not too large.
        if self.backbone.shared_parameters:
            self.register_shared_parameters(self.backbone.shared_parameters)

        self._train_dataset_sizes: list[int] | None = None
        if self.config.log_task_steps_and_epochs:
            task_steps: dict[str, SumMetric] = {}
            for task in self.config.tasks:
                metric = SumMetric()
                metric.persistent(True)
                task_steps[task.name] = metric
            self.task_steps = TypedModuleDict(task_steps)

    def backbone_state_dict(self):
        return {
            "backbone": self.backbone.state_dict(),
            "embedding": self.embedding.atom_embedding.state_dict(),
        }

    @override
    def on_train_batch_start(self, batch: BaseData, batch_idx: int):
        if not self.config.log_task_steps_and_epochs:
            return

        assert self._train_dataset_sizes

        task_mask = batch.task_mask  # (b, t)
        task_idx = reduce(task_mask, "b t -> t", "sum")  # (t,)
        for idx, task in enumerate(self.config.tasks):
            metric = self.task_steps[task.name]
            metric(task_idx[idx])

            step = metric.compute()
            self.log(f"train/{task.name}/step", step)

            epoch = step / self._train_dataset_sizes[idx]
            self.log(f"train/{task.name}/epoch", epoch)

    @override
    def forward(self, batch: BaseData):
        h = self.embedding(batch)
        out: GOCBackboneOutput = self.backbone(batch, h=h)
        return self.output(batch, out)  # (n h), (n p h)

    def _task_idx_onehot(self, task_idx: int):
        return F.one_hot(
            torch.tensor([task_idx], device=self.device, dtype=torch.long),
            num_classes=len(self.config.tasks),
        ).bool()

    def _force_loss(
        self, batch: BaseData, forces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.debug:
            assert forces.shape == batch.force.shape

        pred: torch.Tensor = rearrange(forces, "n p t -> n t p")
        target: torch.Tensor = rearrange(batch.force, "n p t -> n t p")

        mask = batch.task_mask  # b t
        mask = mask[batch.batch]  # n t
        if self.config.train_on_free_atoms_only:
            mask = mask & rearrange(~batch.fixed, "n -> n 1")

        force_loss = F.pairwise_distance(pred, target, p=2.0)  # (n, t)

        if (scale := getattr(batch, "force_scale", None)) is not None:
            # force_loss_scale: (b,)
            scale = scale[batch.batch]  # (n, t)
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            force_loss = force_loss * scale

        if (scale := getattr(batch, "force_scale_node", None)) is not None:
            # force_scale_node: (n, t)
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            force_loss = force_loss * scale

        force_loss = force_loss.masked_fill(~mask, 0.0)

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_force_loss = force_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/force_loss",
                        self._reduce_loss(
                            task_force_loss,
                            task_mask,
                            reduction=self.config.force_loss_reduction,
                        ),
                    )

        # force_loss = self._reduce_force_loss(force_loss, mask)
        return force_loss, mask

    def _energy_loss(
        self, batch: BaseData, energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.task_mask  # (b, h)

        energy_loss = F.l1_loss(energy, batch.y, reduction="none")  # (b, num_tasks)

        if (scale := getattr(batch, "y_scale", None)) is not None:
            energy_loss = energy_loss * scale  # (b, t)

        energy_loss = energy_loss.masked_fill(~mask, 0.0)

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_energy_loss = energy_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/energy_loss",
                        self._reduce_loss(
                            task_energy_loss,
                            task_mask,
                            reduction=self.config.energy_loss_reduction,
                        ),
                    )

        return energy_loss, mask

    @staticmethod
    def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = b.masked_fill(b == 0.0, 1.0)
        return a / b

    def _reduce_loss(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor,
        reduction: Literal["sum", "mean"],
    ):
        match reduction:
            case "sum":
                loss = reduce(loss, "b t -> ", "sum")
            case "mean":
                # loss = reduce(loss, "b t -> ", "sum") / reduce(mask, "b t -> ", "sum")
                loss = self._safe_divide(
                    reduce(loss, "b t -> ", "sum"),
                    reduce(mask, "b t -> ", "sum"),
                )
            case _:
                raise ValueError(f"Unknown redution: {reduction}")

        return loss

    def compute_losses(
        self, batch: BaseData, energy: torch.Tensor, forces: torch.Tensor
    ):
        # Compute the energy loss
        energy_loss, energy_loss_mask = self._energy_loss(
            batch, energy
        )  # (b, t), (b, t)
        energy_loss = self._reduce_loss(
            energy_loss, energy_loss_mask, reduction=self.config.energy_loss_reduction
        )
        self.log("energy_loss", energy_loss)

        # Compute the force loss
        force_loss, force_loss_mask = self._force_loss(batch, forces)
        if self.config.structurewise_loss_reduction:
            # Compute the per-structure force loss
            force_loss = scatter(force_loss, batch.batch, dim=0, reduce="sum")  # (b, t)
            force_loss_mask_natoms = scatter(
                force_loss_mask.float(), batch.batch, dim=0, reduce="sum"
            )  # (b, t)
            force_loss = self._safe_divide(force_loss, force_loss_mask_natoms)  # (b, t)
            force_loss_mask = force_loss_mask_natoms > 0.0  # (b, t)
        force_loss = self._reduce_loss(
            force_loss, force_loss_mask, reduction=self.config.force_loss_reduction
        )
        self.log("force_loss", force_loss)

        # Combine the losses
        loss = energy_loss + force_loss
        self.log("loss", loss)

        return loss

    @override
    def training_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix="train/"):
            energy, forces = self(batch)

            loss = self.compute_losses(batch, energy=energy, forces=forces)
            self.log_dict(self.train_metrics(batch, energy=energy, forces=forces))

            return loss

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix="val/"):
            energy, forces = self(batch)

            metrics = self.val_metrics(batch, energy=energy, forces=forces)
            self.log_dict(metrics)

    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> LRSchedulerConfigType | None:
        match self.config.lr_scheduler:
            case None:
                return None
            case LinearWarmupCosineAnnealingSchedulerConfig() as config:
                if not (max_steps := config.max_steps):
                    if max_epochs := config.max_epochs:
                        _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                        num_steps_per_epoch = math.ceil(
                            self.trainer.num_training_batches
                            / self.trainer.accumulate_grad_batches
                        )
                        max_steps = max_epochs * num_steps_per_epoch
                    else:
                        max_steps = self.trainer.estimated_stepping_batches
                        assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                        max_steps = int(max_steps)

                    log.critical(f"Setting {max_steps=} by default.")

                optim_lr = float(optimizer.param_groups[0]["lr"])
                min_lr = optim_lr * config.min_lr_factor
                warmup_start_lr = optim_lr * config.warmup_start_lr_factor
                lr_scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=config.warmup_steps,
                    max_epochs=max_steps,
                    warmup_start_lr=warmup_start_lr,
                    eta_min=min_lr,
                    last_epoch=config.last_step,
                )
                return {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "strict": True,  # type: ignore
                }

            case _:
                assert_never(self.config.lr_scheduler)

    @override
    def configure_optimizers(self):
        optimizer = optimizer_from_config([(self.config.optimizer, self.parameters())])
        out: OptimizerLRSchedulerConfig = {"optimizer": optimizer}
        if (lr_scheduler := self.configure_lr_scheduler(optimizer)) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out

    def _task_dataset(self, task: TaskConfig, training: bool):
        config = task.val_dataset if not training else task.train_dataset
        dataset = PretrainLmdbDataset(config)
        dataset = wrap_common_dataset(dataset, config)

        # Apply data transform to the dataset
        if (transform := getattr(self, f"{task.name}_transform")) is None:
            raise ValueError(f"Transform not defined for {task.name}")
        transform = cast(
            Callable[[BaseData], BaseData], partial(transform, training=training)
        )

        # Apply normalization to the dataset
        if task.normalization:
            log.info(f"Normalizing {task.name} with {task.normalization}")
            transform = T.compose([transform, T.normalize(task.normalization)])

        dataset = DT.transform(dataset, transform)

        return dataset

    def _construct_fm_datasets(self, training: bool):
        datasets = []
        for task in self.config.tasks:
            datasets.append(self._task_dataset(task, training=training))
        return datasets

    @cache
    def train_dataset(self):
        datasets = self._construct_fm_datasets(training=True)
        self._train_dataset_sizes = [len(d) for d in datasets]
        # if self.config.log_task_steps_and_epochs:
        dataset = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=False,
            num_tasks=len(self.config.tasks),
        )
        dataset = DT.transform(dataset, self.train_data_transform)
        return dataset

    def representative_batch_for_testing(self, *, n: int, start_index: int = 0):
        dataset = self.train_dataset()
        data_list = dataset.representative_batch_for_testing(
            n=n, start_index=start_index
        )
        data_list = [self.train_data_transform(data) for data in data_list]
        return data_list

    @cache
    def val_dataset(self):
        datasets = self._construct_fm_datasets(training=False)
        dataset = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=True,
            num_tasks=len(self.config.tasks),
        )
        dataset = DT.transform(dataset, self.val_data_transform)
        return dataset

    def collate_fn(self, data_list: list[BaseData]):
        return Batch.from_data_list(data_list, exclude_keys=self.config.exclude_keys)

    def distributed_sampler(self, dataset: Dataset, shuffle: bool):
        return DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=shuffle,
        )

    @override
    def train_dataloader(self):
        dataset = self.train_dataset()
        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_train)
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return data_loader

    @override
    def val_dataloader(self):
        dataset = self.val_dataset()
        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_val)
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return data_loader

    def _task_config(self, name: str):
        return next((task for task in self.config.tasks if task.name == name), None)

    @staticmethod
    def _to_int(value):
        return int(value.item() if torch.is_tensor(value) else value)

    def train_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data

    def val_data_transform(self, data: BaseData):
        data = self.data_transform(data)
        return data

    def data_transform(self, data: BaseData):
        data.y = (
            data.y.float()
            if torch.is_tensor(data.y)
            else torch.tensor(data.y, dtype=torch.float)
        )

        data.fixed = data.fixed.bool()
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = self._to_int(data.natoms)
        data.sid = self._to_int(data.sid)
        for graph_type in ["main", "a2a", "a2ee2a", "qint"]:
            key = f"{graph_type}_num_neighbors"
            setattr(data, key, self._to_int(data[key]))

        for attr in ("y", "force"):
            key = f"{attr}_scale"
            if not hasattr(data, key):
                raise ValueError(f"{key=} not found in data")

        # make all tensors contiguous
        for key in data.keys():
            if not torch.is_tensor(data[key]):
                continue

            data[key] = data[key].contiguous()

        return data

    def _process_aint_graph(self, graph: Graph, *, training: bool):
        if self.config.edge_dropout:
            graph["edge_index"], mask = dropout_edge(
                graph["edge_index"],
                p=self.config.edge_dropout,
                training=training,
            )
            graph["distance"] = graph["distance"][mask]
            graph["vector"] = graph["vector"][mask]
            graph["cell_offset"] = graph["cell_offset"][mask]

            if "id_swap_edge_index" in graph:
                graph["id_swap_edge_index"] = graph["id_swap_edge_index"][mask]

        return graph

    def _generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        *,
        training: bool,
    ):
        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
        )
        aint_graph = self._process_aint_graph(aint_graph, training=training)
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
        qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        qint_graph = tag_mask(data, qint_graph, tags=self.config.backbone.qint_tags)

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }

        for graph_type, graph in graphs.items():
            graph["num_neighbors"] = graph["edge_index"].shape[1]
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        return data

    def _initial_data_transform(self, data: BaseData):
        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)

        return data

    def oc20_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("oc20")
        ) is not None, "OC20 task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()

        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=True,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        return data

    def oc22_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("oc22")
        ) is not None, "OC22 task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()
        try:
            data.y = torch.tensor(float(data.y)).view(-1)
        except BaseException:
            data.y = torch.tensor(float(data.y_relaxed)).view(-1)
        data.name = "oc22"

        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=True,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        return data

    @staticmethod
    def _set_inf_cell(data: BaseData, max_length: float = 1000.0):
        data.cell = (torch.eye(3) * max_length).unsqueeze(dim=0)
        return data

    def ani1x_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("ani1x")
        ) is not None, "ANI1x task is not configured"

        data.y = data.y.view(-1).float()
        if not hasattr(data, "sid"):
            data.sid = data.absolute_idx
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes

        # data.fixed = torch.ones(data.natoms)
        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
        data.name = "ani1x"

        data = self._set_inf_cell(data)
        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(8.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        return data

    def transition1x_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("transition1x")
        ) is not None, "Transition1x task is not configured"

        data.y = data.y.view(-1).float()
        if not hasattr(data, "sid"):
            data.sid = data.absolute_idx
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes

        # data.fixed = torch.ones(data.natoms)
        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
        data.name = "transition1x"

        data = self._set_inf_cell(data)
        data = self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(8.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
            training=training,
        )

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        return data
