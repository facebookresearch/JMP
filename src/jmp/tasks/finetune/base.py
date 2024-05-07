"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import fnmatch
import itertools
import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, assert_never, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jmp.lightning import Base, BaseConfig, Field, LightningModuleBase, TypedConfig
from jmp.lightning.data.balanced_batch_sampler import (
    BalancedBatchSampler,
    DatasetWithSizes,
)
from jmp.lightning.util.typed import TypedModuleDict
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, TypeVar, override

from ...datasets.finetune.base import LmdbDataset
from ...datasets.finetune_pdbbind import PDBBindConfig, PDBBindDataset
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import BackboneConfig
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.early_stopping import EarlyStoppingWithMinLR
from ...modules.ema import EMAConfig
from ...modules.scheduler.gradual_warmup_lr import GradualWarmupScheduler
from ...modules.scheduler.linear_warmup_cos_rlp import (
    PerParamGroupLinearWarmupCosineAnnealingRLPLR,
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
from ...utils.state_dict import load_state_dict
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)
from .metrics import FinetuneMetrics, MetricPair, MetricsConfig

log = getLogger(__name__)

DatasetType: TypeAlias = LmdbDataset


class RLPWarmupConfig(TypedConfig):
    steps: int
    """Number of steps for the warmup"""

    start_lr_factor: float
    """The factor to multiply the initial learning rate by at the start of the warmup"""


class RLPConfig(TypedConfig):
    name: Literal["rlp"] = "rlp"

    monitor: str | None = None
    mode: str | None = None
    patience: int = 10
    factor: float = 0.1
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: str = "rel"
    interval: str = "epoch"
    frequency: int = 1
    warmup: RLPWarmupConfig | None = None

    def _to_linear_warmup_cos_rlp_dict(self):
        """
        Params for PerParamGroupLinearWarmupCosineAnnealingRLPLR's RLP
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        """
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "verbose": True,
        }


class WarmupCosRLPConfig(TypedConfig):
    name: Literal["warmup_cos_rlp"] = "warmup_cos_rlp"

    warmup_steps: int | None = None
    warmup_epochs: int | None = None
    max_steps: int | None = None
    max_epochs: int | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = False

    rlp: RLPConfig

    @override
    def __post_init__(self):
        super().__post_init__()

        assert self.rlp.warmup is None, "RLP warmup is not supported"


LRSchedulerConfig: TypeAlias = Annotated[
    RLPConfig | WarmupCosRLPConfig, Field(discriminator="name")
]


class FreezeConfig(TypedConfig):
    backbone: bool = False
    """Should the backbone be frozen?"""
    embedding: bool = False
    """Should the embedding layer be frozen?"""

    backbone_bases: bool = False
    """Should the basis functions in the backbone be frozen?"""
    backbone_interaction_layers: list[int] | None = None
    """Which interaction layers, if any, in the backbone should be frozen?"""
    backbone_output_layers: list[int] | None = None
    """Which output layers, if any, in the backbone should be frozen?"""

    parameter_patterns: list[str] = []
    """List of parameter patterns to freeze"""


class ParamSpecificOptimizerConfig(TypedConfig):
    name: str | None = None
    """The name of the parameter group for this config"""

    paremeter_patterns: list[str] = []
    """List of parameter patterns to match for this config"""

    optimizer: OptimizerConfig | None = None
    """
    The optimizer config for this parameter group.
    If None, the default optimizer will be used.
    """

    lr_scheduler: LRSchedulerConfig | None = None
    """
    The learning rate scheduler config for this parameter group.
    If None, the default learning rate scheduler will be used.
    """


class CheckpointLoadConfig(TypedConfig):
    ignored_key_patterns: list[str] = []
    """Patterns to ignore when loading the checkpoint"""

    ignored_missing_keys: list[str] = []
    """Keys to ignore if they are missing in the checkpoint"""

    ignored_unexpected_keys: list[str] = []
    """Keys to ignore if they are unexpected in the checkpoint"""

    reset_embeddings: bool = False
    """
    If true, it will reset the embeddings to the initial state
    after loading the checkpoint
    """


class CheckpointBestConfig(TypedConfig):
    monitor: str | None = None
    """
    The metric to monitor for checkpointing.
    If None, the primary metric will be used.
    """
    mode: Literal["min", "max"] | None = None
    """
    The mode for the metric to monitor for checkpointing.
    If None, the primary metric mode will be used.
    """


class EarlyStoppingConfig(TypedConfig):
    monitor: str | None = None
    """
    The metric to monitor for early stopping.
    If None, the primary metric will be used.
    """
    mode: Literal["min", "max"] | None = None
    """
    The mode for the metric to monitor for early stopping.
    If None, the primary metric mode will be used.
    """

    patience: int
    """
    Number of epochs with no improvement after which training will be stopped.
    """

    min_delta: float = 1.0e-8
    """
    Minimum change in the monitored quantity to qualify as an improvement.
    """
    min_lr: float | None = None
    """
    Minimum learning rate. If the learning rate of the model is less than this value,
    the training will be stopped.
    """
    strict: bool = True
    """
    Whether to enforce that the monitored quantity must improve by at least `min_delta`
    to qualify as an improvement.
    """


class BinaryClassificationTargetConfig(TypedConfig):
    name: str
    """The name of the target"""
    num_classes: int
    """The number of classes for the target"""

    pos_weight: float | None = None
    """The positive weight for the target"""

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.num_classes != 2:
            raise ValueError(
                f"Binary classification target {self.name} has {self.num_classes} classes"
            )


class MulticlassClassificationTargetConfig(TypedConfig):
    name: str
    """The name of the target"""
    num_classes: int
    """The number of classes for the target"""

    class_weights: list[float] | None = None
    """The class weights for the target"""
    dropout: float | None = None
    """The dropout probability to use before the output layer"""


class PrimaryMetricConfig(TypedConfig):
    name: str
    """The name of the primary metric"""
    mode: Literal["min", "max"]
    """
    The mode of the primary metric:
    - "min" for metrics that should be minimized (e.g., loss)
    - "max" for metrics that should be maximized (e.g., accuracy)
    """


class TestConfig(TypedConfig):
    save_checkpoint_base_dir: Path | None = None
    """Where to save the checkpoint information for this run (or None to disable)"""

    save_results_base_dir: Path | None = None
    """Where to save the results for this run (or None to disable)"""


class FinetuneLmdbDatasetConfig(CommonDatasetConfig):
    name: Literal["lmdb"] = "lmdb"

    src: Path
    """Path to the LMDB file or directory containing LMDB files."""

    metadata_path: Path | None = None
    """Path to the metadata npz file containing the number of atoms in each structure."""

    def __post_init__(self):
        super().__post_init__()

        # If metadata_path is not provided, assume it is src/metadata.npz
        if self.metadata_path is None:
            self.metadata_path = self.src / "metadata.npz"

    def create_dataset(self):
        return LmdbDataset(src=self.src, metadata_path=self.metadata_path)


class FinetunePDBBindDatasetConfig(PDBBindConfig, CommonDatasetConfig):
    name: Literal["pdbbind"] = "pdbbind"

    def create_dataset(self):
        return PDBBindDataset(task=self.task, split=self.split)


FinetuneDatasetConfig: TypeAlias = Annotated[
    FinetuneLmdbDatasetConfig | FinetunePDBBindDatasetConfig,
    Field(discriminator="name"),
]


class FinetuneConfigBase(BaseConfig):
    train_dataset: FinetuneDatasetConfig | None = None
    """Configuration for the train dataset"""
    val_dataset: FinetuneDatasetConfig | None = None
    """Configuration for the val dataset"""
    test_dataset: FinetuneDatasetConfig | None = None
    """Configuration for the test dataset"""

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

    embedding: EmbeddingConfig = EmbeddingConfig(
        num_elements=BackboneConfig.base().num_elements,
        embedding_size=BackboneConfig.base().emb_size_atom,
    )
    """Configuration for the embedding layer."""
    backbone: BackboneConfig
    """Configuration for the backbone."""
    output: OutputConfig = OutputConfig(num_mlps=5)
    """Configuration for the output head."""

    batch_size: int
    """Batch size to use."""
    eval_batch_size: int | None = None
    """Batch size to use for evaluation. If None, use the same as batch_size."""
    num_workers: int = 8
    """Number of workers to use for data loading."""
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""

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

    primary_metric: PrimaryMetricConfig
    """Primary metric to use for early stopping and checkpointing"""
    early_stopping: EarlyStoppingConfig | None = None
    """Configuration for early stopping"""
    ckpt_best: CheckpointBestConfig | None = CheckpointBestConfig()
    """Configuration for saving the best checkpoint"""

    test: TestConfig | None = None
    """Configuration for test stage"""

    graph_scalar_targets: list[str] = []
    """List of graph scalar targets (e.g., energy)"""
    graph_classification_targets: list[
        BinaryClassificationTargetConfig | MulticlassClassificationTargetConfig
    ] = []
    """List of graph classification targets (e.g., is_metal)"""
    node_vector_targets: list[str] = []
    """List of node vector targets (e.g., force)"""

    @property
    def regression_targets(self):
        """List of all regression targets, i.e., graph scalar and node vector targets"""
        return self.node_vector_targets + self.graph_scalar_targets

    @property
    def all_targets(self):
        """List of all targets, i.e., graph scalar, graph classification, and node vector targets"""
        return (
            self.node_vector_targets
            + self.graph_scalar_targets
            + [target.name for target in self.graph_classification_targets]
        )

    graph_scalar_loss_coefficient_default: float = 1.0
    """Default loss coefficient for graph scalar targets, if not specified in `graph_scalar_loss_coefficients`"""
    graph_classification_loss_coefficient_default: float = 1.0
    """Default loss coefficient for graph classification targets, if not specified in `graph_classification_loss_coefficients`"""
    node_vector_loss_coefficient_default: float = 1.0
    """Default loss coefficient for node vector targets, if not specified in `node_vector_loss_coefficients`"""
    graph_scalar_loss_coefficients: dict[str, float] = {}
    """Loss coefficients for graph scalar targets"""
    graph_classification_loss_coefficients: dict[str, float] = {}
    """Loss coefficients for graph classification targets"""
    node_vector_loss_coefficients: dict[str, float] = {}
    """Loss coefficients for node vector targets"""

    graph_scalar_reduction_default: Literal["sum", "mean", "max"] = "sum"
    """Default reduction method, if not specified in `graph_scalar_reduction`, for computing graph scalar targets from each node's scalar prediction"""
    graph_classification_reduction_default: Literal["sum", "mean", "max"] = "sum"
    """Default reduction method, if  method fornot specified in `graph_classification_reduction`, graph classification targets from each node's classification prediction"""
    node_vector_reduction_default: Literal["sum", "mean", "max"] = "sum"
    """Default reduction method, if not specified in `node_vector_reduction`, for computing node vector targets from each edge's vector prediction"""

    graph_scalar_reduction: dict[str, Literal["sum", "mean", "max"]] = {}
    """Reduction methods for computing graph scalar targets from each node's scalar prediction"""
    graph_classification_reduction: dict[str, Literal["sum", "mean", "max"]] = {}
    """Reduction methods for computing graph classification targets from each node's classification prediction"""
    node_vector_reduction: dict[str, Literal["sum", "mean", "max"]] = {}
    """Reduction methods for computing node vector targets from each edge's vector prediction"""

    normalization: dict[str, NormalizationConfig] = {}
    """Normalization parameters for each target"""

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] | None = None
    """Configuration for parameter-specific optimizers"""

    use_balanced_batch_sampler: bool = False
    """
    Whether to use balanced batch sampler.

    This balances the batches across all distributed nodes (i.e., GPUs, TPUs, nodes, etc.)
    to ensure that each batch has an equal number of **atoms** across all nodes.
    """

    freeze: FreezeConfig = FreezeConfig()
    """Configuration for freezing parameters"""

    ckpt_load: CheckpointLoadConfig = CheckpointLoadConfig()
    """Configuration for behavior when loading checkpoints"""

    shuffle_val: bool = False
    """Whether to shuffle the validation set"""
    shuffle_test: bool = False
    """Whether to shuffle the test set"""

    metrics: MetricsConfig = MetricsConfig()
    """Configuration for metrics"""

    ema: EMAConfig | None = None
    """Configuration for exponential moving average"""

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.use_balanced_batch_sampler:
            assert not self.trainer.use_distributed_sampler, "config.trainer.use_distributed_sampler must be False when using balanced batch sampler"


TConfig = TypeVar("TConfig", bound=FinetuneConfigBase)


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class GraphScalarOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps)
            + [self.config.backbone.num_targets],
            activation=self.config.activation_cls,
        )
        self.reduction = reduction

    @override
    def forward(
        self,
        input: OutputHeadInput,
        *,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n_atoms, 1)
        if scale is not None:
            output = output * scale
        if shift is not None:
            output = output + shift

        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, 1)
        output = rearrange(output, "b 1 -> b")
        return output


class GraphBinaryClassificationOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        classification_config: BinaryClassificationTargetConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        assert (
            classification_config.num_classes == 2
        ), "Only binary classification supported"

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps) + [1],
            activation=self.config.activation_cls,
        )
        self.classification_config = classification_config
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n, num_classes)
        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, num_classes)
        output = rearrange(output, "b 1 -> b")
        return output


class GraphMulticlassClassificationOutputHead(
    Base[TConfig], nn.Module, Generic[TConfig]
):
    @override
    def __init__(
        self,
        config: TConfig,
        classification_config: MulticlassClassificationTargetConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_atom] * self.config.output.num_mlps)
            + [classification_config.num_classes],
            activation=self.config.activation_cls,
        )
        self.classification_config = classification_config
        self.reduction = reduction

        self.dropout = None
        if classification_config.dropout:
            self.dropout = nn.Dropout(classification_config.dropout)

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        n_molecules = int(torch.max(data.batch).item() + 1)

        x = input["backbone_output"]["energy"]
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.out_mlp(x)  # (n, num_classes)
        x = scatter(
            x,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.reduction,
        )  # (bsz, num_classes)
        return x


class NodeVectorOutputHead(Base[TConfig], nn.Module, Generic[TConfig]):
    @override
    def __init__(
        self,
        config: TConfig,
        reduction: str | None = None,
    ):
        super().__init__(config)

        if reduction is None:
            reduction = self.config.graph_scalar_reduction_default

        self.out_mlp = self.mlp(
            ([self.config.backbone.emb_size_edge] * self.config.output.num_mlps)
            + [self.config.backbone.num_targets],
            activation=self.config.activation_cls,
        )
        self.reduction = reduction

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_atoms = data.atomic_numbers.shape[0]

        output = self.out_mlp(backbone_output["forces"])
        output = output * backbone_output["V_st"]  # (n_edges, 3)
        output = scatter(
            output,
            backbone_output["idx_t"],
            dim=0,
            dim_size=n_atoms,
            reduce=self.reduction,
        )
        return output


class FinetuneModelBase(LightningModuleBase[TConfig], Generic[TConfig]):
    @abstractmethod
    def metric_prefix(self) -> str: ...

    @override
    def on_test_end(self):
        super().on_test_end()

        match self.config.test:
            case TestConfig(save_checkpoint_base_dir=Path() as base):
                # The save dir for this run should be base/{metric_prefix()}/{config.name}-{config.id}
                base = base / self.metric_prefix()
                base.mkdir(parents=True, exist_ok=True)
                save_dir = base / f"{self.config.name}-{self.config.id}"
                if save_dir.exists():
                    i = 0
                    while (
                        save_dir := base / f"{self.config.name}-{self.config.id}-{i}"
                    ).exists():
                        i += 1
                save_dir.mkdir(parents=True, exist_ok=True)

                # Get ckpt path from config
                ckpt_path = self.config.meta.get("ckpt_path")
                if ckpt_path is None:
                    raise ValueError(
                        f"Checkpoint path not found in meta: {self.config.meta=}"
                    )
                ckpt_path = Path(ckpt_path)
                if not ckpt_path.exists():
                    raise ValueError(f"Checkpoint path does not exist: {ckpt_path=}")

                # Create a symlink to the checkpoint
                symlink_path = base / f"pretrained-{ckpt_path.name}"
                if symlink_path.exists():
                    raise ValueError(f"Symlink path already exists: {symlink_path=}")
                symlink_path.symlink_to(ckpt_path)

                # Also create an ckptpath.txt file that contains the original ckpt path
                _ = (base / "ckptpath.txt").write_text(
                    str(ckpt_path.resolve().absolute())
                )

                log.critical(f"Saving checkpoint information to {save_dir}")
            case _:
                pass

    def primary_metric(self, split: Literal["train", "val", "test"] | None = "val"):
        config = self.config.primary_metric
        metric = f"{self.metric_prefix()}/{config.name}"
        if split is not None:
            metric = f"{split}/{metric}"
        return metric, config.mode

    def _set_rlp_config_monitors(self):
        match self.config.lr_scheduler:
            case RLPConfig(monitor=None) as rlp_config:
                rlp_config.monitor, rlp_config.mode = self.primary_metric()
            case WarmupCosRLPConfig(rlp=RLPConfig(monitor=None) as rlp_config):
                rlp_config.monitor, rlp_config.mode = self.primary_metric()
            case _:
                pass

    def validate_config(self, config: TConfig):
        assert config.activation.lower() == config.backbone.activation.lower()

        assert config.embedding.num_elements == config.backbone.num_elements
        assert config.embedding.embedding_size == config.backbone.emb_size_atom

        assert config.all_targets, f"No targets specified, {config.all_targets=}"
        for a, b in itertools.combinations(
            [
                config.graph_scalar_targets,
                [target.name for target in config.graph_classification_targets],
                config.node_vector_targets,
            ],
            2,
        ):
            assert (
                set(a) & set(b) == set()
            ), f"Targets must be disjoint, but they are not: {a} and {b}"
        # config.targets = config.graph_scalar_targets + config.node_vector_targets

        if config.graph_scalar_loss_coefficients:
            assert set(config.graph_scalar_loss_coefficients.keys()).issubset(
                set(config.graph_scalar_targets)
            ), (
                f"Loss coefficients must correspond to graph scalar targets, but they "
                f"do not: {config.graph_scalar_loss_coefficients.keys()=} vs "
                f"{config.graph_scalar_targets=}"
            )

        if config.node_vector_loss_coefficients:
            assert set(config.node_vector_loss_coefficients.keys()).issubset(
                set(config.node_vector_targets)
            ), (
                f"Loss coefficients must correspond to node vector targets, but they "
                f"do not: {config.node_vector_loss_coefficients.keys()=} vs "
                f"{config.node_vector_targets=}"
            )

    def _construct_backbone(self):
        log.critical("Using regular backbone")

        backbone = GemNetOCBackbone(self.config.backbone, **dict(self.config.backbone))

        return backbone

    def metrics_provider(
        self,
        prop: str,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
    ) -> MetricPair | None:
        if (pred := preds.get(prop)) is None or (
            target := getattr(batch, prop, None)
        ) is None:
            return None

        if (
            self.config.normalization
            and (norm := self.config.normalization.get(prop)) is not None
        ):
            # Denormalize the predictions and targets
            pred = pred * norm.std + norm.mean
            target = target * norm.std + norm.mean

        return MetricPair(predicted=pred, ground_truth=target)

    @override
    def __init__(self, hparams: TConfig):
        self.validate_config(hparams)
        super().__init__(hparams)

        # Set up callbacks
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        self._set_rlp_config_monitors()

        self.embedding = nn.Embedding(
            num_embeddings=self.config.embedding.num_elements,
            embedding_dim=self.config.embedding.embedding_size,
        )

        self.backbone = self._construct_backbone()
        self.register_shared_parameters(self.backbone.shared_parameters)

        self.construct_output_heads()

        self.train_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_scalar_targets,
            self.config.graph_classification_targets,
            self.config.node_vector_targets,
        )
        self.val_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_scalar_targets,
            self.config.graph_classification_targets,
            self.config.node_vector_targets,
        )
        self.test_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_scalar_targets,
            self.config.graph_classification_targets,
            self.config.node_vector_targets,
        )

        # Sanity check: ensure all named_parameters have requires_grad=True,
        #   otherwise add them to ignored_parameters.
        self.ignored_parameters = set[nn.Parameter]()
        for name, param in self.named_parameters():
            if param.requires_grad:
                continue
            self.ignored_parameters.add(param)
            log.info(f"Adding {name} to ignored_parameters")

        self.process_freezing()

        if (ckpt_best := self.config.ckpt_best) is not None:
            if (monitor := ckpt_best.monitor) is None:
                monitor, mode = self.primary_metric()
            else:
                if (mode := ckpt_best.mode) is None:
                    mode = "min"

            self.register_callback(lambda: ModelCheckpoint(monitor=monitor, mode=mode))

        if (early_stopping := self.config.early_stopping) is not None:
            if (monitor := early_stopping.monitor) is None:
                monitor, mode = self.primary_metric()
            else:
                if (mode := early_stopping.mode) is None:
                    mode = "min"

            self.register_callback(
                lambda: EarlyStoppingWithMinLR(
                    monitor=monitor,
                    mode=mode,
                    patience=early_stopping.patience,
                    min_delta=early_stopping.min_delta,
                    min_lr=early_stopping.min_lr,
                    strict=early_stopping.strict,
                )
            )

        for cls_target in self.config.graph_classification_targets:
            match cls_target:
                case MulticlassClassificationTargetConfig(
                    class_weights=class_weights
                ) if class_weights:
                    self.register_buffer(
                        f"{cls_target.name}_class_weights",
                        torch.tensor(class_weights, dtype=torch.float),
                        persistent=False,
                    )
                case _:
                    pass

    def freeze_parameters(self, parameters: Iterable[nn.Parameter], *, name: str):
        n_params = 0
        for param in parameters:
            if param in self.ignored_parameters:
                continue

            param.requires_grad = False
            n_params += param.numel()
        log.critical(f"Freezing {n_params} parameters in {name}")

    def named_parameters_matching_patterns(self, patterns: list[str]):
        for name, param in self.named_parameters():
            if param in self.ignored_parameters:
                continue
            if (
                matching_pattern := next(
                    (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                    None,
                )
            ) is None:
                continue

            yield name, param, matching_pattern

    def process_freezing(self):
        if self.config.freeze.backbone:
            self.freeze_parameters(self.backbone.parameters(), name="backbone")

        if self.config.freeze.embedding:
            self.freeze_parameters(self.embedding.parameters(), name="embedding")

        if self.config.freeze.backbone_interaction_layers:
            for layer_idx in self.config.freeze.backbone_interaction_layers:
                self.freeze_parameters(
                    self.backbone.int_blocks[layer_idx].parameters(),
                    name=f"backbone.int_blocks[{layer_idx}]",
                )

        if self.config.freeze.backbone_output_layers:
            for layer_idx in self.config.freeze.backbone_output_layers:
                self.freeze_parameters(
                    self.backbone.out_blocks[layer_idx].parameters(),
                    name=f"backbone.out_blocks[{layer_idx}]",
                )

        if self.config.freeze.backbone_bases:
            self.freeze_parameters(
                self.backbone.bases.parameters(), name="backbone.bases"
            )

        if self.config.freeze.parameter_patterns:
            for (
                name,
                param,
                matching_pattern,
            ) in self.named_parameters_matching_patterns(
                self.config.freeze.parameter_patterns
            ):
                param.requires_grad = False
                log.info(f"Freezing {name} (pattern: {matching_pattern})")

        all_parameters = [
            param for param in self.parameters() if param not in self.ignored_parameters
        ]
        num_frozen = sum(
            param.numel() for param in all_parameters if not param.requires_grad
        )
        num_train = sum(
            param.numel() for param in all_parameters if param.requires_grad
        )
        num_total = sum(param.numel() for param in all_parameters)
        percent_frozen = num_frozen / num_total * 100
        log.critical(
            f"Freezing {num_frozen:,} parameters ({percent_frozen:.2f}%) out of "
            f"{num_total:,} total parameters ({num_train:,} trainable)"
        )

    def construct_graph_scalar_output_head(self, target: str) -> nn.Module:
        return GraphScalarOutputHead(
            self.config,
            reduction=self.config.graph_scalar_reduction.get(
                target, self.config.graph_scalar_reduction_default
            ),
        )

    def construct_graph_classification_output_head(
        self,
        target: BinaryClassificationTargetConfig | MulticlassClassificationTargetConfig,
    ) -> nn.Module:
        match target:
            case BinaryClassificationTargetConfig():
                return GraphBinaryClassificationOutputHead(
                    self.config,
                    target,
                    reduction=self.config.graph_classification_reduction.get(
                        target.name, self.config.graph_classification_reduction_default
                    ),
                )
            case MulticlassClassificationTargetConfig():
                return GraphMulticlassClassificationOutputHead(
                    self.config,
                    target,
                    reduction=self.config.graph_classification_reduction.get(
                        target.name, self.config.graph_classification_reduction_default
                    ),
                )
            case _:
                raise ValueError(f"Invalid target: {target}")

    def construct_node_vector_output_head(self, target: str) -> nn.Module:
        return NodeVectorOutputHead(
            self.config,
            reduction=self.config.node_vector_reduction.get(
                target, self.config.node_vector_reduction_default
            ),
        )

    def construct_output_heads(self):
        self.graph_outputs = TypedModuleDict(
            {
                target: self.construct_graph_scalar_output_head(target)
                for target in self.config.graph_scalar_targets
            },
            key_prefix="ft_mlp_",
        )
        self.graph_classification_outputs = TypedModuleDict(
            {
                target.name: self.construct_graph_classification_output_head(target)
                for target in self.config.graph_classification_targets
            },
            key_prefix="ft_mlp_",
        )
        self.node_outputs = TypedModuleDict(
            {
                target: self.construct_node_vector_output_head(target)
                for target in self.config.node_vector_targets
            },
            key_prefix="ft_mlp_",
        )

    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        ignored_key_patterns = self.config.ckpt_load.ignored_key_patterns
        # If we're dumping the backbone's force out heads, then we need to ignore
        #   the unexpected keys for the force out MLPs and force out heads.
        if (
            not self.config.backbone.regress_forces
            or not self.config.backbone.direct_forces
        ):
            ignored_key_patterns.append("out_mlp_F.*")
            for block_idx in range(self.config.backbone.num_blocks + 1):
                ignored_key_patterns.append(f"out_blocks.{block_idx}.scale_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.dense_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.seq_forces.*")

        load_state_dict(
            self.backbone,
            backbone,
            strict=strict,
            ignored_key_patterns=ignored_key_patterns,
            ignored_missing_keys=self.config.ckpt_load.ignored_missing_keys,
            ignored_unexpected_keys=self.config.ckpt_load.ignored_unexpected_keys,
        )
        if not self.config.ckpt_load.reset_embeddings:
            load_state_dict(self.embedding, embedding, strict=strict)
        log.critical("Loaded backbone state dict (backbone and embedding).")

    @override
    def forward(self, data: BaseData):
        atomic_numbers = data.atomic_numbers - 1
        h = self.embedding(atomic_numbers)  # (N, d_model)
        out = cast(GOCBackboneOutput, self.backbone(data, h=h))

        output_head_input: OutputHeadInput = {
            "backbone_output": out,
            "data": data,
        }

        preds = {
            **{
                target: module(output_head_input)
                for target, module in self.graph_outputs.items()
            },
            **{
                target: module(output_head_input)
                for target, module in self.graph_classification_outputs.items()
            },
            **{
                target: module(output_head_input)
                for target, module in self.node_outputs.items()
            },
        }
        return preds

    def compute_losses(self, batch: BaseData, preds: dict[str, torch.Tensor]):
        losses: list[torch.Tensor] = []

        for target in self.config.graph_scalar_targets:
            loss = F.l1_loss(preds[target], batch[target])
            self.log(f"{target}_loss", loss)

            coef = self.config.graph_scalar_loss_coefficients.get(
                target, self.config.graph_scalar_loss_coefficient_default
            )
            loss = coef * loss
            self.log(f"{target}_loss_scaled", loss)

            losses.append(loss)

        for target in self.config.graph_classification_targets:
            match target:
                case BinaryClassificationTargetConfig():
                    y_input = preds[target.name]
                    y_target = batch[target.name].float()
                    pos_weight = None
                    if target.pos_weight is not None:
                        pos_weight = y_input.new_tensor(target.pos_weight)
                    loss = F.binary_cross_entropy_with_logits(
                        y_input, y_target, reduction="sum", pos_weight=pos_weight
                    )
                case MulticlassClassificationTargetConfig():
                    weight = None
                    if target.class_weights:
                        weight = self.get_buffer(f"{target.name}_class_weights")

                    loss = F.cross_entropy(
                        preds[target.name],
                        batch[target.name].long(),
                        weight=weight,
                        reduction="sum",
                    )
                case _:
                    raise ValueError(f"Unknown target type: {target}")
            self.log(f"{target.name}_loss", loss)

            coef = self.config.graph_classification_loss_coefficients.get(
                target.name, self.config.graph_classification_loss_coefficient_default
            )
            loss = coef * loss
            self.log(f"{target.name}_loss_scaled", loss)

            losses.append(loss)

        for target in self.config.node_vector_targets:
            assert preds[target].shape[-1] == 3
            loss = F.pairwise_distance(preds[target], batch[target], p=2.0).mean()
            self.log(f"{target}_loss", loss)

            coef = self.config.node_vector_loss_coefficients.get(
                target, self.config.node_vector_loss_coefficient_default
            )
            loss = coef * loss
            self.log(f"{target}_loss_scaled", loss)

            losses.append(loss)

        loss = sum(losses)
        self.log("loss", loss)

        return loss

    def _rlp_metric(self, config: RLPConfig):
        monitor = config.monitor
        assert monitor is not None, "RLP monitor must be specified."

        metric_prefix = f"val/{self.metric_prefix()}/"
        assert monitor.startswith(
            metric_prefix
        ), f"RLP {monitor=} must start with {metric_prefix}"
        monitor = monitor[len(metric_prefix) :]

        if (
            monitor.endswith("_mae")
            and (mae_metric := self.val_metrics.maes.get(monitor[: -len("_mae")]))
            is not None
        ):
            return mae_metric

        if (
            monitor.endswith("_balanced_accuracy")
            and (
                cls_metric := self.val_metrics.cls_metrics.get(
                    monitor[: -len("_balanced_accuracy")]
                )
            )
            is not None
        ):
            return cls_metric

        avail_mae_metrics = list(self.val_metrics.maes.keys())
        avail_cls_metrics = list(self.val_metrics.cls_metrics.keys())
        raise ValueError(
            f"RLP monitor {monitor} not found in metrics. "
            f"Available MAE metrics: {avail_mae_metrics}. "
            f"Available classification metrics: {avail_cls_metrics}"
        )

    def _cos_rlp_schedulers(self):
        if (lr_schedulers := self.lr_schedulers()) is None:
            log.warning("No LR scheduler found.")
            return

        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        for scheduler in lr_schedulers:
            if isinstance(scheduler, PerParamGroupLinearWarmupCosineAnnealingRLPLR):
                yield scheduler

    def _on_validation_epoch_end_cos_rlp(self, config: WarmupCosRLPConfig):
        rlp_monitor = self._rlp_metric(config.rlp)
        log.info(f"LR scheduler metrics: {rlp_monitor}")

        metric_value: torch.Tensor | None = None
        for scheduler in self._cos_rlp_schedulers():
            if scheduler.is_in_rlp_stage(self.global_step):
                if metric_value is None:
                    metric_value = rlp_monitor.compute()

                log.info(f"LR scheduler is in RLP mode. RLP metric: {metric_value}")
                scheduler.rlp_step(metric_value)

    def _on_train_batch_start_cos_rlp(self):
        for scheduler in self._cos_rlp_schedulers():
            scheduler.on_new_step(self.global_step)

    @override
    def on_train_batch_start(self, batch: BaseData, batch_idx: int):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig():
                self._on_train_batch_start_cos_rlp()
            case _:
                pass

    @override
    def on_validation_epoch_end(self):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig() as config:
                self._on_validation_epoch_end_cos_rlp(config)
            case _:
                pass

    @override
    def training_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"train/{self.metric_prefix()}/"):
            preds = self(batch)

            loss = self.compute_losses(batch, preds)
            self.log_dict(self.train_metrics(batch, preds))

            return loss

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"val/{self.metric_prefix()}/"):
            preds = self(batch)

            self.log_dict(self.val_metrics(batch, preds))

    @override
    def test_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"test/{self.metric_prefix()}/"):
            preds = self(batch)

            self.log_dict(self.test_metrics(batch, preds))

    def outhead_parameters(self):
        head_params = (
            list(self.graph_outputs.parameters())
            + list(self.node_outputs.parameters())
            + list(self.graph_classification_outputs.parameters())
        )
        return head_params

    def backbone_outhead_parameters(
        self,
    ):
        main_params = list(self.parameters())
        head_params = self.outhead_parameters()
        head_params_set = set(head_params)
        main_params = [p for p in main_params if p not in head_params_set]
        return main_params, head_params

    @override
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        match self.config.lr_scheduler:
            case RLPConfig(warmup=RLPWarmupConfig()):
                lr_scheduler = self.lr_schedulers()
                assert isinstance(lr_scheduler, GradualWarmupScheduler)
                if not lr_scheduler.finished:
                    lr_scheduler.step()
            case _:
                pass

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def split_parameters(self, pattern_lists: list[list[str]]):
        all_parameters = list(self.parameters())

        parameters: list[list[torch.nn.Parameter]] = []
        for patterns in pattern_lists:
            matching = [
                p for _, p, _ in self.named_parameters_matching_patterns(patterns)
            ]
            parameters.append(matching)
            # remove matching parameters from all_parameters
            all_parameters = [
                p for p in all_parameters if all(p is not m for m in matching)
            ]

        return parameters, all_parameters

    def _cos_annealing_hparams(
        self, lr_config: WarmupCosRLPConfig, *, lr_initial: float
    ):
        if (warmup_steps := lr_config.warmup_steps) is None:
            if warmup_epochs := lr_config.warmup_epochs:
                assert warmup_epochs >= 0, f"Invalid warmup_epochs: {warmup_epochs}"
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                warmup_steps = warmup_epochs * num_steps_per_epoch
            else:
                warmup_steps = 0
        log.critical(f"Computed warmup_steps: {warmup_steps}")

        if not (max_steps := lr_config.max_steps):
            if max_epochs := lr_config.max_epochs:
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

        log.critical(f"Computed max_steps: {max_steps}")

        assert (
            lr_config.min_lr_factor > 0 and lr_config.min_lr_factor <= 1
        ), f"Invalid {lr_config.min_lr_factor=}"
        min_lr = lr_initial * lr_config.min_lr_factor

        assert (
            lr_config.warmup_start_lr_factor > 0
            and lr_config.warmup_start_lr_factor <= 1
        ), f"Invalid {lr_config.warmup_start_lr_factor=}"
        warmup_start_lr = lr_initial * lr_config.warmup_start_lr_factor

        lr_scheduler_hparams = dict(
            warmup_epochs=warmup_steps,
            max_epochs=max_steps,
            warmup_start_lr=warmup_start_lr,
            eta_min=min_lr,
            should_restart=lr_config.should_restart,
        )

        return lr_scheduler_hparams

    def _construct_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, config: RLPConfig
    ):
        assert config.monitor is not None, f"{config=}"
        assert config.mode is not None, f"{config=}"

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            patience=config.patience,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
            verbose=True,
        )
        if config.warmup is not None:
            optim_lr = float(optimizer.param_groups[0]["lr"])
            warmup_start_lr = optim_lr * config.warmup.start_lr_factor

            lr_scheduler = GradualWarmupScheduler(
                optimizer,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=config.warmup.steps,
                after_scheduler=lr_scheduler,
            )
            return {
                "scheduler": lr_scheduler,
                "monitor": config.monitor,
                "interval": config.interval,
                "frequency": config.frequency,
                "strict": False,
                "reduce_on_plateau": True,
            }
        else:
            return {
                "scheduler": lr_scheduler,
                "monitor": config.monitor,
                "interval": config.interval,
                "frequency": config.frequency,
                "strict": True,
            }

    def configure_optimizers_param_specific_optimizers(
        self, configs: list[ParamSpecificOptimizerConfig]
    ):
        params_list, rest_params = self.split_parameters(
            [c.paremeter_patterns for c in configs]
        )
        optimizer = optimizer_from_config(
            [
                *(
                    (
                        self.config.optimizer if c.optimizer is None else c.optimizer,
                        params,
                        c.name or ",".join(c.paremeter_patterns),
                    )
                    for c, params in zip(configs, params_list)
                ),
                (self.config.optimizer, rest_params, "rest"),
            ],
            base=self.config.optimizer,
        )

        out: dict[str, Any] = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        match lr_config:
            case RLPConfig():
                assert all(
                    c.lr_scheduler is None for c in configs
                ), f"lr_scheduler is not None for some configs: {configs=}"

                if (
                    lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
                ) is not None:
                    out["lr_scheduler"] = lr_scheduler
            case WarmupCosRLPConfig():
                param_group_lr_scheduler_settings = [
                    *(
                        self._cos_annealing_hparams(
                            (
                                lr_config
                                if c.lr_scheduler is None
                                or not isinstance(c.lr_scheduler, WarmupCosRLPConfig)
                                else c.lr_scheduler
                            ),
                            lr_initial=param_group["lr"],
                        )
                        for c, param_group in zip(configs, optimizer.param_groups[:-1])
                    ),
                    self._cos_annealing_hparams(
                        lr_config, lr_initial=optimizer.param_groups[-1]["lr"]
                    ),
                ]

                log.critical(f"{param_group_lr_scheduler_settings=}")
                lr_scheduler = PerParamGroupLinearWarmupCosineAnnealingRLPLR(
                    optimizer,
                    param_group_lr_scheduler_settings,
                    lr_config.rlp._to_linear_warmup_cos_rlp_dict(),
                    max_epochs=next(
                        (s["max_epochs"] for s in param_group_lr_scheduler_settings)
                    ),
                )
                out["lr_scheduler"] = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            case _:
                assert_never(lr_config)

        return out

    @override
    def configure_optimizers(self):
        if self.config.parameter_specific_optimizers is not None:
            return self.configure_optimizers_param_specific_optimizers(
                self.config.parameter_specific_optimizers
            )

        optimizer = optimizer_from_config(
            [(self.config.optimizer, self.parameters())],
        )

        out: dict[str, Any] = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        assert isinstance(
            lr_config, RLPConfig
        ), "Only RLPConfig is supported if `parameter_specific_optimizers` is None"
        if (
            lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
        ) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out

    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    def generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
    ):
        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
        )
        aint_graph = self.process_aint_graph(aint_graph)
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
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        return data

    def create_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> DatasetType | None:
        match split:
            case "train":
                if (config := self.config.train_dataset) is None:
                    return None
            case "val":
                if (config := self.config.val_dataset) is None:
                    return None
            case "test":
                if (config := self.config.test_dataset) is None:
                    return None
            case _:
                assert_never(split)

        dataset = config.create_dataset()
        dataset = wrap_common_dataset(dataset, config)
        return dataset

    def validate_dataset(self, dataset: DatasetType):
        if self.config.use_balanced_batch_sampler:
            assert isinstance(
                dataset, DatasetWithSizes
            ), f"BalancedBatchSampler requires a DatasetWithSizes, but got {type(dataset)}"

    def _transform_cls_data(self, data: BaseData):
        """
        Transforms the classification targets in the given data object based on the configuration.

        For binary classification targets, the target is converted to a float tensor (i.e., 0.0 or 1.0).
        For multiclass classification targets, the target is converted to a long tensor (which is used as
            the class index by `F.cross_entropy`).

        Args:
            data (BaseData): The data object containing the classification targets.

        Returns:
            BaseData: The transformed data object.
        """
        for target_config in self.config.graph_classification_targets:
            match target_config:
                case BinaryClassificationTargetConfig():
                    if (value := getattr(data, target_config.name, None)) is None:
                        log.warning(f"target {target_config.name} not found in data")
                        continue

                    setattr(data, target_config.name, value.float())
                case MulticlassClassificationTargetConfig():
                    if (value := getattr(data, target_config.name, None)) is None:
                        log.warning(f"target {target_config.name} not found in data")
                        continue

                    setattr(data, target_config.name, value.long())
                case _:
                    pass

        return data

    def _apply_dataset_transforms(self, dataset: DatasetType):
        dataset = DT.transform(dataset, self.data_transform)
        if self.config.normalization:
            dataset = DT.transform(dataset, T.normalize(self.config.normalization))
        dataset = DT.transform(dataset, self._transform_cls_data)
        return dataset

    def train_dataset(self):
        if (dataset := self.create_dataset("train")) is None:
            return None
        self.validate_dataset(dataset)
        dataset = self._apply_dataset_transforms(dataset)
        return dataset

    def val_dataset(self):
        if (dataset := self.create_dataset("val")) is None:
            return None
        self.validate_dataset(dataset)
        dataset = self._apply_dataset_transforms(dataset)
        return dataset

    def test_dataset(self):
        if (dataset := self.create_dataset("test")) is None:
            return None
        self.validate_dataset(dataset)
        dataset = self._apply_dataset_transforms(dataset)
        return dataset

    def distributed_sampler(self, dataset: Dataset, shuffle: bool):
        return DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=shuffle,
        )

    @override
    def train_dataloader(self):
        if (dataset := self.train_dataset()) is None:
            raise ValueError("No train dataset")

        sampler = self.distributed_sampler(dataset, shuffle=True)
        if not self.config.use_balanced_batch_sampler:
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        else:
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
            )

        return data_loader

    @override
    def val_dataloader(self):
        if (dataset := self.val_dataset()) is None:
            raise ValueError("No val dataset")

        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_val)
        batch_size = self.config.eval_batch_size or self.config.batch_size
        if not self.config.use_balanced_batch_sampler:
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        else:
            batch_sampler = BalancedBatchSampler(
                sampler,
                batch_size=batch_size,
                device=self.device,
            )
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        return data_loader

    @override
    def test_dataloader(self):
        if (dataset := self.test_dataset()) is None:
            raise ValueError("No test  dataset")

        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_test)
        batch_size = self.config.eval_batch_size or self.config.batch_size
        if not self.config.use_balanced_batch_sampler:
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        else:
            batch_sampler = BalancedBatchSampler(
                sampler,
                batch_size=batch_size,
                device=self.device,
            )
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        return data_loader

    def data_transform(self, data: BaseData):
        return data

    def collate_fn(self, data_list: list[BaseData]):
        return Batch.from_data_list(data_list)
