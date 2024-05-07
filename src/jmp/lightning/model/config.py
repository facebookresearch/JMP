"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import string
import time
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.plugins.layer_sync import LayerSync
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.profilers import Profiler
from typing_extensions import TypeVar, deprecated, override

from ..config import Field, TypedConfig

logger = getLogger(__name__)


class IdSeedWarning(Warning):
    pass


class BaseProfilerConfig(TypedConfig, ABC):
    dirpath: str | Path | None = None
    """
    Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
        ``trainer.log_dir`` (from :class:`~lightning.pytorch.loggers.tensorboard.TensorBoardLogger`)
        will be used.
    """
    filename: str | None = None
    """
    If present, filename where the profiler results will be saved instead of printing to stdout.
        The ``.txt`` extension will be used automatically.
    """

    @abstractmethod
    def construct_profiler(self) -> Profiler: ...


class SimpleProfilerConfig(BaseProfilerConfig):
    kind: Literal["simple"] = "simple"

    extended: bool = True
    """
    If ``True``, adds extra columns representing number of calls and percentage of
        total time spent onrespective action.
    """

    @override
    def construct_profiler(self):
        from lightning.pytorch.profilers.simple import SimpleProfiler

        return SimpleProfiler(
            extended=self.extended,
            dirpath=self.dirpath,
            filename=self.filename,
        )


class AdvancedProfilerConfig(BaseProfilerConfig):
    kind: Literal["advanced"] = "advanced"

    line_count_restriction: float = 1.0
    """
    This can be used to limit the number of functions
        reported for each action. either an integer (to select a count of lines),
        or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
    """

    @override
    def construct_profiler(self):
        from lightning.pytorch.profilers.advanced import AdvancedProfiler

        return AdvancedProfiler(
            line_count_restriction=self.line_count_restriction,
            dirpath=self.dirpath,
            filename=self.filename,
        )


class PyTorchProfilerConfig(BaseProfilerConfig):
    kind: Literal["pytorch"] = "pytorch"

    group_by_input_shapes: bool = False
    """Include operator input shapes and group calls by shape."""

    emit_nvtx: bool = False
    """
    Context manager that makes every autograd operation emit an NVTX range
        Run::

            nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

        To visualize, you can either use::

            nvvp trace_name.prof
            torch.autograd.profiler.load_nvprof(path)
    """

    export_to_chrome: bool = True
    """
    Whether to export the sequence of profiled operators for Chrome.
        It will generate a ``.json`` file which can be read by Chrome.
    """

    row_limit: int = 20
    """
    Limit the number of rows in a table, ``-1`` is a special value that
        removes the limit completely.
    """

    sort_by_key: str | None = None
    """
    Attribute used to sort entries. By default
        they are printed in the same order as they were registered.
        Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
        ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
        ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
    """

    record_module_names: bool = True
    """Whether to add module names while recording autograd operation."""

    table_kwargs: dict[str, Any] | None = None
    """Dictionary with keyword arguments for the summary table."""

    additional_profiler_kwargs: dict[str, Any] = {}
    """Keyword arguments for the PyTorch profiler. This depends on your PyTorch version"""

    @override
    def construct_profiler(self):
        from lightning.pytorch.profilers.pytorch import PyTorchProfiler

        return PyTorchProfiler(
            group_by_input_shapes=self.group_by_input_shapes,
            emit_nvtx=self.emit_nvtx,
            export_to_chrome=self.export_to_chrome,
            row_limit=self.row_limit,
            sort_by_key=self.sort_by_key,
            record_module_names=self.record_module_names,
            table_kwargs=self.table_kwargs,
            dirpath=self.dirpath,
            filename=self.filename,
            **self.additional_profiler_kwargs,
        )


ProfilerConfig: TypeAlias = Annotated[
    SimpleProfilerConfig | AdvancedProfilerConfig | PyTorchProfilerConfig,
    Field(discriminator="kind"),
]


class EnvironmentConfig(TypedConfig):
    cwd: str | None = None

    python_executable: str | None = None
    python_path: list[str] | None = None
    python_version: str | None = None

    config: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    data: dict[str, Any] | None = None

    slurm: dict[str, Any] | None = None

    log_dir: str | None = None

    seed: int | None = None
    seed_workers: bool | None = None

    sweep_id: str | None = None
    sweep_config: dict[str, Any] | None = None


class WandbWatchConfig(TypedConfig):
    enabled: bool = True
    """Enable watching the model for wandb."""

    log: str | None = None
    log_graph: bool = True
    log_freq: int = 100


class WandbLoggingConfig(TypedConfig):
    enabled: bool = True
    """Enable logging to wandb."""

    log_model: bool | str = False
    """
    Whether to log the model checkpoints to wandb.
    Valid values are:
        - False: Do not log the model checkpoints.
        - True: Log the latest model checkpoint.
        - "all": Log all model checkpoints.
    """

    watch: WandbWatchConfig = WandbWatchConfig()
    """WandB model watch configuration. Used to log model architecture, gradients, and parameters."""


class CSVLoggingConfig(TypedConfig):
    enabled: bool = True
    """Enable logging to CSV files."""


class TensorboardLoggingConfig(TypedConfig):
    enabled: bool = False
    """Enable logging to tensorboard."""


class LoggingConfig(TypedConfig):
    enabled: bool = True
    """Enable logging."""

    log_lr: bool | Literal["step", "epoch"] = True
    """If enabled, will register a `LearningRateMonitor` callback to log the learning rate to the logger."""
    log_epoch: bool = True
    """If enabled, will log the fractional epoch number to the logger."""

    wandb: WandbLoggingConfig = WandbLoggingConfig()
    """WandB configuration"""

    csv: CSVLoggingConfig = CSVLoggingConfig()
    """CSV configuration"""

    tensorboard: TensorboardLoggingConfig = TensorboardLoggingConfig()
    """Tensorboard configuration"""


class GradientClippingConfig(TypedConfig):
    enabled: bool = True
    """Enable gradient clipping."""
    value: int | float
    """Value to use for gradient clipping."""
    algorithm: Literal["value", "norm"] = "norm"
    """Norm type to use for gradient clipping."""


class GradientSkippingConfig(TypedConfig):
    enabled: bool = True
    """Enable gradient skipping."""
    norm_type: str | float = 2.0
    """Norm type to use for gradient skipping."""
    threshold: float = float("inf")
    """Threshold to use for gradient skipping."""
    start_after_n_steps: int | None = 100
    """Number of steps to wait before starting gradient skipping."""


class OptimizerConfig(TypedConfig):
    grad_finite_checks: bool = False
    """If enabled, will check that the gradients are finite after each backward pass."""
    grad_none_checks: bool = False
    """If enabled, will check that the gradients are not None after each backward pass."""

    log_grad_norm: bool | str | float = False
    """If enabled, will log the gradient norm (averaged across all model parameters) to the logger."""
    log_grad_norm_per_param: bool | str | float = False
    """If enabled, will log the gradient norm for each model parameter to the logger."""

    log_param_norm: bool | str | float = False
    """If enabled, will log the parameter norm (averaged across all model parameters) to the logger."""
    log_param_norm_per_param: bool | str | float = False
    """If enabled, will log the parameter norm for each model parameter to the logger."""

    gradient_clipping: GradientClippingConfig | None = None
    """Gradient clipping configuration, or None to disable gradient clipping."""

    gradient_skipping: GradientSkippingConfig | None = None
    """Gradient skipping configuration, or None to disable gradient skipping."""


class PythonLogging(TypedConfig):
    log_level: (
        Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"] | None
    ) = None
    """Log level to use for the Python logger (or None to use the default)."""

    rich: bool = True
    """If enabled, will use the rich library to format the Python logger output."""
    rich_tracebacks: bool = True
    """If enabled, will use the rich library to format the Python logger tracebacks."""

    lovely_tensors: bool = True
    """If enabled, will use the lovely-tensors library to format PyTorch tensors."""
    lovely_numpy: bool = False
    """If enabled, will use the lovely-numpy library to format numpy arrays. False by default as it causes some issues with other libaries."""


TPlugin = TypeVar(
    "TPlugin",
    Precision,
    ClusterEnvironment,
    CheckpointIO,
    LayerSync,
    infer_variance=True,
)


@runtime_checkable
class PluginConfigProtocol(Protocol[TPlugin]):
    def construct_plugin(self) -> TPlugin: ...


class TrainerConfig(TypedConfig):
    python_logging: PythonLogging = PythonLogging()
    """Python logging configuration options."""

    logging: LoggingConfig = LoggingConfig()
    """Logging (e.g., WandB logging) configuration options."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """Optimizer configuration options."""

    seed: int | None = 0
    """Seed for the random number generator. If None, will use a random seed."""
    seed_workers: bool = False
    """Whether to seed the workers of the dataloader."""
    default_ckpt_path: str | None = None
    """Default checkpoint path to use when loading a checkpoint. "last" will load the last checkpoint. "hpc" will load the SLURM pre-empted checkpoint."""

    auto_wrap_trainer: bool = True
    """If enabled, will automatically wrap the `run` function with a `Trainer.context()` context manager. Should be `True` most of the time."""
    auto_set_default_root_dir: bool = True
    """If enabled, will automatically set the default root dir to [cwd/lightning_logs/<id>/]. Should be `True` most of the time."""
    auto_set_loggers: bool = True
    """If enabled, will automatically set the loggers to [WandbLogger, CSVLogger, TensorboardLogger] as defined in `config.logging`. Should be `True` most of the time."""
    checkpoint_last_by_default: bool = True
    """If enabled, will update the trainer to save the last checkpoint by default."""
    on_exception_checkpoint: bool = True
    """If enabled, will checkpoint the model when an exception is thrown during training."""
    auto_add_trainer_finalizer: bool = True
    """If enabled, will automatically finalize the trainer (e.g., call `wandb.finish()`) when the run ends. Should be `True` most of the time."""
    enable_logger_validation: bool = True
    """If enabled, will validate loggers. This makes sure that the logger's log_dirs are correct given the current config id. Should be `True` most of the time."""

    supports_skip_batch_exception: bool = True
    """If enabled, the model supports skipping an entire batch by throwing a `SkipBatch` exception."""
    supports_shared_parameters: bool = True
    """If enabled, the model supports scaling the gradients of shared parameters that are registered using `LightningModuleBase.register_shared_parameters(...)`"""
    supports_parameter_hooks: bool = True
    """If enabled, the model supports registering parameter hooks using `LightningModuleBase.register_parameter_hook(...)`"""
    log_batch_info_on_error: bool = False
    """If enabled, will log the batch info (e.g. batch index, batch object, etc.) when an exception is thrown during training."""
    reduce_lr_on_plateau_sanity_checks: Literal["disable", "error", "warn"] = "error"
    """
    Valid values are: "disable", "warn", "error"
    If enabled, will do some sanity checks if the `ReduceLROnPlateau` scheduler is used:
        - If the `interval` is step, it makes sure that validation is called every `frequency` steps.
        - If the `interval` is epoch, it makes sure that validation is called every `frequency` epochs.
    """

    additional_trainer_kwargs: dict[str, Any] = {}
    """Additional keyword arguments to pass to the Lightning `pl.Trainer` constructor."""
    additional_env_vars: dict[str, str] = {}
    """Additional environment variables to set when running the trainer."""
    set_nccl_optimal_params: bool = False
    """If enabled, will set the NCCL optimal parameters when running on multiple GPUs + nodes."""

    set_float32_matmul_precision: Literal["medium", "high", "highest"] | None = None
    """If enabled, will set the torch float32 matmul precision to the specified value. Useful for faster training on Ampere+ GPUs."""

    accelerator: Literal["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"] = "auto"
    """
    Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
        as well as custom accelerator instances.
    """
    strategy: str = "auto"
    """
    Supports different training strategies with aliases as well custom strategies.
        Default: ``"auto"``.
    """
    devices: list[int] | str | int = "auto"
    """
    The devices to use. Can be set to a positive number (int or str), a sequence of device indices
        (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
        automatic selection based on the chosen accelerator. Default: ``"auto"``.
    """
    num_nodes: Literal["auto"] | int = "auto"
    """
    Number of GPU nodes for distributed training,
        or ``"auto"`` to automatically detect the number of nodes.
    """
    precision: _PRECISION_INPUT = "32-true"
    """
    Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
        16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        Can be used on CPU, GPU, TPUs, HPUs or IPUs.
        Default: ``'32-true'``.
    """
    logger: bool | None = None
    """
    Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
        the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
        ``False`` will disable logging. If multiple loggers are provided, local files
        (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
        Default: ``True``.
    """
    fast_dev_run: int | bool = False
    """
    Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
        of train, val and test to find any bugs (ie: a sort of unit test).
        Default: ``False``.
    """
    max_epochs: int | None = None
    """
    Stop training once this number of epochs is reached. Disabled by default (None).
        If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
        To enable infinite training, set ``max_epochs = -1``.
    """
    min_epochs: int | None = None
    """Force training for at least these many epochs. Disabled by default (None)."""
    max_steps: int = -1
    """
    Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
        and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
        ``max_epochs`` to ``-1``.
    """
    min_steps: int | None = None
    """Force training for at least these number of steps. Disabled by default (``None``)."""
    max_time: str | timedelta | dict[str, Any] | None = None
    """
    Stop training after this amount of time has passed. Disabled by default (``None``).
        The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
        :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
        :class:`datetime.timedelta`.
    """
    limit_train_batches: int | float | None = None
    """
    How much of training dataset to check (float = fraction, int = num_batches).
        Default: ``1.0``.
    """
    limit_val_batches: int | float | None = None
    """
    How much of validation dataset to check (float = fraction, int = num_batches).
        Default: ``1.0``.
    """
    limit_test_batches: int | float | None = None
    """
    How much of test dataset to check (float = fraction, int = num_batches).
        Default: ``1.0``.
    """
    limit_predict_batches: int | float | None = None
    """
    How much of prediction dataset to check (float = fraction, int = num_batches).
        Default: ``1.0``.
    """
    overfit_batches: int | float = 0.0
    """
    Overfit a fraction of training/validation data (float) or a set number of batches (int).
        ``0.0`` means no overfitting. Default: ``0.0``.
    """
    val_check_interval: int | float | None = None
    """
    How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
        after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
        batches. An ``int`` value can only be higher than the number of training batches when
        ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
        across epochs or during iteration-based training.
        Default: ``1.0``.
    """
    check_val_every_n_epoch: int | None = 1
    """
    Perform a validation loop every after every `N` training epochs. If ``None``,
        validation will be done solely based on the number of training batches, requiring ``val_check_interval``
        to be an integer value.
        Default: ``1``.
    """
    num_sanity_val_steps: int | None = None
    """
    Sanity check runs n validation batches before starting the training routine.
        Set it to `-1` to run all batches in all validation dataloaders.
        Default: ``2``.
    """
    log_every_n_steps: int = 50
    """
    How often to log within steps.
        Default: ``50``.
    """
    enable_checkpointing: bool | None = None
    """
    If ``True``, enable checkpointing.
        It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`.
        Default: ``True``.
    """
    enable_progress_bar: bool | None = None
    """
    Whether to enable to progress bar by default.
        Default: ``True``.
    """
    enable_model_summary: bool | None = None
    """
    Whether to enable model summarization by default.
        Default: ``True``.
    """
    accumulate_grad_batches: int = 1
    """
    Accumulates gradients over k batches before stepping the optimizer.
        ``1`` means no gradient accumulation (i.e., performs a step after each batch).
        Default: ``1``.
    """
    deterministic: bool | str | None = None
    """
    If ``True``, sets whether PyTorch operations must use deterministic algorithms.
        Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
        that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.
    """
    benchmark: bool | None = None
    """
    The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
        The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
        (``False`` if not manually set). If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic`
        is set to ``True``, this will default to ``False``. Override to manually set a different value.
        Default: ``None``.
    """
    inference_mode: bool = True
    """
    Whether to use :func:`torch.inference_mode` (if `True`) or :func:`torch.no_grad` (if `False`) during
        evaluation (``validate``/``test``/``predict``).
    """
    use_distributed_sampler: bool = True
    """
    Whether to wrap the DataLoader's sampler with
        :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
        strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
        ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
        ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
        sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
        we don't do this automatically.
    """
    profiler: str | ProfilerConfig | None = None
    """
    To profile individual steps during training and assist in identifying bottlenecks.
        Default: ``None``.
    """
    detect_anomaly: bool = False
    """
    Enable anomaly detection for the autograd engine.
        Default: ``False``.
    """
    barebones: bool = False
    """
    Whether to run in "barebones mode", where all features that may impact raw speed are
        disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training
        runs. The following features are deactivated:
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_checkpointing`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.logger`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_progress_bar`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.log_every_n_steps`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_model_summary`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.num_sanity_val_steps`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.fast_dev_run`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.detect_anomaly`,
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.profiler`,
        :meth:`~lightning.pytorch.core.LightningModule.log`,
        :meth:`~lightning.pytorch.core.LightningModule.log_dict`.
    """
    plugins: list[PluginConfigProtocol] | None = None
    """
    Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
        Default: ``None``.
    """
    sync_batchnorm: bool = False
    """
    Synchronize batch norm layers between process groups/whole world.
        Default: ``False``.
    """
    reload_dataloaders_every_n_epochs: int = 0
    """
    Set to a positive integer to reload dataloaders every n epochs.
        Default: ``0``.
    """
    default_root_dir: str | Path | None = None
    """
    Default path for logs and weights when no logger/ckpt_callback passed.
        Default: ``os.getcwd()``.
        Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
    """

    # region Deprecated fields
    @property
    @deprecated("Please use trainer.optimizer.gradient_clipping instead.")
    def automatic_gradient_clip(self):
        if (config := self.optimizer.gradient_clipping) is None or not config.enabled:
            return False
        return True

    @automatic_gradient_clip.setter
    @deprecated("Please use trainer.optimizer.gradient_clipping instead.")
    def automatic_gradient_clip(self, value: bool):
        if self.optimizer.gradient_clipping is None:
            self.optimizer.gradient_clipping = GradientClippingConfig(
                enabled=False, value=1.0, algorithm="norm"
            )
        self.optimizer.gradient_clipping.enabled = value

    @property
    @deprecated("Please use trainer.optimizer.gradient_clipping instead.")
    def gradient_clip_algorithm(self):
        if (config := self.optimizer.gradient_clipping) is None or not config.enabled:
            return "norm"
        return config.algorithm

    @gradient_clip_algorithm.setter
    @deprecated("Please use trainer.optimizer.gradient_clipping instead.")
    def gradient_clip_algorithm(self, value: Literal["value", "norm"]):
        if self.optimizer.gradient_clipping is None:
            self.optimizer.gradient_clipping = GradientClippingConfig(
                enabled=False, value=1.0, algorithm=value
            )
        self.optimizer.gradient_clipping.algorithm = value

    @property
    @deprecated("Please use trainer.optimizer.gradient_clipping instead.")
    def gradient_clip_val(self):
        if (config := self.optimizer.gradient_clipping) is None or not config.enabled:
            return None
        return config.value

    @gradient_clip_val.setter
    @deprecated("Please use trainer.optimizer.gradient_clipping instead.")
    def gradient_clip_val(self, value: int | float | None):
        if value is None:
            self.optimizer.gradient_clipping = None
            return

        if self.optimizer.gradient_clipping is None:
            self.optimizer.gradient_clipping = GradientClippingConfig(
                enabled=False, value=value, algorithm="norm"
            )
        self.optimizer.gradient_clipping.enabled = True
        self.optimizer.gradient_clipping.value = value

    # endregion


class RunnerOutputSaveConfig(TypedConfig):
    enabled: bool = True
    """Enable saving the runner stdout and stderr to a file."""
    dirpath: str | Path | None = None
    """Directory path for the output file. If None, will use the current working directory/ll_runner_logs/{id}"""


class RunnerConfig(TypedConfig):
    auto_call_trainer_init_from_runner: bool = True
    """If enabled, will automatically call the Trainer.runner_init() function from the Runner. Should be `True` most of the time."""
    save_output: RunnerOutputSaveConfig | None = None
    """Output saving configuration options, or ``None`` to disable output saving."""


class BaseConfig(TypedConfig):
    id: str = Field(default_factory=lambda: BaseConfig.generate_id())
    """ID of the run."""
    name: str | None = None
    """Run name."""
    project: str | None = None
    """Project name."""
    tags: list[str] = []
    """Tags for the run."""
    notes: list[str] = []
    """Human readable notes for the run."""

    debug: bool = False
    """Whether to run in debug mode. This will enable debug logging and enable debug code paths."""
    environment: EnvironmentConfig = EnvironmentConfig()
    """A snapshot of the current environment information (e.g. python version, slurm info, etc.). This is automatically populated by the run script."""
    trainer: TrainerConfig = TrainerConfig()
    """PyTorch Lightning trainer configuration options. Check Lightning's `Trainer` documentation for more information."""
    runner: RunnerConfig = RunnerConfig()
    """`jmp.lightning.Runner` configuration options."""

    """Additional metadata for this run. This can be used to store arbitrary data that is not part of the config schema."""
    meta: dict[str, Any] = {}

    def clone(self, with_new_id: bool = True) -> Self:
        c = copy.deepcopy(self)
        if with_new_id:
            c.id = BaseConfig.generate_id()
        return c

    # region Seeding

    _rng: ClassVar[np.random.Generator | None] = None

    @staticmethod
    def generate_id(
        *,
        length: int = 8,
        ignore_rng: bool = False,
    ) -> str:
        rng = BaseConfig._rng if not ignore_rng else np.random.default_rng()
        if rng is None:
            warnings.warn(
                "BaseConfig._rng is None. The generated IDs will not be reproducible. "
                + "To fix this, call BaseConfig.set_seed(...) before generating any IDs.",
                category=IdSeedWarning,
            )
            rng = np.random.default_rng()

        alphabet = list(string.ascii_lowercase + string.digits)

        id = "".join(rng.choice(alphabet) for _ in range(length))
        return id

    @staticmethod
    def set_seed(seed: int | None = None) -> None:
        if seed is None:
            seed = int(time.time() * 1000)
        logger.critical(f"Seeding BaseConfig with seed {seed}")
        BaseConfig._rng = np.random.default_rng(seed)

    # endregion
