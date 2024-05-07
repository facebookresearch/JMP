"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import logging
from collections import abc
from pathlib import Path
from types import NoneType
from typing import Any, Callable

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from lightning_fabric.utilities.types import _PATH
from typing_extensions import override

from ..model.config import (
    BaseConfig,
    BaseProfilerConfig,
    PythonLogging,
    RunnerOutputSaveConfig,
)
from ..util import seed
from ..util.environment import set_additional_env_vars
from ..util.typing_utils import copy_method_with_param
from .logging import (
    default_root_dir,
    finalize_loggers,
    loggers_from_config,
    validate_logger,
)

log = logging.getLogger(__name__)


def _save_output_dir(root_config: BaseConfig, config: RunnerOutputSaveConfig):
    if not (dirpath := config.dirpath):
        dirpath = default_root_dir(root_config, logs_dirname="ll_runner_logs")
    dirpath = Path(dirpath).resolve()

    # Make sure that the directory exists
    dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


def _default_log_handlers(root_config: BaseConfig):
    if (config := root_config.runner.save_output) is None or not config.enabled:
        return

    # Get the directory path
    dirpath = _save_output_dir(root_config, config)

    # Capture the logs to `dirpath`/log.log
    log_file = dirpath / "log.log"
    log_file.touch(exist_ok=True)
    yield logging.FileHandler(log_file)


def _setup_logger(root_config: BaseConfig, config: PythonLogging):
    if config.lovely_tensors:
        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            log.warning(
                "Failed to import lovely-tensors. Ignoring pretty PyTorch tensor formatting"
            )

    if config.lovely_numpy:
        try:
            import lovely_numpy

            lovely_numpy.set_config(repr=lovely_numpy.lovely)
        except ImportError:
            log.warning(
                "Failed to import lovely-numpy. Ignoring pretty numpy array formatting"
            )

    log_handlers: list[logging.Handler] = [*_default_log_handlers(root_config)]
    if config.rich:
        try:
            from rich.logging import RichHandler

            log_handlers.append(RichHandler())
        except ImportError:
            log.warning(
                "Failed to import rich. Falling back to default Python logging."
            )

    logging.basicConfig(
        level=config.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=log_handlers,
    )

    logging.basicConfig(level=config.log_level)


class Trainer(LightningTrainer):
    _finalizers: list[Callable[[], None]] = []

    def finalize(self):
        """
        Call this method to clean up after training.
        """
        finalize_loggers(self.loggers)

    @staticmethod
    def ll_default_root_dir(
        config: BaseConfig, *, logs_dirname: str = "lightning_logs"
    ):
        return default_root_dir(config, logs_dirname=logs_dirname)

    @classmethod
    def setup_python_logging(cls, config: BaseConfig):
        _setup_logger(config, config.trainer.python_logging)

    @classmethod
    @contextlib.contextmanager
    def output_save_context(cls, root_config: BaseConfig):
        if (config := root_config.runner.save_output) is None or not config.enabled:
            yield
            return

        # Get the directory path
        dirpath = _save_output_dir(root_config, config)

        # Capture the stdout and stderr logs to `dirpath`/stdout.log and `dirpath`/stderr.log
        stdout_log = dirpath / "stdout.log"
        stderr_log = dirpath / "stderr.log"
        stdout_log.touch(exist_ok=True)
        stderr_log.touch(exist_ok=True)
        with stdout_log.open("a") as file:
            with contextlib.redirect_stdout(file):
                with stderr_log.open("a") as file:
                    with contextlib.redirect_stderr(file):
                        yield

    @classmethod
    @contextlib.contextmanager
    def ll_initialize(cls, config: BaseConfig):
        with contextlib.ExitStack() as stack:
            if not config.runner.auto_call_trainer_init_from_runner:
                stack.enter_context(cls.runner_init(config))

            if config.trainer.auto_set_default_root_dir:
                if config.trainer.default_root_dir:
                    raise ValueError(
                        "You have set both `config.trainer.default_root_dir` and `config.trainer.auto_set_default_root_dir`. "
                        "Please set only one of them."
                    )
                config.trainer.default_root_dir = str(
                    cls.ll_default_root_dir(config).absolute()
                )
                log.critical(f"Setting {config.trainer.default_root_dir=}.")

            yield

    @classmethod
    @contextlib.contextmanager
    def runner_init(cls, config: BaseConfig):
        with contextlib.ExitStack() as stack:
            cls.setup_python_logging(config)
            # Save stdout/stderr to a file
            stack.enter_context(Trainer.output_save_context(config))
            yield

    @classmethod
    def ll_default_callbacks(cls, config: BaseConfig):
        if config.trainer.on_exception_checkpoint:
            if config.trainer.default_root_dir is None:
                raise ValueError(
                    "You must specify `config.trainer.default_root_dir` "
                    "to use `config.trainer.on_exception_checkpoint`."
                )
            log_dir = Path(config.trainer.default_root_dir)
            yield OnExceptionCheckpoint(log_dir, filename=f"on_exception_{config.id}")

    @classmethod
    @contextlib.contextmanager
    def context(cls, config: BaseConfig):
        with contextlib.ExitStack() as stack:
            stack.enter_context(cls.ll_initialize(config))

            cls._finalizers.clear()
            if config.trainer.seed is not None:
                stack.enter_context(
                    seed.seed_context(
                        config.trainer.seed, workers=config.trainer.seed_workers
                    )
                )

            additional_nccl_env_vars: dict[str, str] = {}
            if config.trainer.set_nccl_optimal_params:
                # We need to set these env vars before the NCCL library is loaded.
                # Reportedly, the training performance can be improved quite a bit (see).
                # Details on all available env vars: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
                additional_nccl_env_vars["NCCL_NSOCKS_PERTHREAD"] = "4"
                additional_nccl_env_vars["NCCL_SOCKET_NTHREADS"] = "2"

            if (precision := config.trainer.set_float32_matmul_precision) is not None:
                torch.set_float32_matmul_precision(precision)

            stack.enter_context(
                set_additional_env_vars(
                    config.trainer.additional_env_vars | additional_nccl_env_vars
                )
            )

            try:
                yield
            finally:
                n_finalizers = 0
                for finalizer in reversed(cls._finalizers):
                    finalizer()
                    n_finalizers += 1

                cls._finalizers.clear()
                log.critical(
                    f"Ran {n_finalizers} finalizers for {cls.__name__} cleanup."
                )

    @classmethod
    def _update_kwargs(cls, config: BaseConfig, kwargs_ctor: dict[str, Any]):
        kwargs = {
            "accelerator": config.trainer.accelerator,
            "strategy": config.trainer.strategy,
            "devices": config.trainer.devices,
            "num_nodes": config.trainer.num_nodes,
            "precision": config.trainer.precision,
            "logger": config.trainer.logger,
            "fast_dev_run": config.trainer.fast_dev_run,
            "max_epochs": config.trainer.max_epochs,
            "min_epochs": config.trainer.min_epochs,
            "max_steps": config.trainer.max_steps,
            "min_steps": config.trainer.min_steps,
            "max_time": config.trainer.max_time,
            "limit_train_batches": config.trainer.limit_train_batches,
            "limit_val_batches": config.trainer.limit_val_batches,
            "limit_test_batches": config.trainer.limit_test_batches,
            "limit_predict_batches": config.trainer.limit_predict_batches,
            "overfit_batches": config.trainer.overfit_batches,
            "val_check_interval": config.trainer.val_check_interval,
            "check_val_every_n_epoch": config.trainer.check_val_every_n_epoch,
            "num_sanity_val_steps": config.trainer.num_sanity_val_steps,
            "log_every_n_steps": config.trainer.log_every_n_steps,
            "enable_checkpointing": config.trainer.enable_checkpointing,
            "enable_progress_bar": config.trainer.enable_progress_bar,
            "enable_model_summary": config.trainer.enable_model_summary,
            "accumulate_grad_batches": config.trainer.accumulate_grad_batches,
            "deterministic": config.trainer.deterministic,
            "benchmark": config.trainer.benchmark,
            "inference_mode": config.trainer.inference_mode,
            "use_distributed_sampler": config.trainer.use_distributed_sampler,
            "detect_anomaly": config.trainer.detect_anomaly,
            "barebones": config.trainer.barebones,
            "plugins": config.trainer.plugins,
            "sync_batchnorm": config.trainer.sync_batchnorm,
            "reload_dataloaders_every_n_epochs": config.trainer.reload_dataloaders_every_n_epochs,
        }
        # if config.trainer.automatic_gradient_clip:
        #     kwargs["gradient_clip_val"] = config.trainer.gradient_clip_val
        #     kwargs["gradient_clip_algorithm"] = config.trainer.gradient_clip_algorithm
        if (
            grad_clip_config := config.trainer.optimizer.gradient_clipping
        ) is not None and grad_clip_config.enabled:
            kwargs["gradient_clip_algorithm"] = grad_clip_config.algorithm
            kwargs["gradient_clip_val"] = grad_clip_config.value

        if profiler := config.trainer.profiler:
            # If the profiler is an ProfilerConfig instance, then we instantiate it.
            if isinstance(profiler, BaseProfilerConfig):
                profiler = profiler.construct_profiler()
                # Make sure that the profiler is an instance of `Profiler`.
                if not isinstance(profiler, Profiler):
                    raise ValueError(f"{profiler=} is not an instance of `{Profiler}`.")

            # Otherwise, if the profiler is a string (e.g., "simpe", "advanced", "pytorch"),
            #   then we just pass it through.
            kwargs["profiler"] = profiler

        kwargs.update(kwargs_ctor)

        kwargs["plugins"] = []
        if config.trainer.plugins is not None:
            kwargs["plugins"].extend(config.trainer.plugins)
        if (plugins := kwargs_ctor.get("plugins")) is not None:
            plugins = [plugins] if not isinstance(plugins, list) else plugins
            kwargs["plugins"].extend(plugins)

        if config.trainer.logger is False:
            log.critical(f"Disabling logger because {config.trainer.logger=}.")
            kwargs["logger"] = False
        elif kwargs.get("logger") is False:
            log.critical(f"Disabling logger because {kwargs.get('logger')=}.")

        if (
            existing_loggers := kwargs.get("logger")
        ) is not False and config.trainer.auto_set_loggers:
            if int(config.trainer.fast_dev_run) > 0:
                log.critical("Disabling loggers because fast_dev_run is enabled.")
            else:
                loggers = loggers_from_config(config)
                if existing_loggers is not None and not isinstance(
                    existing_loggers, bool
                ):
                    if not isinstance(existing_loggers, list):
                        existing_loggers = [existing_loggers]
                    loggers.extend(existing_loggers)

                kwargs["logger"] = loggers

        if kwargs.get("num_nodes") == "auto":
            # when num_nodes is auto, we need to detect the number of nodes
            # when on slurm, this would be the number of SLURM nodes allocated
            if SLURMEnvironment.detect():
                from submitit import JobEnvironment

                job = JobEnvironment()
                if not job.activated():
                    raise ValueError(
                        "SLURMEnvironment detected through PL but not submitit. This is a bug."
                    )

                kwargs["num_nodes"] = job.num_nodes
                log.critical(
                    f"Setting num_nodes to {job.num_nodes} (detected through submitit)."
                )
            # otherweise, we assume 1 node
            else:
                kwargs["num_nodes"] = 1
                log.critical("Setting num_nodes to 1 (no SLURM detected).")

        if config.trainer.default_root_dir:
            kwargs["default_root_dir"] = str(config.trainer.default_root_dir)
        kwargs.update(config.trainer.additional_trainer_kwargs)

        # Set the callbacks
        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.extend(cls.ll_default_callbacks(config))
        kwargs["callbacks"] = callbacks

        return kwargs

    @override
    @copy_method_with_param(
        LightningTrainer.__init__,
        param_type=BaseConfig,
        return_type=NoneType,
    )
    def __init__(self, config: BaseConfig, *args, **kwargs):
        self._ll_config = config
        kwargs = self._update_kwargs(config, kwargs)
        log.critical(f"LightningTrainer.__init__ with {args=} and {kwargs=}.")
        super().__init__(*args, **kwargs)

        if config.trainer.enable_logger_validation:
            for logger in self.loggers:
                validate_logger(logger, config.id)

        if config.trainer.checkpoint_last_by_default:
            self._patch_checkpoint_last_by_default()
        if config.trainer.auto_add_trainer_finalizer:
            type(self)._finalizers.append(self.finalize)

        # Print out the log dir, so that we can easily find it in the logs.
        if log_dir := self.log_dir:
            log_dir = str(Path(log_dir).resolve())
        log.critical(f"LightningTrainer log directory: {self.log_dir}.")

    def _patch_checkpoint_last_by_default(self):
        """
        Patch the default ModelCheckpoint callback to save the last checkpoint by default.
        """
        enable_checkpointing = (
            True
            if self._ll_config.trainer.enable_checkpointing is None
            else self._ll_config.trainer.enable_checkpointing
        )
        if not enable_checkpointing:
            return

        if not (callbacks := getattr(self, "callbacks", None)) or not isinstance(
            callbacks, abc.Iterable
        ):
            return

        if (
            model_ckpt := next(
                (c for c in callbacks if isinstance(c, ModelCheckpoint)), None
            )
        ) is None:
            return

        log.critical(f"Setting {model_ckpt.__class__.__name__}.save_last=True.")
        model_ckpt.save_last = True
        # hacky: call the `__validate_init_configuration` method to ensure that the `save_last` parameter is valid.
        # model_ckpt.__validate_init_configuration() <- this doesn't work because it's a private method
        if (
            validate_init_configuration := getattr(
                model_ckpt,
                f"_{model_ckpt.__class__.__name__}__validate_init_configuration",
                None,
            )
        ) is not None and callable(validate_init_configuration):
            validate_init_configuration()
        else:
            log.warning(
                f"Failed to find {model_ckpt.__class__.__name__}.__validate_init_configuration. "
                "This means that we cannot validate the `save_last` parameter for ModelCheckpoint."
            )

    @override
    def _run(
        self, model: LightningModule, ckpt_path: _PATH | None = None
    ) -> _EVALUATE_OUTPUT | _PREDICT_OUTPUT | None:
        """
        Lightning doesn't support gradient clipping with manual optimization.
        We patch the `Trainer._run` method to throw if gradient clipping is enabled
        and `model.automatic_optimization` is False.
        """
        if not model.automatic_optimization and (
            self.gradient_clip_val is not None
            or self.gradient_clip_algorithm is not None
        ):
            raise ValueError(
                "Gradient clipping is not supported with manual optimization. "
                f"Please set {model.__class__.__name__}.automatic_optimization to True "
                "or disable automatic gradient clipping. "
                "If you want to use gradient clipping with manual optimization, you can "
                "set `config.trainer.automatic_gradient_clip=False` and "
                "use the values in `config.trainer.gradient_clip_val` and `config.trainer.gradient_clip_algorithm`."
            )

        return super()._run(model, ckpt_path)
