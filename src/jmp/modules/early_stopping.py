"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from logging import getLogger

from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping as LightningEarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message
from typing_extensions import override

log = getLogger(__name__)


class EarlyStoppingWithMinLR(LightningEarlyStopping):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0,
        min_lr: float | None = None,
        patience: int = 3,
        verbose: bool = True,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            log_rank_zero_only,
        )

        self.min_lr = min_lr

    @override
    @staticmethod
    def _log_info(
        trainer: Trainer | None, message: str, log_rank_zero_only: bool
    ) -> None:
        rank = _get_rank()
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.critical(message)

    @override
    def _run_early_stopping_check(self, trainer: Trainer):
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        # Disable early_stopping with fast_dev_run
        if getattr(trainer, "fast_dev_run", False):
            return

        should_stop, reason = False, None

        if not should_stop:
            should_stop, reason = self._evaluate_stopping_criteria_min_lr(trainer)

        # If metric present
        if not should_stop and self._validate_condition_metric(logs):
            current = logs[self.monitor].squeeze()
            should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria_min_lr(
        self, trainer: Trainer
    ) -> tuple[bool, str | None]:
        if self.min_lr is None:
            return False, None

        # Get the maximum LR across all param groups in all optimizers
        model_max_lr = max(
            [
                param_group["lr"]
                for optimizer in trainer.optimizers
                for param_group in optimizer.param_groups
            ]
        )
        if not isinstance(model_max_lr, float) or not math.isfinite(model_max_lr):
            return False, None

        # If the maximum LR is less than the minimum LR, stop training
        if model_max_lr >= self.min_lr:
            return False, None

        return True, (
            "Stopping threshold reached: "
            f"The maximum LR of the model across all param groups is {model_max_lr:.2e} "
            f"which is less than the minimum LR {self.min_lr:.2e}"
        )

    def on_early_stopping(self, trainer: Trainer):
        pass
