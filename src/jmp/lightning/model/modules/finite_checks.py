"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import cast

from typing_extensions import override

from ...callbacks.bad_gradients import PrintBadGradientsCallback, print_bad_gradients
from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin


class FiniteChecksModuleMixin(mixin_base_type(CallbackModuleMixin)):
    def print_bad_gradients(self):
        print_bad_gradients(self)

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def _cb():
            nonlocal self
            config = cast(BaseConfig, self.hparams)
            if (
                not config.trainer.optimizer.grad_finite_checks
                and not config.trainer.optimizer.grad_none_checks
            ):
                return None

            return PrintBadGradientsCallback(
                none_grads=config.trainer.optimizer.grad_none_checks,
                nonfinite_grads=config.trainer.optimizer.grad_finite_checks,
            )

        self.register_callback(_cb)
