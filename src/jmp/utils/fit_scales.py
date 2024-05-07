"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Literal

import torch
from jmp.lightning import BaseConfig as Config
from jmp.lightning import LightningModuleBase, Trainer
from jmp.models.gemnet.backbone import GemNetOCBackbone
from jmp.modules.scaling import ScaleFactor
from typing_extensions import TypeVar

log = getLogger(__name__)

TModel = TypeVar("TModel", bound=LightningModuleBase, infer_variance=True)


def fit_scales(
    config: Config,
    model_cls: Callable[[Config], TModel],
    out_path: Path,
    *,
    backbone: Callable[[TModel], GemNetOCBackbone],
    num_batches: int = 16,
    fitted_mode: Literal["replace", "ignore", "error"] = "replace",
    replace_out_path_if_exists: bool = False,
):
    config = copy.deepcopy(config)

    # config.trainer.fast_dev_run = num_batches
    config.trainer.precision = "32-true"
    config.trainer.logger = False
    config.trainer.max_steps = num_batches
    config.trainer.limit_val_batches = num_batches
    config.trainer.num_sanity_val_steps = 0

    with Trainer.context(config):
        model = model_cls(config)

        scale_factors = {
            name: module
            for name, module in backbone(model).named_modules()
            if isinstance(module, ScaleFactor)
        }

        # region detect fitted/unfitted factors
        fitted_scale_factors = [
            f"{name}: {module.scale_factor.item():.3f}"
            for name, module in scale_factors.items()
            if module.fitted
        ]
        unfitted_scale_factors = [
            name for name, module in scale_factors.items() if not module.fitted
        ]
        fitted_scale_factors_str = ", ".join(fitted_scale_factors)
        log.info(f"Fitted scale factors: [{fitted_scale_factors_str}]")
        unfitted_scale_factors_str = ", ".join(unfitted_scale_factors)
        log.info(f"Unfitted scale factors: [{unfitted_scale_factors_str}]")

        if fitted_scale_factors:
            match fitted_mode:
                case "replace":
                    log.info("Replacing fitted scale factors with new ones.")
                case "ignore":
                    log.info("Ignoring fitted scale factors.")
                case "error":
                    log.error("Found fitted scale factors.")
                    log.error("Exiting script.")
                    return
                case _:
                    raise ValueError(f"Unknown fitted_mode: {fitted_mode}")
        # endregion

        log.info(
            f"Output path for fitted scale factors: {out_path}, {out_path.exists()=}"
        )
        if out_path.exists() and not replace_out_path_if_exists:
            raise FileExistsError(f"Output path already exists: {out_path}")

        # region reset the scale factors if mode == "all"
        if fitted_mode == "replace":
            log.info("Fitting all scale factors and resetting the fitted ones.")
            for name, scale_factor in scale_factors.items():
                if scale_factor.fitted:
                    log.info(
                        f"{name} is already fitted in the checkpoint, resetting it. {scale_factor.scale_factor}"
                    )
                scale_factor.reset_()
        # endregion

        # loop over the scale factors in the computation order
        # and fit them one by one
        log.info("Start fitting")
        trainer = Trainer(config)

        for name, module in scale_factors.items():
            try:
                if module.fitted and fitted_mode == "ignore":
                    log.info(f"Skipping {name} (already fitted)")
                    continue

                log.info(f"Fitting {name}...")
                with module.fit_context_():
                    _ = trainer.validate(model, verbose=False)
                    stats, ratio, value = module.fit_()

                    log.info(
                        f"Variable: {name}, "
                        f"Var_in: {stats['variance_in']:.3f}, "
                        f"Var_out: {stats['variance_out']:.3f}, "
                        f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
                    )
            except BaseException as e:
                raise RuntimeError(f"Failed to fit {name}") from e

        # make sure all scale factors are fitted
        for name, module in scale_factors.items():
            if not module.fitted:
                raise RuntimeError(f"Scale factor {name} is not fitted.")

        # region save the scale factors to the checkpoint file
        scale_factors_out = {
            name: module.scale_factor.clone() for name, module in scale_factors.items()
        }
        torch.save(scale_factors_out, out_path)
        log.info(f"Saved results to: {out_path}")
        # endregion
