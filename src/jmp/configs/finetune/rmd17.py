"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import RMD17Config
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import (
    EarlyStoppingConfig,
    PrimaryMetricConfig,
    RLPConfig,
    WarmupCosRLPConfig,
)

STATS: dict[str, dict[str, NC]] = {
    "aspirin": {
        "y": NC(mean=-17617.379355234374, std=0.2673998440577667),
        "force": NC(mean=0.0, std=1.2733363),
    },
    "azobenzene": {
        "y": NC(mean=-15553.118351233397, std=0.2866098335926971),
        "force": NC(mean=0.0, std=1.2940075),
    },
    "benzene": {
        "y": NC(mean=-6306.374855859375, std=0.10482645661015047),
        "force": NC(mean=0.0, std=0.90774584),
    },
    "ethanol": {
        "y": NC(mean=-4209.534573266602, std=0.18616576961275716),
        "force": NC(mean=0.0, std=1.1929188),
    },
    "malonaldehyde": {
        "y": NC(mean=-7254.903633896484, std=0.1812291921138577),
        "force": NC(mean=0.0, std=1.302443),
    },
    "naphthalene": {
        "y": NC(mean=-10478.192319667969, std=0.24922674853668708),
        "force": NC(mean=0.0, std=1.3102233),
    },
    "paracetamol": {
        "y": NC(mean=-13998.780924130859, std=0.26963984094801224),
        "force": NC(mean=0.0, std=1.2707518),
    },
    "salicylic": {
        "y": NC(mean=-13472.110348867187, std=0.2437920552529055),
        "force": NC(mean=0.0, std=1.3030343),
    },
    "toluene": {
        "y": NC(mean=-7373.347077485351, std=0.22534282741069667),
        "force": NC(mean=0.0, std=1.246547),
    },
    "uracil": {
        "y": NC(mean=-11266.351949697266, std=0.2227113171300836),
        "force": NC(mean=0.0, std=1.3692871),
    },
}


def jmp_l_rmd17_config_(
    config: RMD17Config, molecule: DC.RMD17Molecule, base_path: Path
):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set data config
    config.batch_size = 4

    # Set up dataset
    config.train_dataset = DC.rmd17_config(molecule, base_path, "train")
    config.val_dataset = DC.rmd17_config(molecule, base_path, "val")
    config.test_dataset = DC.rmd17_config(molecule, base_path, "test")

    # RMD17 specific settings
    config.molecule = molecule
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.model_type = "forces"
    config.gradient_forces = True
    config.trainer.inference_mode = False

    # Set up normalization
    if (normalization_config := STATS.get(molecule)) is None:
        raise ValueError(f"Normalization for {molecule} not found")
    config.normalization = normalization_config

    # We use more conservative early stopping for rMD17
    #   (we essentially copy Allegro here).
    config.trainer.max_epochs = 100_000
    config.trainer.max_time = "07:00:00:00"
    config.early_stopping = EarlyStoppingConfig(
        patience=1000,
        min_delta=1.0e-8,
        min_lr=1.0e-10,
    )

    # We also use a conservative set of hyperparameters
    #   for ReduceLROnPlateau (again, we copy Allegro here).
    # The main difference is that we use a larger patience (25 vs 3).
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=5,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.1,
        rlp=RLPConfig(patience=25, factor=0.8),
    )
