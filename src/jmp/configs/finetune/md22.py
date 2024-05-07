"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import MD22Config
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig

STATS: dict[str, dict[str, NC]] = {
    "Ac-Ala3-NHMe": {
        "y": NC(mean=-26913.953, std=0.35547638),
        "force": NC(mean=0.0, std=1.1291506),
    },
    "DHA": {
        "y": NC(mean=-27383.035, std=0.41342595),
        "force": NC(mean=0.0, std=1.1258113),
    },
    "stachyose": {
        "y": NC(mean=-68463.59, std=0.5940788),
        "force": NC(mean=0.0, std=1.1104717),
    },
    "AT-AT": {
        "y": NC(mean=-50080.08, std=0.47309175),
        "force": NC(mean=0.0, std=1.2109985),
    },
    "AT-AT-CG-CG": {
        "y": NC(mean=-101034.23, std=0.680055),
        "force": NC(mean=0.0, std=1.2021886),
    },
    "buckyball-catcher": {
        "y": NC(mean=-124776.7, std=0.64662045),
        "force": NC(mean=0.0, std=1.0899031),
    },
    "double-walled_nanotube": {
        "y": NC(mean=-338224.16, std=3.3810701),
        "force": NC(mean=0.0, std=1.0137014),
    },
}


def jmp_l_md22_config_(
    config: MD22Config,
    molecule: DC.MD22Molecule,
    base_path: Path,
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
    config.train_dataset = DC.md22_config(molecule, base_path, "train")
    config.val_dataset = DC.md22_config(molecule, base_path, "val")
    config.test_dataset = DC.md22_config(molecule, base_path, "test")

    # MD22 specific settings
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
