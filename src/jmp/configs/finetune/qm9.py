"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import QM9Config
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig
from ...tasks.finetune.qm9 import QM9Target, SpatialExtentConfig

STATS: dict[str, NC] = {
    "mu": NC(mean=2.674587, std=1.5054824),
    "alpha": NC(mean=75.31013, std=8.164021),
    "eps_HMO": NC(mean=-6.5347567, std=0.59702325),
    "eps_LUMO": NC(mean=0.323833, std=1.273586),
    "delta_eps": NC(mean=6.8585854, std=1.283122),
    "R_2_Abs": NC(mean=1189.6819, std=280.0421),
    "ZPVE": NC(mean=-0.00052343315, std=0.04904531),
    "U_0": NC(mean=0.0028667436, std=1.0965848),
    "U": NC(mean=0.0028711546, std=1.0941933),
    "H": NC(mean=0.0029801112, std=1.0942822),
    "G": NC(mean=0.000976671, std=1.101572),
    "c_v": NC(mean=-0.005799451, std=2.2179737),
    "U_0_ATOM": NC(mean=-76.15232, std=10.309152),
    "U_ATOM": NC(mean=-76.6171, std=10.400515),
    "H_ATOM": NC(mean=-77.05511, std=10.474532),
    "G_ATOM": NC(mean=-70.87026, std=9.484609),
    "A": NC(mean=11.58375, std=2046.5049),
    "B": NC(mean=1.40327, std=1.1445134),
    "C": NC(mean=1.1256535, std=0.85679144),
}


def jmp_l_qm9_config_(config: QM9Config, target: QM9Target, base_path: Path):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set up dataset
    config.train_dataset = DC.qm9_config(base_path, "train")
    config.val_dataset = DC.qm9_config(base_path, "val")
    config.test_dataset = DC.qm9_config(base_path, "test")

    # Set up normalization
    if (normalization_config := STATS.get(target)) is None:
        raise ValueError(f"Normalization for {target} not found")
    config.normalization = {target: normalization_config}

    # QM9 specific settings
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

    # Make sure we only optimize for the target
    config.graph_scalar_targets = [target]
    config.node_vector_targets = []
    config.graph_classification_targets = []
    config.graph_scalar_reduction = {target: "sum"}

    # Handle R_2_Abs separately
    if target == "R_2_Abs":
        config.output_head = SpatialExtentConfig()
