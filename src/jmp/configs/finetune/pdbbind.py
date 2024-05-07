"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from ...tasks.config import AdamWConfig
from ...tasks.finetune import PDBBindConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig


def jmp_l_pdbbind_config_(config: PDBBindConfig, target: str = "y"):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set up dataset
    config.train_dataset = DC.pdbbind_config("train")
    config.val_dataset = DC.pdbbind_config("val")
    config.test_dataset = DC.pdbbind_config("test")

    # PDBBind specific settings
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

    # Make sure we only optimize for the single target
    config.graph_scalar_targets = [target]
    config.node_vector_targets = []
    config.graph_classification_targets = []
    config.graph_scalar_reduction = {target: "sum"}

    # PDBBind specific settings
    config.pbdbind_task = "-logKd/Ki"
    config.metrics.report_rmse = True
