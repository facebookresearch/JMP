"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Literal, TypeAlias

from ...modules.dataset.common import DatasetAtomRefConfig
from .base import FinetuneLmdbDatasetConfig, FinetunePDBBindDatasetConfig
from .matbench import MatbenchDataset, MatbenchFold
from .md22 import MD22Molecule
from .rmd17 import RMD17Molecule
from .spice import SPICEDataset

Split: TypeAlias = Literal["train", "val", "test"]


def matbench_config(
    dataset: MatbenchDataset,
    base_path: Path,
    split: Split,
    fold: MatbenchFold,
):
    lmdb_path = base_path / "lmdb" / f"matbench_{dataset}" / f"{fold}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def rmd17_config(
    molecule: RMD17Molecule,
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{molecule}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def md22_config(
    molecule: MD22Molecule,
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{molecule}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def qm9_config(
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(
        src=lmdb_path,
        atom_ref=DatasetAtomRefConfig(
            refs={
                "ZPVE": [
                    0.0000e00,
                    3.1213e-01,
                    -1.0408e-17,
                    2.7756e-17,
                    -1.3878e-17,
                    0.0000e00,
                    1.3694e-01,
                    1.3955e-01,
                    1.1424e-01,
                    9.3038e-02,
                ],
                "U_0": [
                    0.0000e00,
                    -1.6430e01,
                    1.1369e-13,
                    -4.5475e-13,
                    0.0000e00,
                    0.0000e00,
                    -1.0360e03,
                    -1.4898e03,
                    -2.0470e03,
                    -2.7175e03,
                ],
                "U": [
                    0.0000e00,
                    -1.6419e01,
                    2.2737e-13,
                    -4.5475e-13,
                    0.0000e00,
                    0.0000e00,
                    -1.0360e03,
                    -1.4898e03,
                    -2.0470e03,
                    -2.7175e03,
                ],
                "H": [
                    0.0000e00,
                    -1.6419e01,
                    3.4106e-13,
                    -4.5475e-13,
                    0.0000e00,
                    0.0000e00,
                    -1.0360e03,
                    -1.4898e03,
                    -2.0470e03,
                    -2.7175e03,
                ],
                "G": [
                    0.0000e00,
                    -1.6443e01,
                    3.4106e-13,
                    -4.5475e-13,
                    0.0000e00,
                    0.0000e00,
                    -1.0361e03,
                    -1.4899e03,
                    -2.0471e03,
                    -2.7176e03,
                ],
                "c_v": [
                    0.0000e00,
                    1.2409e00,
                    -3.3307e-16,
                    4.4409e-16,
                    -1.1102e-16,
                    0.0000e00,
                    2.0350e00,
                    2.7877e00,
                    3.0860e00,
                    3.3401e00,
                ],
            }
        ),
    )
    return config


def qmof_config(
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def spice_config(
    dataset: SPICEDataset,
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{dataset}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def pdbbind_config(split: Split):
    config = FinetunePDBBindDatasetConfig(split=split)
    return config
