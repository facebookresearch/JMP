"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path
from typing import assert_never

import numpy as np
from jmp.datasets.finetune.base import LmdbDataset as FinetuneLmdbDataset
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig, PretrainLmdbDataset
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data.data import BaseData
from tqdm import tqdm


def _gather_metadata(dataset: Dataset[BaseData], num_workers: int, batch_size: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda data_list: np.array(
            [data.pos.shape[0] for data in data_list]
        ),
        shuffle=False,
    )

    natoms_list: list[np.ndarray] = []
    for natoms in tqdm(loader, total=len(loader)):
        natoms_list.append(natoms)

    natoms = np.concatenate(natoms_list)
    return natoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        help="Path to the LMDB file or directory containing LMDB files.",
        required=True,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Where to save the metadata npz file.",
        required=False,
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["pretrain", "finetune"],
        help="Type of dataset to gather metadata from.",
        required=True,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers to use for data loading.",
        default=32,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to use for data loading.",
        default=256,
    )
    args = parser.parse_args()

    # Parse and validate arguments
    src: Path = args.src
    dest: Path | None = args.dest
    dataset_type: str = args.type
    num_workers: int = args.num_workers
    batch_size: int = args.batch_size

    if dest is None:
        dest = src / "metadata.npz"

    assert src.exists(), f"{src} does not exist"
    assert src.is_file() or src.is_dir(), f"{src} is not a file or directory"

    assert dest.suffix == ".npz", f"{dest} is not a .npz file"
    assert not dest.exists(), f"{dest} already exists"

    assert dataset_type in ("pretrain", "finetune"), f"{dataset_type} is not valid"

    # Load dataset
    match dataset_type:
        case "pretrain":
            dataset = PretrainLmdbDataset(PretrainDatasetConfig(src=src))
        case "finetune":
            dataset = FinetuneLmdbDataset(src=src)
        case _:
            assert_never(dataset_type)

    # Gather metadata
    natoms = _gather_metadata(dataset, num_workers, batch_size)
    assert natoms.shape[0] == len(dataset), f"{natoms.shape[0]=} != {len(dataset)=}"

    # Save metadata
    np.savez(dest, natoms=natoms)


if __name__ == "__main__":
    main()
