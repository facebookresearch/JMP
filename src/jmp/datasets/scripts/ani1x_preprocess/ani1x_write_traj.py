"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import random
import time

import ase.io

from .ani1x_dataloader_ase_traj import Dataloader


def write_traj(data_i):
    id_list, atoms_list = data_i
    traj_file = os.path.join(args.traj_dir, id_list[0][0] + ".traj")
    # write ase traj file
    ase.io.write(traj_file, atoms_list, format="traj")
    return id_list


def main(args):
    start = time.time()

    pool = mp.Pool(args.num_workers)
    split_keys = pickle.load(open(args.split_keys, "rb"))
    dataloader = Dataloader(args.ani1x_h5, split_keys)

    out_pool = list(pool.imap_unordered(write_traj, dataloader))
    # flatten list of lists
    sampled_ids = [item for sublist in out_pool for item in sublist]

    random.shuffle(sampled_ids)
    random.shuffle(sampled_ids)
    random.shuffle(sampled_ids)
    # write pkl files for ids
    id_file = os.path.join(args.traj_dir, args.split + "_ids.pkl")
    with open(id_file, "wb") as f:
        pickle.dump(sampled_ids, f)

    end = time.time()
    total_time = end - start
    print("Total time: ", total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ASE trajectory files from ANI-1x dataset."
    )
    parser.add_argument(
        "--ani1x_h5", type=str, required=True, help="Path to the ANI-1x HDF5 file."
    )
    parser.add_argument(
        "--split_keys",
        type=str,
        required=True,
        help="Path to the pickle file containing split keys.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Name of the split (e.g., train, test, val).",
    )
    parser.add_argument(
        "--traj_dir",
        type=str,
        required=True,
        help="Directory to save the ASE trajectory files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of worker processes (default: 64).",
    )

    args = parser.parse_args()

    # Load split keys from pickle file
    with open(args.split_keys, "rb") as f:
        split_keys = pickle.load(f)

    main(args)
