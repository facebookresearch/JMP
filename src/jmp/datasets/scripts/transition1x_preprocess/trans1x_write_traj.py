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
from functools import partial

import ase.io

from .trans1x_dataloader_ase_traj import Dataloader


def write_traj(data_i, *, args: argparse.Namespace):
    id_list, atoms_list, sm_sys = data_i
    traj_file = os.path.join(args.traj_dir, id_list[0][0] + ".traj")
    # write ase traj file
    ase.io.write(traj_file, atoms_list, format="traj")
    return id_list, sm_sys


def main(args: argparse.Namespace):
    start = time.time()

    pool = mp.Pool(args.num_workers)
    dataloader = Dataloader(args.transition1x_h5, datasplit=args.split)

    out_pool = list(zip(*pool.imap(partial(write_traj, args=args), dataloader)))
    sampled_ids, sm_sys = list(out_pool[0]), list(out_pool[1])
    # flatten list of lists
    sampled_ids = [item for sublist in sampled_ids for item in sublist]

    random.shuffle(sampled_ids)
    random.shuffle(sampled_ids)
    random.shuffle(sampled_ids)
    # write plk files for ids and small systems
    id_file = os.path.join(args.traj_dir, args.split + "_ids.pkl")
    with open(id_file, "wb") as f:
        pickle.dump(sampled_ids, f)
    sm_sys_file = os.path.join(args.traj_dir, args.split + "_sm_sys.pkl")
    with open(sm_sys_file, "wb") as f:
        pickle.dump(sm_sys, f)

    end = time.time()
    total_time = end - start
    print("Total time: ", total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition1x_h5",
        type=str,
        help="Path to the HDF5 file containing the dataset.",
    )
    parser.add_argument(
        "--traj_dir",
        type=str,
        help="Directory to save the trajectory files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes.",
    )
    args = parser.parse_args()
    main(args)
