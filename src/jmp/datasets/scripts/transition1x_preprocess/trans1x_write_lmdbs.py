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

import ase.io
import lmdb
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg):
    db_path, samples, sampled_ids, idx, pid, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for sysid, fid in samples:
        fid = int(fid)
        sid = int(sysid.split("rxn")[1])
        traj_file = os.path.join(args.data_path, f"{sysid}.traj")
        # traj_logs = open(sample, "r").read().splitlines()
        # xyz_idx = os.path.splitext(os.path.basename(sample))[0]
        # traj_path = os.path.join(args.data_path, f"{xyz_idx}.extxyz")
        atoms = ase.io.read(traj_file, index=fid)
        data_object = Data(
            pos=torch.Tensor(atoms.get_positions()),
            atomic_numbers=torch.Tensor(atoms.get_atomic_numbers()),
            sid=sid,
            fid=fid,
            natoms=atoms.get_positions().shape[0],
            tags=torch.LongTensor(atoms.get_tags()),
            force=torch.Tensor(atoms.get_forces()),
            pbc=torch.Tensor(atoms.pbc),
            y=atoms.get_potential_energy(),
        )

        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        idx += 1
        sampled_ids.append(f"{sysid},{fid}" + "\n")
        pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def main(args):
    # xyz_logs = glob.glob(os.path.join(args.data_path, "*.txt"))
    # if not xyz_logs:
    #    raise RuntimeError("No *.txt files found. Did you uncompress?")

    # if args.num_workers > len(xyz_logs):
    #   args.num_workers = len(xyz_logs)
    ids_file = os.path.join(args.data_path, f"{args.split}_ids.pkl")
    with open(ids_file, "rb") as f:
        ids = pickle.load(f)

    # Initialize feature extractor.
    """a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=not args.test_data,
        r_forces=not args.test_data,
        r_fixed=True,
        r_distances=False,
        r_edges=args.get_edges,
    )"""

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_ids = np.array_split(ids, args.num_workers)

    # Extract features
    sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            db_paths[i],
            chunked_ids[i],
            sampled_ids[i],
            idx[i],
            i,
            args,
        )
        for i in range(args.num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    sampled_ids, idx = list(op[0]), list(op[1])

    # Log sampled image, trajectory trace
    for j, i in enumerate(range(args.num_workers)):
        ids_log = open(os.path.join(args.out_path, "data_log.%04d.txt" % i), "w")
        ids_log.writelines(sampled_ids[j])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to dir containing *.traj files")
    parser.add_argument(
        "--out_path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--split",
        help="train, test, or val",
    )
    """parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )"""
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    """parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )"""
    # parser.add_argument(
    #     "--test-data",
    #     action="store_true",
    #     help="Is data being processed test data?",
    # )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
