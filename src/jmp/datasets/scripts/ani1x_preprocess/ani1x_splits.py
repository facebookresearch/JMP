"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import copy
import pickle
import random

import h5py


def remove_small_systems(h5_file, key_list):
    sm_sys_list = []
    sm_sys_frames = []
    for key in key_list:
        if h5_file[key]["atomic_numbers"].shape[0] < 4:
            sm_sys_list.append(key)
            sm_sys_frames.append(h5_file[key]["coordinates"].shape[0])

    print(f"total frames removed: {sum(sm_sys_frames)}")
    updated_keys = [i for i in key_list if i not in sm_sys_list]
    return updated_keys


def get_split(h5_file, rand_key_list, tot_split_size):
    split_keys = []
    counter = 0
    for key in rand_key_list:
        split_keys.append(key)
        counter += h5_file[key]["coordinates"].shape[0]
        if counter >= tot_split_size:
            break
    print(f"total frames in split: {counter}")
    return split_keys


def main(args):
    ani_h5 = h5py.File(args.input_file, "r")
    ani_keys = list(ani_h5.keys())  # get all keys which happen to be unique molecules
    updated_ani_keys = remove_small_systems(ani_h5, ani_keys)
    rand_ani_keys = copy.deepcopy(updated_ani_keys)

    for i in range(6):
        random.shuffle(rand_ani_keys)

    val_keys = get_split(ani_h5, rand_ani_keys, args.split_size)
    tmp_keys = [i for i in rand_ani_keys if i not in val_keys]
    test_keys = get_split(ani_h5, tmp_keys, args.split_size)
    train_keys = [i for i in tmp_keys if i not in test_keys]

    with open(args.train_keys_output, "wb") as f:
        pickle.dump(train_keys, f)
    with open(args.test_keys_output, "wb") as f:
        pickle.dump(val_keys, f)
    with open(args.val_keys_output, "wb") as f:
        pickle.dump(test_keys, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split ANI-1x dataset into train, test, and validation sets."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input ANI-1x HDF5 file.",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=495600,
        help="Size of each split (default: 495600).",
    )
    parser.add_argument(
        "--train_keys_output",
        type=str,
        required=True,
        help="Path to save the train keys pickle file.",
    )
    parser.add_argument(
        "--test_keys_output",
        type=str,
        required=True,
        help="Path to save the test keys pickle file.",
    )
    parser.add_argument(
        "--val_keys_output",
        type=str,
        required=True,
        help="Path to save the validation keys pickle file.",
    )

    args = parser.parse_args()
    main(args)
