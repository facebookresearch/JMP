"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# pylint: disable=stop-iteration-return

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from ase.units import Hartree


def generator(formula, grp):
    """Iterates through a h5 group"""

    energies = grp["wb97x_dz.energy"]
    forces = grp["wb97x_dz.forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["coordinates"]
    fid = 0

    for energy, force, positions in zip(energies, forces, positions):
        # skip if energy/force is nan
        if np.isnan(energy) or np.isnan(force).any():
            continue
        # get ase atoms object
        atoms = Atoms(atomic_numbers, positions=positions)
        # convert from hartree to eV and hartree/angstrom to eV/angstrom
        energy = energy * Hartree
        force = force * Hartree
        sp_calc = SPCalc(atoms=atoms, energy=energy, forces=force.tolist())
        sp_calc.implemented_properties = ["energy", "forces"]
        atoms.set_calculator(sp_calc)
        atoms.set_tags(2 * np.ones(len(atomic_numbers)))
        id = (formula, fid)
        fid += 1

        yield id, atoms


class Dataloader:
    """
    Can iterate through h5 data set for paper ####

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, split_keys):
        self.hdf5_file = hdf5_file
        self.split_keys = split_keys

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as h5_file:
            for key in self.split_keys:
                atoms_list = []
                id_list = []
                for id, molecule in generator(key, h5_file[key]):
                    atoms_list.append(molecule)
                    id_list.append(id)
                assert len(atoms_list) == h5_file[key]["coordinates"].shape[0]

                yield id_list, atoms_list
