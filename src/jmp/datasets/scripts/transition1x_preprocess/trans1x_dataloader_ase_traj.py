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

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp):
    """Iterates through a h5 group"""

    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)
    fid = 0

    for energy, force, positions in zip(energies, forces, positions):
        # get ase atoms object
        atoms = Atoms(atomic_numbers, positions=positions)
        sp_calc = SPCalc(atoms=atoms, energy=energy, forces=force.tolist())
        sp_calc.implemented_properties = ["energy", "forces"]
        atoms.set_calculator(sp_calc)
        atoms.set_tags(2 * np.ones(len(atomic_numbers)))
        id = (f"{formula}_{rxn}", fid)
        fid += 1

        """d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }"""

        yield id, atoms


class Dataloader:
    """
    Can iterate through h5 data set for paper ####

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, datasplit="data", only_final=False):
        self.hdf5_file = hdf5_file
        self.only_final = only_final

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    # reactant = next(generator(formula, rxn, subgrp["reactant"]))
                    # product = next(generator(formula, rxn, subgrp["product"]))

                    """if self.only_final:
                        transition_state = next(
                            generator(formula, rxn, subgrp["transition_state"])
                        )
                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                        }"""
                    # yield (reactant, "reactant")
                    # yield (product, "product")
                    rxn_atoms_list = []
                    id_list = []
                    sm_sys = None
                    for id, molecule in generator(formula, rxn, subgrp):
                        rxn_atoms_list.append(molecule)
                        id_list.append(id)
                    assert len(rxn_atoms_list) == subgrp["positions"].shape[0]
                    # marking systems that have less than 4 atoms
                    if subgrp["atomic_numbers"].shape[0] < 4:
                        sm_sys = (f"{formula}_{rxn}", subgrp["atomic_numbers"].shape[0])
                    yield id_list, rxn_atoms_list, sm_sys
