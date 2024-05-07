"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Literal

from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar

VALID_UNITS = ("eV", "kcal/mol", "hartree", "bohr", "angstrom")
Unit = Literal["eV", "kcal/mol", "hartree", "bohr", "angstrom"]


def _determine_factor(from_: Unit, to: Unit, *, reciprocal: bool = False):
    if from_ == to:
        return 1.0

    match (from_, to):
        case ("eV", "kcal/mol"):
            factor = 23.061
        case ("eV", "hartree"):
            factor = 0.0367493
        case ("kcal/mol", "eV"):
            factor = 1 / 23.061
        case ("kcal/mol", "hartree"):
            factor = 1 / 627.509
        case ("hartree", "eV"):
            factor = 1 / 0.0367493
        case ("hartree", "kcal/mol"):
            factor = 627.509
        case ("bohr", "angstrom"):
            factor = 0.529177
        case ("angstrom", "bohr"):
            factor = 1 / 0.529177
        case _:
            raise ValueError(f"Cannot convert {from_} to {to}")

    return 1 / factor if reciprocal else factor


T = TypeVar("T", bound=BaseData, infer_variance=True)


def update_units_transform(
    data: T,
    attributes: list[str] = ["y", "force"],
    *,
    from_: Unit,
    to: Unit,
    reciprocal: bool = False,
) -> T:
    factor = _determine_factor(from_, to, reciprocal=reciprocal)

    for attr in attributes:
        if (value := getattr(data, attr, None)) is None:
            continue
        setattr(data, attr, value * factor)

    return data


def update_pyg_data_units(
    data: BaseData,
    attributes: list[str],
    *,
    from_: Unit,
    to: Unit,
):
    factor = _determine_factor(from_, to)
    for attr in attributes:
        if (value := getattr(data, attr, None)) is None:
            continue
        setattr(data, attr, value * factor)

    return data
