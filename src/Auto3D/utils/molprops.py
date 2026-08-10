#!/usr/bin/env python
"""Scalar properties read straight off a molecular graph.

Formal charge and the conformer budget: both are functions of the graph alone
(no coordinates, no force field, no energy), which is what separates them from
``utils/geometry.py`` and ``utils/connectivity.py``.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops

from Auto3D.constants import (
    CONFORMER_MULTIPLIER,
    CONFORMER_ROTATABLE_COEFF,
    CONFORMER_ROTATABLE_EXP,
    MAX_CONFORMERS_CAP,
)

__all__ = ["calculate_conformer_count", "get_mol_charge"]


def calculate_conformer_count(mol: Chem.Mol) -> int:
    """Calculate the number of conformers to generate for a molecule.

    Uses a formula based on the number of rotatable bonds, with a minimum
    of the heavy atom count and a maximum cap. The result is floored at 1 so
    a molecule never gets 0 conformers (which would silently drop tiny species
    such as ``[H+]`` or a lone atom from the pipeline).

    Formula: min(max(1, num_heavy, 2 * 8.481 * (num_rotatable ** 1.642)), 1000)
    Reference: https://doi.org/10.1021/acs.jctc.0c01213

    Args:
        mol: RDKit molecule object (with or without hydrogens).

    Returns:
        Number of conformers to generate (always >= 1).

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCCCCC")  # hexane
        >>> count = calculate_conformer_count(mol)
        >>> 1 <= count <= 1000
        True
    """
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    formula_count = int(
        CONFORMER_MULTIPLIER * CONFORMER_ROTATABLE_COEFF * (num_rotatable**CONFORMER_ROTATABLE_EXP)
    )

    # Floor at 1: a heavy-atom-free species (e.g. [H+]) or a single atom must
    # still receive at least one conformer instead of being silently dropped.
    return min(max(1, num_heavy, formula_count), MAX_CONFORMERS_CAP)


def get_mol_charge(mol: Chem.Mol) -> int:
    """Get the formal charge of a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        The total formal charge of the molecule.

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("[NH4+]")
        >>> get_mol_charge(mol)
        1
    """
    return rdmolops.GetFormalCharge(mol)
