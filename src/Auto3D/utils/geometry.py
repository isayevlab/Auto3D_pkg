#!/usr/bin/env python
"""Geometric measurements on a conformer's coordinates.

Distances and RMSD only. Nothing here reads or writes a molecular property,
consults a force field, or decides whether a structure is acceptable -- those
belong to ``utils/connectivity.py``, ``Auto3D.clash_relief`` and
``Auto3D.filtering`` respectively.
"""
from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

logger = logging.getLogger("auto3d")

__all__ = ["min_pairwise_distance", "get_rmsd"]


def min_pairwise_distance(points: np.ndarray) -> float:
    """Find the minimum pairwise distance among n points in 3D space.

    This function computes all pairwise distances between the provided points
    and returns the minimum distance. It uses vectorized NumPy operations
    for efficiency.

    Args:
        points: A (n, 3) array representing the coordinates of n points
            in 3D space.

    Returns:
        The minimum pairwise distance among the n points.

    Example:
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
        >>> min_pairwise_distance(points)
        1.0
    """
    # Ensure input is a NumPy array with float32 type
    points = points.astype(np.float32)
    n = points.shape[0]

    # Guard for single atom or empty input
    if n < 2:
        # Single atom: no pairwise distance exists
        return float('inf')

    # Expand dimensions of points to enable broadcasting
    points_expanded = np.expand_dims(points, axis=1).repeat(n, axis=1)

    # Compute pairwise squared differences
    diff_squared = (points_expanded - points_expanded.transpose(1, 0, 2)) ** 2

    # Sum along the last dimension to get pairwise squared distances
    pairwise_squared_distances = np.sum(diff_squared, axis=-1)

    # Find the minimum squared distance from upper triangle
    upp_indices = np.triu_indices(n, 1)
    upp_values = pairwise_squared_distances[upp_indices]
    min_squared_distance = np.min(upp_values)

    # Return the square root of the minimum squared distance
    return float(np.sqrt(min_squared_distance))


def get_rmsd(mol1: Chem.Mol, mol2: Chem.Mol, remove_hs: bool = True) -> float:
    """Calculate the RMSD between two molecular conformers.

    Uses RDKit's GetBestRMS function which finds the optimal alignment
    between the two molecules before computing RMSD.

    Args:
        mol1: First RDKit Mol object with a conformer.
        mol2: Second RDKit Mol object with a conformer.
        remove_hs: If True (default), remove hydrogens before RMSD calculation.
            This speeds up the calculation and focuses on heavy atom positions.

    Returns:
        The RMSD value in Angstroms. Returns ``float("inf")`` if alignment
        fails (e.g., due to atom mismatch). An incomparable pair is treated as
        "distinct" rather than "identical", which is the same convention used
        by ``filter_unique``; a downstream ``rmsd < threshold`` check therefore
        keeps the structure instead of dropping it as a false duplicate.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import AllChem
        >>> mol1 = Chem.MolFromSmiles("CCO")
        >>> mol1 = Chem.AddHs(mol1)
        >>> AllChem.EmbedMolecule(mol1)
        0
        >>> mol2 = Chem.Mol(mol1)  # Copy
        >>> get_rmsd(mol1, mol2)
        0.0
    """
    try:
        if remove_hs:
            mol1_proc = Chem.RemoveHs(mol1)
            mol2_proc = Chem.RemoveHs(mol2)
        else:
            mol1_proc = mol1
            mol2_proc = mol2
        # Temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
        rmsd = rdMolAlign.GetBestRMS(mol1_proc, mol2_proc)
    except RuntimeError:
        # Incomparable pair: treat as distinct (inf), matching filter_unique.
        rmsd = float("inf")
    return float(rmsd)
