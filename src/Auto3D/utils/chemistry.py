#!/usr/bin/env python
"""Chemistry-related utility functions for Auto3D.

This module provides:
- Energy unit conversions (Hartree, eV, kcal/mol)
- Molecular charge calculation
- Geometry utilities (pairwise distances, RMSD)
- Molecular connectivity analysis
- RMSD-based duplicate filtering
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors, rdMolTransforms, rdmolops

from Auto3D.constants import (
    CONFORMER_MULTIPLIER,
    CONFORMER_ROTATABLE_COEFF,
    CONFORMER_ROTATABLE_EXP,
    DEFAULT_RMSD_THRESHOLD,
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    MAX_CONFORMERS_CAP,
    MIN_ATOM_DISTANCE,
)

logger = logging.getLogger("auto3d")

# Re-export constants for convenience
__all__ = [
    # Constants
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_PER_MOL",
    "EV_TO_KCAL_PER_MOL",
    # Backward compatibility aliases
    "hartree2ev",
    "hartree2kcalpermol",
    "ev2kcalpermol",
    # Functions
    "calculate_conformer_count",
    "get_mol_charge",
    "min_pairwise_distance",
    "relieve_clash",
    "get_rmsd",
    "check_connectivity",
    "getidx",
    "amend_mol",
    "get_mol_connectivity",
    "filter_unique",
]

# ANI2xt element mapping: atomic number -> index
ANI2XT_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

# Backward compatibility aliases for energy conversion factors
hartree2ev: float = HARTREE_TO_EV
hartree2kcalpermol: float = HARTREE_TO_KCAL_PER_MOL
ev2kcalpermol: float = EV_TO_KCAL_PER_MOL


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
        CONFORMER_MULTIPLIER * CONFORMER_ROTATABLE_COEFF *
        (num_rotatable ** CONFORMER_ROTATABLE_EXP)
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


def relieve_clash(
    mol: Chem.Mol,
    conf_id: int,
    min_distance: float = MIN_ATOM_DISTANCE,
) -> bool:
    """Optimize a clashing conformer in place and report whether it is usable.

    A conformer is considered "clashing" when its minimum pairwise interatomic
    distance is below ``min_distance``. Such conformers are relaxed with MMFF;
    when the molecule lacks full MMFF parameters (elements like B, Se or some
    Si valences, where ``MMFFOptimizeMolecule`` returns -1 and does nothing),
    the function falls back to UFF so the conformer is not discarded for lack
    of a force field.

    Args:
        mol: RDKit molecule holding the conformer.
        conf_id: Index of the conformer to check/optimize.
        min_distance: Minimum acceptable interatomic distance (Angstroms).

    Returns:
        True if the (possibly optimized) conformer's minimum pairwise distance
        is >= ``min_distance`` and should be kept; False if it still clashes.
    """
    positions = mol.GetConformer(conf_id).GetPositions()
    # Closing the dead band: a conformer exactly at the threshold is kept.
    if min_pairwise_distance(positions) >= min_distance:
        return True

    # Clashing conformer: try MMFF, fall back to UFF when MMFF is unavailable.
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    else:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    positions = mol.GetConformer(conf_id).GetPositions()
    return min_pairwise_distance(positions) >= min_distance


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
        The RMSD value in Angstroms. Returns 0.0 if alignment fails
        (e.g., due to atom mismatch).

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
        rmsd = 0.0
    return float(rmsd)


def check_connectivity(mol: Chem.Mol) -> bool:
    """Check if there is a new bond formed or a bond broken in the molecule.

    This function validates molecular connectivity by comparing actual interatomic
    distances against reference bond lengths based on UFF radii. It detects both
    broken bonds (distances too large) and formed bonds (distances too small).

    Args:
        mol: RDKit molecule object with conformer information.

    Returns:
        True if connectivity is valid (no broken/formed bonds), False otherwise.

    Note:
        Uses UFF bond radii from Rappe et al. JACS 1992. The radii neglect bond-order
        and electronegativity corrections. Bond is considered broken if length > 1.25x
        reference, and formed if distance < 1.1x reference.
    """
    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # Units of angstroms
    # These radii neglect the bond-order and electronegativity corrections in the original paper.
    # Where several values exist for the same atom, the largest was used.
    Radii = {
        1: 0.354,
        5: 0.838,
        6: 0.757,
        7: 0.700,
        8: 0.658,
        9: 0.668,
        14: 1.117,
        15: 1.117,
        16: 1.064,
        17: 1.044,
        32: 1.197,
        33: 1.211,
        34: 1.190,
        35: 1.192,
        51: 1.407,
        52: 1.386,
        53: 1.382,
    }

    atoms = [atom for atom in mol.GetAtoms()]
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n, 1):
            atom_i = atoms[i]
            atom_i_idx = atom_i.GetIdx()
            atomic_num_i = atom_i.GetAtomicNum()
            pos_i = mol.GetConformer().GetAtomPosition(atom_i_idx)

            atom_j = atoms[j]
            atom_j_idx = atom_j.GetIdx()
            atomic_num_j = atom_j.GetAtomicNum()
            pos_j = mol.GetConformer().GetAtomPosition(atom_j_idx)

            # Elements outside the UFF radii table (e.g. Na, K, Mg, Fe, Zn in
            # salts/metal complexes) have no reference radius. Skip such pairs
            # ("no opinion") rather than indexing the dict blindly, which would
            # raise KeyError and crash the whole filtering pass.
            if atomic_num_i not in Radii or atomic_num_j not in Radii:
                continue

            bond = mol.GetBondBetweenAtoms(atom_i_idx, atom_j_idx)
            reference_length = Radii[atomic_num_i] + Radii[atomic_num_j]
            if bond:
                # make sure the bond is not broken
                length = rdMolTransforms.GetBondLength(mol.GetConformers()[0], atom_i_idx, atom_j_idx)
                if length > reference_length * 1.25:
                    return False
            else:
                # make sure the bond is not formed
                dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if dist < reference_length * 1.1:
                    return False
    return True


def getidx(atomic_num: int, model: str = "default") -> int:
    """Get the element index for a given atomic number.

    Maps atomic numbers to element indices used by different neural network models.
    For ANI2xt, this maps to the specific element ordering (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6).
    For other models, the atomic number is returned unchanged.

    Args:
        atomic_num: The atomic number of the element (e.g., 1 for H, 6 for C).
        model: The model name. Use "ANI2xt" for ANI2xt-specific indexing,
            or any other value for standard atomic number passthrough.

    Returns:
        The element index appropriate for the specified model.

    Raises:
        KeyError: If the element is not supported by the specified model (ANI2xt only).

    Example:
        >>> getidx(6, model="ANI2xt")  # Carbon in ANI2xt
        1
        >>> getidx(6)  # Carbon with default model
        6
    """
    if model == "ANI2xt":
        return ANI2XT_INDEX[atomic_num]
    return atomic_num


def amend_mol(
    mol: Chem.Mol,
    sanitize: bool = False,
    check_valid: bool = False,
) -> Optional[Chem.Mol]:
    """Attempt to fix or validate a molecule.

    This function can optionally sanitize a molecule and check its validity.
    If check_valid is True and the molecule has invalid connectivity (broken
    or formed bonds), None is returned.

    Args:
        mol: RDKit Mol object to amend.
        sanitize: If True, sanitize the molecule using RDKit's SanitizeMol.
        check_valid: If True, check connectivity and return None if invalid.

    Returns:
        The amended molecule, or None if the molecule is invalid and check_valid is True.

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> amended = amend_mol(mol, sanitize=True)
        >>> amended is not None
        True
    """
    if mol is None:
        return None

    try:
        if sanitize:
            Chem.SanitizeMol(mol)

        if check_valid:
            # Check if molecule has valid 3D coordinates
            if mol.GetNumConformers() > 0:
                if not check_connectivity(mol):
                    return None

        return mol
    except (ValueError, RuntimeError, KeyError) as e:
        # ValueError: from RDKit SanitizeMol validation errors
        # RuntimeError: from RDKit internal errors during molecule processing
        # KeyError: from check_connectivity if atom's atomic number not in Radii dict
        logger.debug(f"Molecule amendment failed: {type(e).__name__}: {e}")
        return None


def get_mol_connectivity(
    mol: Chem.Mol,
    include_bond_order: bool = False,
) -> set[tuple[int, int]] | set[tuple[int, int, float]]:
    """Get the bond connectivity of a molecule.

    Returns a set of tuples representing bonds in the molecule. Each tuple
    contains the indices of the two bonded atoms, optionally with the bond order.

    Args:
        mol: RDKit Mol object.
        include_bond_order: If True, include bond order as the third element
            of each tuple.

    Returns:
        A set of tuples. Each tuple is (atom1_idx, atom2_idx) if include_bond_order
        is False, or (atom1_idx, atom2_idx, bond_order) if True. The atom indices
        are sorted so that atom1_idx < atom2_idx.

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CC")
        >>> get_mol_connectivity(mol)
        {(0, 1)}
        >>> mol = Chem.MolFromSmiles("C=C")
        >>> get_mol_connectivity(mol, include_bond_order=True)
        {(0, 1, 2.0)}
    """
    connectivity: set = set()

    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        # Ensure consistent ordering (smaller index first)
        if atom1_idx > atom2_idx:
            atom1_idx, atom2_idx = atom2_idx, atom1_idx

        if include_bond_order:
            bond_order = bond.GetBondTypeAsDouble()
            connectivity.add((atom1_idx, atom2_idx, bond_order))
        else:
            connectivity.add((atom1_idx, atom2_idx))

    return connectivity


def filter_unique(mols: list[Chem.Mol], crit: float = DEFAULT_RMSD_THRESHOLD) -> list[Chem.Mol]:
    """Remove structures that are very similar and remove unconverged structures.

    This function filters a list of molecules to keep only unique, converged structures.
    It first removes unconverged structures and those with invalid connectivity,
    then removes similar structures based on RMSD comparison.

    Args:
        mols: List of RDKit molecule objects with 'Converged' property set.
        crit: RMSD threshold for considering two structures as identical.
            Structures with RMSD below this value are considered duplicates.
            Defaults to DEFAULT_RMSD_THRESHOLD (0.3 Angstroms).

    Returns:
        List of unique, converged molecules with valid connectivity.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import AllChem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMolecule(mol, randomSeed=42)
        0
        >>> mol.SetProp("Converged", "true")
        >>> filter_unique([mol], crit=0.3)  # Returns list with 1 molecule
        [...]
    """
    # Remove unconverged structures
    mols_: list[Chem.Mol] = []
    for mol in mols:
        try:
            convergence_flag = mol.GetProp("Converged").lower() == "true"
        except KeyError:
            convergence_flag = False
        has_valid_bonds = check_connectivity(mol)
        if convergence_flag and has_valid_bonds:
            mols_.append(mol)
    mols = mols_

    # Remove similar structures
    unique_mols: list[Chem.Mol] = []
    for mol_i in mols:
        unique = True
        for mol_j in unique_mols:
            try:
                # temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol_i), Chem.RemoveHs(mol_j))
            except RuntimeError:
                # Incomparable pair: treat as distinct (not a duplicate) so the
                # conformer is kept. Using 0 would make it look like a perfect
                # duplicate and drop a genuinely distinct structure.
                rmsd = float("inf")
            if rmsd < crit:
                unique = False
                break
        if unique:
            unique_mols.append(mol_i)
    return unique_mols
