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

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors, rdmolops, rdMolTransforms

from Auto3D.constants import (
    CONFORMER_MULTIPLIER,
    CONFORMER_ROTATABLE_COEFF,
    CONFORMER_ROTATABLE_EXP,
    DEFAULT_DUPLICATE_ENERGY_TOL,
    DEFAULT_RMSD_THRESHOLD,
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    MAX_CONFORMERS_CAP,
    MIN_ATOM_DISTANCE,
)
from Auto3D.utils.stereo_check import stereo_descriptors_from_3d, stereo_preserved

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
    "amend_mol",
    "get_mol_connectivity",
    "filter_unique",
]

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

    The force-field relaxation can itself invert a stereocenter or rotate a
    double bond. This runs before the enumerated SDF is written, so the
    downstream post-optimization stereochemistry check would otherwise read an
    already-changed geometry as its own "before" reference and never notice.
    Stereochemistry is therefore checked before and after the relaxation, on
    this same molecule object, and a conformer whose configuration changed is
    rejected here rather than passed downstream.

    Known limitation: the "before" snapshot is read while the conformer is
    still in violation of ``min_distance`` — by definition, since that is the
    only way execution reaches this branch. CIP perception on a geometry that
    is itself clashing is not a trustworthy baseline, unlike the equivalent
    check in ``batch_opt/batchopt.py``, whose "before" reading is always
    taken from a valid, non-clashing conformer. This matters only when the
    branch is actually reached: across roughly 650 conformers sampled from
    Auto3D's real ``EmbedMultipleConfs`` output (glucose, cholesterol, a
    tripeptide, macrocycles, a cage compound, and molecules with B/Se/
    hypervalent Si), none ever fell below the clash threshold. Under 196
    artificially forced clashes, this guard rejected 96 conformers, and about
    56% of those rejections had a post-relaxation configuration that actually
    matched the molecule's true configuration -- spurious rejections caused
    by the unreliable baseline rather than a real inversion. The known
    improvement is to compare against the molecule's graph-encoded stereo
    tags instead of a 3D read of the clashing geometry, but that needs its
    own measurement first: RDKit's graph ``AssignStereochemistry`` and
    ``AssignStereochemistryFrom3D`` label pseudoasymmetric centers
    differently (``r``/``s`` vs ``R``/``S``), which could introduce a
    systematic false positive.

    Args:
        mol: RDKit molecule holding the conformer.
        conf_id: Index of the conformer to check/optimize.
        min_distance: Minimum acceptable interatomic distance (Angstroms).

    Returns:
        True if the (possibly optimized) conformer's minimum pairwise distance
        is >= ``min_distance`` and its stereochemistry survived unchanged;
        False if it still clashes or if the relaxation changed its
        configuration.
    """
    positions = mol.GetConformer(conf_id).GetPositions()
    # Closing the dead band: a conformer exactly at the threshold is kept.
    if min_pairwise_distance(positions) >= min_distance:
        return True

    # Clashing conformer: try MMFF, fall back to UFF when MMFF is unavailable.
    before = stereo_descriptors_from_3d(mol, conf_id=conf_id)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    else:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    # Clash relief is a force-field relaxation and can invert a center just as
    # the neural network optimization can. It runs before the enumerated SDF is
    # written, so the post-optimization check downstream would read an already
    # inverted geometry as its reference and never notice. Reject the conformer
    # here instead; the embedder simply keeps the ones that survive.
    if stereo_descriptors_from_3d(mol, conf_id=conf_id) != before:
        logger.warning(
            "Discarding a conformer whose stereochemistry changed during clash "
            "relief."
        )
        return False

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

        Bonds involving elements outside the covalent-radii table (e.g. alkali/
        alkaline-earth counterions or transition-metal coordination bonds, M-L)
        are NOT validated -- such pairs are skipped ("no opinion"), so the
        dissociation of an M-L bond will not be flagged as invalid connectivity.
    """
    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # Units of angstroms
    # These radii neglect the bond-order and electronegativity corrections in the
    # original paper. Where several values exist for the same atom, the largest
    # was used. Consequence: a single bond-order-blind reference length makes the
    # broken-bond (1.25x) check lenient and the formed-bond (1.1x) check strict,
    # so a stretched aromatic/conjugated bond or a short multiple bond can be
    # mis-judged. The molecular graph already carries bond orders (see
    # get_mol_connectivity); a bond-order-aware reference would be more accurate
    # but is intentionally not used here.
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


def amend_mol(
    mol: Chem.Mol,
    sanitize: bool = False,
    check_valid: bool = False,
) -> Chem.Mol | None:
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
        # KeyError: defensive only. check_connectivity no longer raises KeyError
        #   for unknown elements (it now skips them); retained to swallow any
        #   stray dict-lookup error from RDKit internals rather than crash the
        #   amendment of a single molecule.
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
            Records marked 'Stereo_changed' are excluded.
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
        if convergence_flag and has_valid_bonds and stereo_preserved(mol):
            mols_.append(mol)
    mols = mols_

    # Remove similar structures. Strip Hs once per molecule (O(n)) instead of on
    # both sides of every comparison (O(n^2)); GetBestRMS on no-H forms is
    # symmetric so results are unchanged. The ORIGINAL (H-explicit) mols are
    # returned; no-H forms are comparison-only.
    #
    # Heavy-atom RMSD alone collapses conformers that differ only in an O-H / N-H
    # rotor orientation. Guard with an energy check: a pair counts as duplicate
    # only when the RMSD is below ``crit`` AND the energies agree within
    # DEFAULT_DUPLICATE_ENERGY_TOL. Mols without a usable 'E_tot' fall back to
    # RMSD-only (energy guard cannot apply).
    unique_mols: list[Chem.Mol] = []
    unique_noH: list[Chem.Mol] = []
    unique_energies: list[float | None] = []
    for mol_i in mols:
        mol_i_noH = Chem.RemoveHs(mol_i)
        try:
            e_i: float | None = float(mol_i.GetProp("E_tot"))
        except (KeyError, ValueError):
            e_i = None
        unique = True
        for mol_j_noH, e_j in zip(unique_noH, unique_energies, strict=True):
            try:
                # temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(mol_i_noH, mol_j_noH)
            except RuntimeError:
                # Incomparable pair: treat as distinct (not a duplicate) so the
                # conformer is kept. Using 0 would make it look like a perfect
                # duplicate and drop a genuinely distinct structure.
                rmsd = float("inf")
            energy_close = (
                e_i is None
                or e_j is None
                or abs(e_i - e_j) < DEFAULT_DUPLICATE_ENERGY_TOL
            )
            if rmsd < crit and energy_close:
                unique = False
                break
        if unique:
            unique_mols.append(mol_i)
            unique_noH.append(mol_i_noH)
            unique_energies.append(e_i)
    return unique_mols
