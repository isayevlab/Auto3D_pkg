#!/usr/bin/env python
"""Which atoms are bonded, and whether a 3D geometry still agrees with that.

:func:`get_mol_connectivity` reads the bonds off the molecular graph;
:func:`check_connectivity` asks whether a conformer's interatomic distances are
consistent with them, which is how a geometry that dissociated or formed a new
bond during optimization is caught. :func:`amend_mol` is the combined
sanitize-and-validate wrapper the filters use.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

logger = logging.getLogger("auto3d")

__all__ = ["UFF_BOND_RADII", "check_connectivity", "amend_mol", "get_mol_connectivity"]

#: UFF bond radii (Rappe et al. JACS 1992, Table 1), in angstroms. These
#: radii neglect the bond-order and electronegativity corrections in the
#: original paper. Where several types exist for the same element, the
#: largest of the types reachable from ordinary organic input was used --
#: not the largest in the table outright. Hydrogen is the one place that
#: distinction matters: UFF also tabulates ``H_b`` (bridging hydrogen, the
#: three-center B-H-B bond) at 0.460 A, larger than the ``H_`` 0.354 A used
#: here, but ``H_b`` is irrelevant to druglike input and adopting it would
#: loosen every X-H broken-bond threshold by 0.13 A (N-m26 rulings Q2/R50;
#: the hydrogen qualification). Every other tabulated element's value here
#: already is its largest type (verified against an independent
#: transcription of Table 1): C_3 > C_R, N_3 > N_R, O_R > O_3, S_R > S_3+2,
#: P_3+3 > P_3+5, B_3 > B_2, and F/Cl/Br/I/Ge/As/Se/Sb/Te have one type each.
#: Consequence of the selection rule itself: a single bond-order-blind
#: reference length makes the broken-bond (1.25x) check lenient and the
#: formed-bond (1.1x) check strict, so a stretched aromatic/conjugated bond
#: or a short multiple bond can be mis-judged. The molecular graph already
#: carries bond orders (see get_mol_connectivity); a bond-order-aware
#: reference would be more accurate but is intentionally not used here.
#:
#: Exposed at module level (rather than local to check_connectivity) so
#: clash_relief.relieve_clash can scale its own bonded-pair floor by the
#: same per-element reference length (C1/NEW-1, ruling R51) instead of
#: carrying a second, independently-maintained copy of these numbers.
UFF_BOND_RADII: dict[int, float] = {
    1: 0.354,
    5: 0.838,
    6: 0.757,
    7: 0.700,
    8: 0.680,  # UFF O_R (resonant/aromatic oxygen); was 0.658 (O_3), not the largest (Q2/M1).
    9: 0.668,
    14: 1.117,
    15: 1.101,  # UFF P_3+3 (trivalent phosphorus); was 1.117, a copy of Si's radius (N-m26).
    16: 1.077,  # UFF S_R (aromatic sulfur); was 1.064 (S_3+2), not the largest (Q2/M1).
    17: 1.044,
    32: 1.197,
    33: 1.211,
    34: 1.190,
    35: 1.192,
    51: 1.407,
    52: 1.386,
    53: 1.382,
}

#: Atomic numbers already named by :func:`check_connectivity`'s untabulated-element
#: warning in this process (N-m27).
#:
#: Module state, deliberately: the skip lives in an O(n^2) atom-pair loop, and a
#: run optimizes the same species thousands of times, so a per-pair -- or even
#: per-call -- warning would bury the log it is supposed to inform. One line per
#: element per process is the budget. It also must not be silent: skipping those
#: pairs means ``check_connectivity`` returns True for a geometry it never
#: actually checked, so a dissociated metal complex passes the filter.
#:
#: Per *process*, and that is not the same as per run: ``check_connectivity``
#: runs inside each ``workflow_workers.optim_rank_wrapper`` worker, which has
#: its own copy of this module, so a run spread over N chunk workers can emit
#: the line up to N times for one element. Deduplicating across processes would
#: need shared state for a log line, which is not worth it; read the budget as
#: "not once per atom pair", not as "exactly once per run".
#:
#: An autouse fixture in ``tests/conftest.py`` gives every test a fresh set
#: (``monkeypatch.setattr(connectivity, "_REPORTED_UNTABULATED_ELEMENTS",
#: set())``) rather than leaving a previous test's elements behind, which would
#: make a test asserting an element *is* named pass or fail by ordering.
_REPORTED_UNTABULATED_ELEMENTS: set[int] = set()


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

        Any atom pair involving an element outside that table is skipped, and the
        element is named in one WARNING the first time it is met in this process
        -- once per worker process, so a chunked run may say it more than once
        for the same element (see ``_REPORTED_UNTABULATED_ELEMENTS``) -- so a
        True from this function
        for a molecule containing such an element is "found nothing wrong among
        the pairs I could judge", not "checked everything".

        Bonds involving elements outside the covalent-radii table (e.g. alkali/
        alkaline-earth counterions or transition-metal coordination bonds, M-L)
        are NOT validated -- such pairs are skipped ("no opinion"), so the
        dissociation of an M-L bond will not be flagged as invalid connectivity.
    """
    # UFF bond radii: see the module-level UFF_BOND_RADII docstring for the
    # selection rule and its one qualification (hydrogen).
    Radii = UFF_BOND_RADII

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
            # raise KeyError and crash the whole filtering pass -- but say so
            # once per element (N-m27), because "no opinion" on every pair means
            # this function returns True for a geometry it never checked.
            if atomic_num_i not in Radii or atomic_num_j not in Radii:
                for z in (atomic_num_i, atomic_num_j):
                    if z in Radii or z in _REPORTED_UNTABULATED_ELEMENTS:
                        continue
                    _REPORTED_UNTABULATED_ELEMENTS.add(z)
                    logger.warning(
                        "Element %s has no reference radius; connectivity checks "
                        "skip pairs involving it.",
                        Chem.GetPeriodicTable().GetElementSymbol(z),
                    )
                continue

            bond = mol.GetBondBetweenAtoms(atom_i_idx, atom_j_idx)
            reference_length = Radii[atomic_num_i] + Radii[atomic_num_j]
            if bond:
                # make sure the bond is not broken
                length = rdMolTransforms.GetBondLength(
                    mol.GetConformers()[0], atom_i_idx, atom_j_idx
                )
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
