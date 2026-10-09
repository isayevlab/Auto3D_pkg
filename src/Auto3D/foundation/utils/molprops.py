#!/usr/bin/env python
"""Scalar properties read straight off a molecular graph.

The conformer budget and the dummy-atom predicate: functions of the graph
alone (no coordinates, no force field, no energy), which is what separates
them from ``utils/geometry.py`` and ``utils/connectivity.py``.

:func:`has_dummy_atoms` lives here rather than beside the other model
precondition in ``engines/models/policy.py`` because ``utils/sdf_io.py`` needs
it too, and ``foundation`` may not import ``engines``. ``policy`` re-exports
it, so both layers read one definition.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from Auto3D.foundation.constants import (
    CONFORMER_MULTIPLIER,
    CONFORMER_ROTATABLE_COEFF,
    CONFORMER_ROTATABLE_EXP,
    MAX_CONFORMERS_CAP,
)

__all__ = ["calculate_conformer_count", "has_dummy_atoms"]


def has_dummy_atoms(mol: Chem.Mol) -> bool:
    """True if any atom has atomic number 0 (``*``, ``[3*]``: an R-group placeholder).

    A dummy atom is not a species. AIMNet2 uses index 0 as its embedding
    padding and would score it as a zero-feature ghost (N-M3); ANI refuses
    it as out-of-set. Records containing one are skipped at every seam
    that consumes records, with a warning, and reported by reconciliation.

    Args:
        mol: RDKit molecule object (with or without hydrogens, with or
            without a conformer).

    Returns:
        True if the molecule carries at least one dummy atom.

    Example:
        >>> from rdkit import Chem
        >>> has_dummy_atoms(Chem.MolFromSmiles("*CCO"))
        True
        >>> has_dummy_atoms(Chem.MolFromSmiles("CCO"))
        False
    """
    return any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())


def calculate_conformer_count(mol: Chem.Mol) -> int:
    """Calculate the number of conformers to generate for a molecule.

    Uses a formula based on the number of rotatable bonds, with a minimum
    of the heavy atom count and a maximum cap. The result is floored at 1 so
    a molecule never gets 0 conformers (which would silently drop tiny species
    such as ``[H+]`` or a lone atom from the pipeline).

    Rotatable bonds are counted on the heavy-atom graph (``Chem.RemoveAllHs``,
    which also drops isotope-labeled hydrogens), not on whatever hydrogen state
    ``mol`` happens to carry: with explicit
    hydrogens, RDKit's ``CalcNumRotatableBonds`` also counts C-O-H / C-N-H
    torsions (glycerol requests 238 conformers with explicit Hs, 52 without),
    for hydroxyl/amine orientations that heavy-atom symmetry pruning collapses
    to one per skeleton (see ``Auto3D.domain.embedding.embed_params``) -- the
    larger with-H request bought nothing for those orientations, only more
    ETKDG attempts, which did leave more distinct skeletons for polyols after
    pruning (measured with those same embedding settings at
    ``CONFORMER_RANDOM_SEED``, ``pruneRmsThresh=0.3``,
    ``useSymmetryForPruning=True``: glycerol, ``OCC(O)CO``, keeps 9 conformers
    whether 52 or 238 are requested, under RDKit 2025.09.6 and under 2026.9.1
    alike; beta-D-glucopyranose,
    ``C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O``, keeps 12 of 16
    requested at the new heavy-atom budget vs. 69 of 321 requested at the old
    with-H budget under RDKit 2025.09.6, and 14 of 16 vs. 83 of 321 under
    2026.9.1 -- the installed RDKit changes how many of the requested pool
    survive pruning, not which budget is smaller; see
    ``benchmarks/results-notes/2026-10-09-rdkit-2026-09-kept-counts.md``).
    Computing the budget on the heavy-atom
    graph makes the SMILES path (no-H) and the SDF/parallel paths (with-H)
    agree on the same request; a caller that wants the larger pool back asks
    for it explicitly with ``max_confs``.

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
    # RemoveAllHs, not RemoveHs: the latter keeps isotope-labeled hydrogens
    # (D, T), so a deuterated hydroxyl would still count its C-O-H torsion and
    # a labeled analog would get a larger budget than its protium parent.
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveAllHs(mol))
    num_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    formula_count = int(
        CONFORMER_MULTIPLIER * CONFORMER_ROTATABLE_COEFF * (num_rotatable**CONFORMER_ROTATABLE_EXP)
    )

    # Floor at 1: a heavy-atom-free species (e.g. [H+]) or a single atom must
    # still receive at least one conformer instead of being silently dropped.
    return min(max(1, num_heavy, formula_count), MAX_CONFORMERS_CAP)
