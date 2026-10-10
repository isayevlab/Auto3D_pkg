#!/usr/bin/env python
"""The dummy-atom predicate, read straight off a molecular graph.

A function of the graph alone (no coordinates, no force field, no energy),
which is what separates it from ``Auto3D.domain.geometry`` and
``Auto3D.domain.connectivity``. The conformer budget used to live here too;
it is domain policy, so it moved to ``Auto3D.domain.conformer_budget``.

:func:`has_dummy_atoms` lives here rather than beside the other model
precondition in ``engines/models/policy.py`` because ``utils/sdf_io.py`` needs
it too, and ``foundation`` may not import ``engines``. ``policy`` re-exports
it, so both layers read one definition.
"""

from __future__ import annotations

from rdkit import Chem

__all__ = ["has_dummy_atoms"]


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
