#!/usr/bin/env python
"""The dummy-atom predicate, read straight off a molecular graph.

A function of the graph alone (no coordinates, no force field, no energy),
which is what separates it from ``Auto3D.domain.geometry`` and
``Auto3D.domain.connectivity``. The conformer budget is domain policy, and
lives in ``Auto3D.domain.conformer_budget``.

:func:`has_dummy_atoms` lives here rather than beside the other model
precondition in ``engines/models/policy.py`` because ``utils/sdf_io.py`` needs
it too, and ``foundation`` may not import ``engines``. ``policy`` delegates
to it, so both layers read one definition.

:func:`record_name` lives here because every layer names records in
diagnostics, and ``foundation`` is the one layer all of them can import.
"""

from __future__ import annotations

from rdkit import Chem

__all__ = ["has_dummy_atoms", "record_name"]


def has_dummy_atoms(mol: Chem.Mol) -> bool:
    """True if any atom has atomic number 0 (``*``, ``[3*]``: an R-group placeholder).

    A dummy atom is not a species. AIMNet2 uses index 0 as its embedding
    padding and would score it as a zero-feature ghost; ANI refuses
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


def record_name(mol: Chem.Mol, default: str = "molecule") -> str:
    """The molecule's ``_Name`` property, or ``default`` when it has none.

    The one definition of the record-label idiom for log lines and error
    messages (``mol.GetProp("_Name") if mol.HasProp("_Name") else <default>``).
    The default is a parameter, not a constant: some callers want the generic
    ``"molecule"``, others thread a position-based fallback such as
    ``f"record {position}"`` through explicitly.
    """
    return mol.GetProp("_Name") if mol.HasProp("_Name") else default
