# src/Auto3D/utils/convergence.py
"""The one place that knows what the ``Converged`` SDF property means.

``Converged`` is written by :mod:`Auto3D.batch_opt.batchopt` for every record
it optimizes -- ``"True"`` or ``"False"`` -- and read by the three filters that
decide which conformers survive (:class:`Auto3D.ranking.ConformerRanker`,
:func:`Auto3D.filtering.filter_unique_optimized`,
:func:`Auto3D.utils.chemistry.filter_unique`).

All three used to read it as::

    try:
        converged = mol.GetProp('Converged').lower() == 'true'
    except KeyError:
        converged = False

which silently deletes every record of any SDF **not** produced by
``batchopt``: an ``opt_geometry`` output, an ORCA/Gaussian export, a
hand-built conformer set. ``ConformerRanker`` is public and documented, so
pointing it at such a file returned ``[]``, wrote a **0-byte** SDF, and exited
0. The only message was an INFO line on a logger tree that has no handler
outside ``main()``.

A record that never claimed to be the output of an optimization is not a
record that failed one. Absence of the property therefore means "not filtered
on convergence" -- the record is kept, and the other filters (connectivity,
stereochemistry, RMSD, energy) still apply to it. An **explicit**
``Converged=False`` still means what it says and is still dropped.
"""
from __future__ import annotations

from rdkit import Chem

__all__ = [
    "CONVERGED_PROP",
    "set_converged",
    "has_convergence_flag",
    "converged_or_unfiltered",
]

#: Name of the SD property. ``"True"``/``"False"``, compared case-insensitively.
CONVERGED_PROP = "Converged"


def set_converged(mol: Chem.Mol, converged: bool) -> None:
    """Record whether the optimizer converged on this record.

    Args:
        mol: Molecule to annotate, in place.
        converged: Whether the optimizer reached its force criterion (and, in
            ``batchopt``, was not dropped for oscillating).
    """
    mol.SetProp(CONVERGED_PROP, str(bool(converged)))


def has_convergence_flag(mol: Chem.Mol) -> bool:
    """True if the record carries a ``Converged`` property at all."""
    return mol.HasProp(CONVERGED_PROP)


def converged_or_unfiltered(mol: Chem.Mol) -> bool:
    """Whether ``mol`` passes the convergence filter.

    Returns:
        False only when the record explicitly says it did not converge.
        A record with no ``Converged`` property is not filtered on
        convergence -- it never claimed to be an optimizer output, so there is
        no failed optimization to filter it out for -- and passes.
    """
    if not mol.HasProp(CONVERGED_PROP):
        return True
    return mol.GetProp(CONVERGED_PROP).strip().lower() == "true"
