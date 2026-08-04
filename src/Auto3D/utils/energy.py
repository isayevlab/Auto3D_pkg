# src/Auto3D/utils/energy.py
"""The one place that knows what unit the ``E_tot`` SDF property is in.

``E_tot`` is written in **Hartree**, always, by every Auto3D writer. That is
the unit README, ``docs/legacy-v2/source/usage.rst`` and ``main()``'s own
"Energy unit: Hartree if implicit." log line have always claimed, and the unit
``Auto3D.tautomer.select_tautomers`` has always read.

Until 4.0.1 it was only *sometimes* true. ``batch_opt.optimizing.run`` wrote
``E_tot`` in eV and ``ASE/geometry.opt_geometry`` converted the same tag to
Hartree afterwards, so the identical property name carried two units depending
on which entry point produced the file -- and the five in-package consumers
(``ranking``, ``filtering.filter_unique_optimized``,
``filtering.filter_unique``) all hard-coded
eV. Feeding an ``opt_geometry`` output straight to
``ConformerRanker(window=2.0)`` therefore opened a window 27.2x too wide, kept
3 conformers where 2 belong, reported ``E_rel`` 0.037 kcal/mol where the truth
is 1.000, and wrote an ``E_tot(Hartree)`` that had been divided by 27.211
twice.

The fix is one unit for the property and one conversion boundary. Writers call
:func:`set_e_tot_from_ev` (models produce eV); readers call :func:`e_tot_ev` /
:func:`try_e_tot_ev` and get eV back, so every in-package energy tolerance
(``DEFAULT_DUPLICATE_ENERGY_TOL``, ``DEFAULT_ENERGY_CLUSTER_WINDOW``, the
kcal/mol -> eV window conversion in ``ranking``) keeps its documented eV
meaning. Writers additionally set :data:`E_TOT_HARTREE_PROP`, the unit-labeled
sibling, so a file states its own unit.
"""
from __future__ import annotations

from rdkit import Chem

from Auto3D.constants import (
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
)

__all__ = [
    "E_TOT_PROP",
    "E_TOT_HARTREE_PROP",
    "set_e_tot_from_ev",
    "e_tot_hartree",
    "e_tot_ev",
    "try_e_tot_ev",
    # Conversion factors, and their lowercase legacy spellings
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_PER_MOL",
    "EV_TO_KCAL_PER_MOL",
    "hartree2ev",
    "hartree2kcalpermol",
    "ev2kcalpermol",
]

#: Unlabeled property name, kept for backward compatibility. Hartree.
E_TOT_PROP = "E_tot"
#: Unit-labeled sibling carrying the identical value.
E_TOT_HARTREE_PROP = "E_tot(Hartree)"

# Legacy lowercase spellings of the three conversion factors in
# ``Auto3D.constants``. They are the names Auto3D 2.x used and several call
# sites still read, so they stay -- here rather than in a "chemistry" grab bag,
# since a unit conversion factor belongs with the module that owns the unit.
hartree2ev: float = HARTREE_TO_EV
hartree2kcalpermol: float = HARTREE_TO_KCAL_PER_MOL
ev2kcalpermol: float = EV_TO_KCAL_PER_MOL


def set_e_tot_from_ev(mol: Chem.Mol, energy_ev: float, *, labeled: bool = True) -> None:
    """Write ``E_tot`` (Hartree) from an energy the model produced in eV.

    Args:
        mol: Molecule to annotate, in place.
        energy_ev: Total energy in eV -- the unit every Auto3D model adapter
            returns (see ``Auto3D.models.adapter.ModelAdapter.forward``).
        labeled: Also write the unit-labeled :data:`E_TOT_HARTREE_PROP`
            sibling. Left True for anything a user reads.
    """
    hartree = float(energy_ev) / HARTREE_TO_EV
    mol.SetProp(E_TOT_PROP, str(hartree))
    if labeled:
        mol.SetProp(E_TOT_HARTREE_PROP, mol.GetProp(E_TOT_PROP))


def e_tot_hartree(mol: Chem.Mol) -> float:
    """Return ``E_tot`` in Hartree, the unit it is stored in.

    Raises:
        KeyError: The molecule has no ``E_tot`` property.
        ValueError: The property is not a number.
    """
    return float(mol.GetProp(E_TOT_PROP))


def e_tot_ev(mol: Chem.Mol) -> float:
    """Return ``E_tot`` converted to eV, the unit the filters compare in.

    Raises:
        KeyError: The molecule has no ``E_tot`` property.
        ValueError: The property is not a number.
    """
    return e_tot_hartree(mol) * HARTREE_TO_EV


def try_e_tot_ev(mol: Chem.Mol) -> float | None:
    """``e_tot_ev`` for callers that treat a missing/garbled energy as absent.

    Returns None instead of raising, which the duplicate-conformer energy
    guard uses to mean "no usable energy, fall back to RMSD only".
    """
    try:
        return e_tot_ev(mol)
    except (KeyError, ValueError):
        return None
