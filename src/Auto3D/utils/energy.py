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
(``ranking``, ``filtering.filter_conformers``) all hard-coded
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

from typing import TYPE_CHECKING

from rdkit import Chem

from Auto3D.constants import (
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
)
from Auto3D.utils.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = get_logger(__name__)

__all__ = [
    "E_TOT_PROP",
    "E_TOT_HARTREE_PROP",
    "E_REL_KCAL_PROP",
    "set_e_tot_from_ev",
    "set_relative_energies",
    "clear_relative_energies",
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
#: Energy relative to the lowest conformer of the same molecule, in kcal/mol.
#: Documented output (README, ``docs/source/usage.rst``); written by
#: ``ranking.run`` for a ``main()`` output and by :func:`set_relative_energies`
#: for a ``calc_thermo`` one.
E_REL_KCAL_PROP = "E_rel(kcal/mol)"

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


def set_relative_energies(mols: Sequence[Chem.Mol]) -> None:
    """Write ``E_rel(kcal/mol)`` per conformer group, against the group minimum.

    "Relative to the best conformer of that molecule" is what README and
    ``docs/source/usage.rst`` promise the property means, so the grouping and
    the reference are part of what this module owns, not an incidental detail of
    whichever caller happens to need it.

    ``ranking.run`` computes the same quantity for a ``main()`` output while it
    still holds one conformer group at a time. ``calc_thermo`` cannot: it
    relaxes each record independently and only has the full set afterwards, and
    the relaxation replaces the ``E_tot`` the incoming value derived from. This
    is the second-pass form for that caller.

    Grouping is on ``_Name`` **verbatim**. It is tempting to reuse
    ``ranking.species_id``, which strips ``<isomer>_<conformer>``; that is wrong
    here and silently so. By the time a file is written, ``ranking.run`` has
    already stripped the name, so ``_Name`` *is* the group key -- applying the
    strip a second time is not idempotent and turns ``aspirin_analog_3`` into
    ``aspirin``, merging compounds that share a prefix.

    Two groups get nothing rather than a wrong number, because a relative
    energy is only defined within one compound and the caller may have been
    handed an arbitrary SDF:

    * an untitled group (``_Name`` empty or absent), which would otherwise
      collect every such record into one bucket;
    * a group whose members are not the same compound, judged by
      ``utils.stereo_check.species_key`` and formal charge. Reusing a title
      across different molecules is ordinary in a hand-built file, and the
      difference between two compounds' energies looks exactly like a
      conformational preference.

    In both cases any pre-existing value is cleared, so the property present on
    a record always means "this took part in a valid comparison". A record with
    no readable ``E_tot`` is skipped individually and does not disqualify the
    rest of its group.

    Args:
        mols: The records to annotate, already filtered to those that are
            comparable. ``calc_thermo`` passes successes only -- a saddle point
            or a record that failed the stationary-point gate must not be a
            group member, and must not be the reference either.
    """
    # Local import: `stereo_check` pulls rdkit stereo perception, which this
    # module's readers (`filtering`, `ranking`) do not otherwise need.
    from Auto3D.utils.stereo_check import species_key

    groups: dict[str, list[Chem.Mol]] = {}
    for mol in mols:
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        groups.setdefault(name, []).append(mol)

    for name, group in groups.items():
        if not name:
            clear_relative_energies(group)
            logger.warning(
                "%d record(s) carry no name; relative energies need a group and "
                "a title is the only thing identifying one, so E_rel(kcal/mol) "
                "is withheld for them.", len(group),
            )
            continue

        identities = {(species_key(mol), Chem.GetFormalCharge(mol)) for mol in group}
        if len(identities) > 1:
            clear_relative_energies(group)
            logger.warning(
                "%d record(s) named %r are not all the same compound (%d distinct "
                "species); E_rel(kcal/mol) is withheld rather than subtracting "
                "the energies of different molecules.",
                len(group), name, len(identities),
            )
            continue

        with_energy = [(mol, try_e_tot_ev(mol)) for mol in group]
        usable = [(mol, e) for mol, e in with_energy if e is not None]
        # A record with no readable E_tot cannot carry a relative one either.
        clear_relative_energies(mol for mol, e in with_energy if e is None)
        if not usable:
            continue

        reference = min(e for _, e in usable)
        for mol, energy in usable:
            mol.SetProp(E_REL_KCAL_PROP, str((energy - reference) * EV_TO_KCAL_PER_MOL))


def clear_relative_energies(mols: Iterable[Chem.Mol]) -> None:
    """Drop a relative energy this run did not recompute.

    The counterpart to :func:`set_relative_energies`, for records that must not
    carry the property: ones whose absolute energy was replaced without a valid
    group to measure against, and ones excluded from the comparison entirely.
    ``calc_thermo`` uses it on its failures, which never reach ``do_mol_thermo``
    and so still hold the *input* ``E_tot``. Without it the property would
    survive on exactly the records a reader is told to discard.
    """
    for mol in mols:
        if mol.HasProp(E_REL_KCAL_PROP):
            mol.ClearProp(E_REL_KCAL_PROP)
