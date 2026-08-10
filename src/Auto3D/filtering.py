#!/usr/bin/env python
"""The one conformer filter: duplicate removal with hierarchical RMSD comparison.

Auto3D carried two implementations of this until 3.0.0 -- this energy-clustered
one and a legacy all-pairs ``filter_unique``, selected by
``ConformerRanker(use_optimized_filtering=...)``. They applied the identical
duplicate criterion and were kept side by side so each could act as the other's
oracle, but a boolean kwarg that swaps one filter for another is two behaviors
to keep in step, and they had already drifted on malformed input: the legacy one
tolerated a record with no usable ``E_tot`` while this one raised ``KeyError``
from its sort key. The survivor tolerates it (see :func:`_energy_sort_key`), and
the flag and the duplicate implementation are gone.

The filter reports **why** it dropped things, not just what survived
(:class:`FilterResult`). ``ranking`` used to log "No structure converged" for a
species whose conformers were all dropped for *stereochemistry* -- a message
that sent the reader to the optimizer settings for a problem in the input's
stereo definitions.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from Auto3D.constants import (
    DEFAULT_DUPLICATE_ENERGY_TOL,
    DEFAULT_ENERGY_CLUSTER_WINDOW,
    DEFAULT_RMSD_THRESHOLD,
)
from Auto3D.utils.connectivity import check_connectivity
from Auto3D.utils.convergence import (
    converged_or_unfiltered,
    thermo_succeeded_or_unfiltered,
)
from Auto3D.utils.energy import try_e_tot_ev
from Auto3D.utils.stereo_check import species_key, stereo_preserved

__all__ = [
    "DROP_REASONS",
    "FilterResult",
    "filter_conformers",
    "filter_unique_optimized",
]

#: Every reason a conformer can leave the selected set, in report order.
#:
#: The authoritative vocabulary: :class:`FilterResult` refuses a count keyed by
#: anything else, so a producer that invents a reason fails at construction
#: rather than contributing a silently unlabeled drop to a diagnostic.
#: ``energy_window`` is produced by ``ranking.ConformerRanker.top_window``
#: rather than by this module -- it is a selection criterion, not a validity
#: one -- but it belongs in the same vocabulary because it reaches the user
#: through the same message.
DROP_REASONS: tuple[str, ...] = (
    "unparsed",
    "unconverged",
    "thermo_failed",
    "stereochemistry",
    "connectivity",
    "duplicate",
    "energy_window",
)

#: Human-readable phrase per reason, used by :meth:`FilterResult.summary`.
_REASON_PHRASES: dict[str, str] = {
    "unparsed": "unparseable by RDKit",
    "unconverged": "marked Converged=false",
    "thermo_failed": "not a minimum (Thermo_failed is set)",
    "stereochemistry": "changed stereochemistry during optimization",
    "connectivity": "have broken or newly formed bonds",
    "duplicate": "duplicates of a kept conformer",
    "energy_window": "outside the energy window",
}


@dataclass(frozen=True)
class FilterResult:
    """What survived filtering, and a count of what did not, by reason.

    Deliberately tiny: a list and a ``{reason: count}`` dict. The alternative
    considered -- attaching a reason to each dropped molecule -- would keep
    every rejected conformer alive for the duration of a chunk, which is the
    memory the filter exists to release.

    Args:
        kept: Surviving molecules, sorted by energy (lowest first); records
            with no usable energy sort last.
        dropped: Count per reason. Keys must come from :data:`DROP_REASONS`;
            reasons that did not fire may be omitted or present as 0.

    Raises:
        ValueError: ``dropped`` carries a key outside :data:`DROP_REASONS`.
    """

    kept: list[Chem.Mol]
    dropped: dict[str, int]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.dropped) - set(DROP_REASONS))
        if unknown:
            raise ValueError(
                f"unknown filter drop reason(s) {unknown}; expected one of "
                f"{list(DROP_REASONS)}"
            )

    @property
    def reasons(self) -> tuple[str, ...]:
        """The reasons that actually fired, in :data:`DROP_REASONS` order."""
        return tuple(r for r in DROP_REASONS if self.dropped.get(r))

    def summary(self) -> str:
        """One phrase per reason that fired, e.g. ``"2 marked Converged=false"``.

        Empty string when nothing was dropped, so a caller can test it for
        truth rather than special-casing a placeholder.
        """
        return ", ".join(
            f"{self.dropped[reason]} {_REASON_PHRASES[reason]}"
            for reason in self.reasons
        )


def _energy_sort_key(mol: Chem.Mol) -> tuple[bool, float]:
    """Sort key that tolerates a record with no usable ``E_tot``.

    Returns ``(False, energy_ev)`` for a record that has one and
    ``(True, 0.0)`` for a record that does not, so the energy-less records land
    **after** every record that has an energy, whatever their values.

    The 0.0 is a tie-break placeholder, never an energy: reading a missing
    ``E_tot`` as 0.0 on its own would sort a garbage record ahead of every
    genuine structure (real ``E_tot`` values are large and negative), making it
    the reference conformer ``E_rel`` is measured from and the single structure
    a ``k=1`` request returns. Among themselves, energy-less records keep their
    input order (``list.sort`` is stable), which is the only ordering there is
    any evidence for.
    """
    energy = try_e_tot_ev(mol)
    if energy is None:
        return (True, 0.0)
    return (False, energy)


def filter_conformers(
    mols: list[Chem.Mol],
    *,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
) -> FilterResult:
    """Remove duplicate conformers, skipping RMSD comparisons that cannot match.

    Sorts by energy and only RMSD-compares molecules close enough in energy to
    be duplicates at all, which avoids comparing all pairs **without changing
    which molecules survive** -- the partitioning is chosen so that no duplicate
    pair can be separated by it. See the comment on the split rule below for why
    that holds; it did not hold before 3.0.0.

    A pair counts as a duplicate only when all three of these agree: the two are
    the same compound (:func:`Auto3D.utils.stereo_check.species_key`), their
    heavy-atom RMSD is under ``rmsd_threshold``, and their energies agree within
    ``DEFAULT_DUPLICATE_ENERGY_TOL`` (or at least one of them has no usable
    energy, in which case that term cannot apply and RMSD alone decides).

    Args:
        mols: List of RDKit Mol objects with 'E_tot' (Hartree) and, optionally,
            'Converged' properties. A record whose 'Converged' property is
            explicitly false is dropped; a record without the property is kept
            (it is not filtered on convergence). Records marked
            'Stereo_changed' are excluded. ``None`` entries -- what
            ``SDMolSupplier`` yields for a record RDKit cannot parse -- are
            counted and skipped.
        rmsd_threshold: RMSD threshold for considering structures similar (Angstrom).
        energy_cluster_window: Energy width (eV) below which molecules are
            compared to each other. A performance knob only: values below the
            duplicate energy tolerance cannot shrink the comparison set, because
            that tolerance is the floor at which a pair can still be a duplicate.
            The stored Hartree energies are converted to eV on read
            (``Auto3D.utils.energy``), so the unit is as documented.

    Returns:
        A :class:`FilterResult` whose ``kept`` list is sorted by energy (lowest
        first) and whose ``dropped`` counts say why the rest are missing.
    """
    dropped: dict[str, int] = {}

    def _drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    # Filter converged structures with valid connectivity. A record with no
    # 'Converged' property is not filtered on convergence (see
    # Auto3D.utils.convergence): treating its absence as failure deleted every
    # record of any SDF batchopt did not write.
    #
    # Checked in this order, one reason attributed per record, so a structure
    # that fails several is reported under the first -- the same short-circuit
    # order the single `and` chain here used to have, hence the same verdicts.
    valid_mols: list[Chem.Mol] = []
    for mol in mols:
        if mol is None:
            _drop("unparsed")
        elif not converged_or_unfiltered(mol):
            _drop("unconverged")
        elif not thermo_succeeded_or_unfiltered(mol):
            # A saddle point or a failed stationary-point gate. Its electronic
            # energy can sit below a genuine minimum's, so leaving it in the
            # running is a way to publish the wrong structure as the most
            # stable conformer.
            _drop("thermo_failed")
        elif not stereo_preserved(mol):
            _drop("stereochemistry")
        elif not check_connectivity(mol):
            _drop("connectivity")
        else:
            valid_mols.append(mol)

    if not valid_mols:
        return FilterResult(kept=[], dropped=dropped)

    # Sort by energy. E_tot is stored in Hartree; energy_cluster_window and
    # the duplicate tolerance below are both in eV, so convert on read.
    # Records with no usable energy sort last -- see _energy_sort_key.
    valid_mols.sort(key=_energy_sort_key)
    energies = [try_e_tot_ev(mol) for mol in valid_mols]

    # Partition the energy-sorted list into runs, and only RMSD-compare within a
    # run. Where a run may end is a correctness question, not a tuning one.
    #
    # The old rule measured each molecule against its run's MINIMUM and started a
    # new run past `energy_cluster_window`. That could put two molecules 2e-6 eV
    # apart into different runs -- one just inside the window, the next just
    # outside -- so a pair that is unambiguously the same conformer was never
    # compared. Three bit-identical geometries at 0.0 / 0.099999 / 0.100001 eV all
    # survived, which is how a k=5 request came back with five slots holding two
    # distinct structures.
    #
    # A pair can only be a duplicate if its energies agree within
    # DEFAULT_DUPLICATE_ENERGY_TOL (`_filter_within_cluster` requires it). So a
    # run may end wherever the gap to the PREVIOUS molecule exceeds that
    # tolerance, and nowhere else: a gap that large proves no pair straddling it
    # can be a duplicate, and conversely, for any duplicate pair every gap
    # between them is smaller still, so no boundary can fall between them. The
    # result is identical to comparing all pairs.
    #
    # `energy_cluster_window` is honored as a lower bound on the split gap, since
    # splitting on LARGER gaps is always safe (it only merges runs and compares
    # more pairs). It therefore stays a performance knob and can no longer become
    # a correctness hole. Runs are no longer width-bounded, so a dense energy
    # ladder degrades to comparing all pairs -- the price of the guarantee, on
    # conformer counts that are tens per species.
    #
    # That entire argument rests on every record HAVING an energy. As soon as one
    # does not, no gap proves anything about it: `_filter_within_cluster` falls
    # back to RMSD-only for a pair where either side's energy is missing, so such
    # a record can be a duplicate of any same-species structure at any energy.
    # The honest response is to stop partitioning and compare all pairs, which is
    # exactly what the legacy all-pairs filter did with this input -- so the
    # survivor keeps its verdicts. Malformed input is rare and reaches here only
    # through a direct API call (`ConformerRanker` refuses a record with no
    # 'E_tot' up front), so the quadratic cost is paid where it is warranted.
    if any(energy is None for energy in energies):
        clusters: list[list[Chem.Mol]] = [valid_mols]
    else:
        split_gap = max(DEFAULT_DUPLICATE_ENERGY_TOL, energy_cluster_window)
        clusters = []
        current_cluster: list[Chem.Mol] = [valid_mols[0]]
        previous_e = energies[0]

        for mol, e in zip(valid_mols[1:], energies[1:], strict=True):
            if e - previous_e <= split_gap:
                current_cluster.append(mol)
            else:
                clusters.append(current_cluster)
                current_cluster = [mol]
            previous_e = e
        clusters.append(current_cluster)

    # Filter unique within each cluster
    unique_mols: list[Chem.Mol] = []
    for cluster in clusters:
        unique_in_cluster = _filter_within_cluster(cluster, rmsd_threshold)
        unique_mols.extend(unique_in_cluster)

    n_duplicates = len(valid_mols) - len(unique_mols)
    if n_duplicates:
        dropped["duplicate"] = n_duplicates
    return FilterResult(kept=unique_mols, dropped=dropped)


def filter_unique_optimized(
    mols: list[Chem.Mol],
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
) -> list[Chem.Mol]:
    """The surviving molecules from :func:`filter_conformers`, nothing else.

    Kept as the public name it has always been, for callers that want the list
    and not the drop counts. Prefer :func:`filter_conformers` when the reason
    something is missing has to reach a user.

    Returns:
        List of unique molecules, sorted by energy (lowest first); records with
        no usable energy sort last.
    """
    return filter_conformers(
        mols,
        rmsd_threshold=rmsd_threshold,
        energy_cluster_window=energy_cluster_window,
    ).kept


def _filter_within_cluster(
    mols: list[Chem.Mol],
    rmsd_threshold: float,
    energy_tol: float = DEFAULT_DUPLICATE_ENERGY_TOL,
) -> list[Chem.Mol]:
    """Filter unique molecules within an energy cluster.

    Args:
        mols: List of RDKit Mol objects to filter.
        rmsd_threshold: RMSD threshold for considering structures similar (Angstrom).
        energy_tol: Energy tolerance (eV). A pair is treated as duplicate only
            when the two are the same compound AND the heavy-atom RMSD is below
            ``rmsd_threshold`` AND the energies agree within this tolerance. The
            energy term preserves conformers that differ only in an O-H / N-H
            rotor orientation (RMSD~=0 on heavy atoms but distinct minima with
            different energies); the compound term preserves distinct
            stereoisomers, which arrive here in one group and whose heavy-atom
            RMSD can fall below the default threshold
            (:func:`Auto3D.utils.stereo_check.species_key`).

    Returns:
        List of unique molecules within the cluster.
    """
    if len(mols) <= 1:
        return list(mols)

    # Strip Hs once per molecule (O(n)), not once per comparison (O(n^2)).
    # GetBestRMS on the no-H forms is symmetric, so results are unchanged. The
    # ORIGINAL (H-explicit) mols are returned; no-H forms are comparison-only.
    unique: list[Chem.Mol] = []
    unique_noH: list[Chem.Mol] = []
    unique_energies: list[float] = []
    unique_species: list[str] = []
    for mol_i in mols:
        mol_i_noH = Chem.RemoveHs(mol_i)
        e_i = _mol_energy(mol_i)
        species_i = species_key(mol_i)
        is_unique = True

        for mol_j_noH, e_j, species_j in zip(
            unique_noH, unique_energies, unique_species, strict=True
        ):
            # Two different compounds are never duplicates of each other, however
            # close their geometries. Checked first, and before the RMSD call it
            # makes unnecessary.
            if species_i != species_j:
                continue
            try:
                # Temporary bug fix for https://github.com/rdkit/rdkit/issues/6826
                # Removing Hs speeds up the calculation
                rmsd = rdMolAlign.GetBestRMS(mol_i_noH, mol_j_noH)
            except RuntimeError:
                rmsd = float("inf")  # incomparable pair -> treat as distinct

            # Heavy-atom RMSD alone collapses distinct H-rotamers; require the
            # energies to also agree before declaring a duplicate. Missing/NaN
            # energy => fall back to RMSD-only (energy guard cannot apply).
            energy_close = (
                e_i is None or e_j is None or abs(e_i - e_j) < energy_tol
            )
            if rmsd < rmsd_threshold and energy_close:
                is_unique = False
                break

        if is_unique:
            unique.append(mol_i)
            unique_noH.append(mol_i_noH)
            unique_energies.append(e_i)
            unique_species.append(species_i)

    return unique


def _mol_energy(mol: Chem.Mol) -> float | None:
    """Return a mol's optimized energy in eV, or None.

    ``E_tot`` is stored in Hartree; the conversion lives in
    ``Auto3D.utils.energy`` so this module and ``ranking`` cannot drift apart on
    it. ``None`` signals "no usable energy" so callers fall back to RMSD-only
    comparison.
    """
    return try_e_tot_ev(mol)

