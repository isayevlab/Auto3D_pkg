#!/usr/bin/env python
"""Optimized conformer filtering with hierarchical RMSD comparison."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from Auto3D.constants import (
    DEFAULT_DUPLICATE_ENERGY_TOL,
    DEFAULT_ENERGY_CLUSTER_WINDOW,
    DEFAULT_RMSD_THRESHOLD,
)
from Auto3D.utils.connectivity import check_connectivity
from Auto3D.utils.convergence import converged_or_unfiltered
from Auto3D.utils.energy import e_tot_ev, try_e_tot_ev
from Auto3D.utils.stereo_check import species_key, stereo_preserved


def filter_unique_optimized(
    mols: list[Chem.Mol],
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
) -> list[Chem.Mol]:
    """Remove duplicate conformers, skipping RMSD comparisons that cannot match.

    Sorts by energy and only RMSD-compares molecules close enough in energy to be
    duplicates at all, which avoids the O(n^2) comparisons of the legacy
    :func:`filter_unique` below **without changing which molecules
    survive** -- the partitioning is chosen so that no duplicate pair can be
    separated by it. See the comment on the split rule below for why that
    holds; it did not hold before 4.0.0.

    A pair counts as a duplicate only when all three of these agree: the two are
    the same compound (:func:`Auto3D.utils.stereo_check.species_key`), their
    heavy-atom RMSD is under ``rmsd_threshold``, and their energies agree within
    ``DEFAULT_DUPLICATE_ENERGY_TOL``.

    Args:
        mols: List of RDKit Mol objects with 'E_tot' (Hartree) and, optionally,
            'Converged' properties. A record whose 'Converged' property is
            explicitly false is dropped; a record without the property is kept
            (it is not filtered on convergence). Records marked
            'Stereo_changed' are excluded.
        rmsd_threshold: RMSD threshold for considering structures similar (Angstrom).
        energy_cluster_window: Energy width (eV) below which molecules are
            compared to each other. A performance knob only: values below the
            duplicate energy tolerance cannot shrink the comparison set, because
            that tolerance is the floor at which a pair can still be a duplicate.
            The stored Hartree energies are converted to eV on read
            (``Auto3D.utils.energy.e_tot_ev``), so the unit is as documented.

    Returns:
        List of unique molecules, sorted by energy (lowest first).
    """
    # Filter converged structures with valid connectivity. A record with no
    # 'Converged' property is not filtered on convergence (see
    # Auto3D.utils.convergence): treating its absence as failure deleted every
    # record of any SDF batchopt did not write.
    valid_mols = []
    for mol in mols:
        if mol is None:
            continue
        if (
            converged_or_unfiltered(mol)
            and stereo_preserved(mol)
            and check_connectivity(mol)
        ):
            valid_mols.append(mol)

    if not valid_mols:
        return []

    # Sort by energy. E_tot is stored in Hartree; energy_cluster_window and
    # the duplicate tolerance below are both in eV, so convert on read.
    valid_mols.sort(key=e_tot_ev)

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
    # ladder degrades to the O(n^2) of the legacy `filter_unique` -- the price of
    # the guarantee, on conformer counts that are tens per species.
    split_gap = max(DEFAULT_DUPLICATE_ENERGY_TOL, energy_cluster_window)
    clusters: list[list[Chem.Mol]] = []
    current_cluster: list[Chem.Mol] = [valid_mols[0]]
    previous_e = e_tot_ev(valid_mols[0])

    for mol in valid_mols[1:]:
        e = e_tot_ev(mol)
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

    return unique_mols


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


def filter_unique(mols: list[Chem.Mol], crit: float = DEFAULT_RMSD_THRESHOLD) -> list[Chem.Mol]:
    """Remove structures that are very similar and remove unconverged structures.

    The legacy all-pairs filter, kept beside :func:`filter_unique_optimized`
    (which supersedes it) because ``ConformerRanker(use_optimized_filtering=False)``
    still selects it and because it is the oracle the optimized filter is
    compared against.

    This function filters a list of molecules to keep only unique, converged structures.
    It first removes unconverged structures and those with invalid connectivity,
    then removes similar structures based on RMSD comparison.

    Args:
        mols: List of RDKit molecule objects, optionally carrying a 'Converged'
            property. A record whose 'Converged' is explicitly false is
            dropped; a record without the property is kept (not filtered on
            convergence). Records marked 'Stereo_changed' are excluded.
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
    # Remove structures that explicitly failed to converge. A record with no
    # 'Converged' property is NOT filtered on convergence -- see
    # Auto3D.utils.convergence for why absence is not failure.
    mols_: list[Chem.Mol] = []
    for mol in mols:
        convergence_flag = converged_or_unfiltered(mol)
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
    # DEFAULT_DUPLICATE_ENERGY_TOL (eV; 'E_tot' is stored in Hartree and is
    # converted on read by Auto3D.utils.energy). Mols without a usable 'E_tot'
    # fall back to RMSD-only (energy guard cannot apply).
    unique_mols: list[Chem.Mol] = []
    unique_noH: list[Chem.Mol] = []
    unique_energies: list[float | None] = []
    unique_species: list[str] = []
    for mol_i in mols:
        mol_i_noH = Chem.RemoveHs(mol_i)
        # E_tot is stored in Hartree; DEFAULT_DUPLICATE_ENERGY_TOL is in eV.
        e_i: float | None = try_e_tot_ev(mol_i)
        species_i = species_key(mol_i)
        unique = True
        for mol_j_noH, e_j, species_j in zip(
            unique_noH, unique_energies, unique_species, strict=True
        ):
            # Two different compounds are never duplicates of each other, however
            # close their geometries. All stereoisomers of one input share a
            # ranking group, and two ring diastereomers can sit below the default
            # 0.3 A threshold, so without this the RMSD test could delete one of
            # them (Auto3D.utils.stereo_check.species_key). Checked before the
            # RMSD call it makes unnecessary.
            if species_i != species_j:
                continue
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
            unique_species.append(species_i)
    return unique_mols
