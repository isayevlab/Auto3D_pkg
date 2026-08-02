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
from Auto3D.utils import check_connectivity
from Auto3D.utils.convergence import converged_or_unfiltered
from Auto3D.utils.energy import e_tot_ev, try_e_tot_ev
from Auto3D.utils.stereo_check import stereo_preserved


def filter_unique_optimized(
    mols: list[Chem.Mol],
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    energy_cluster_window: float = DEFAULT_ENERGY_CLUSTER_WINDOW,
) -> list[Chem.Mol]:
    """Filter unique conformers with optimized O(n log n) approach.

    Uses energy-based clustering to reduce RMSD comparisons:
    1. Sort by energy
    2. Group into energy clusters
    3. Only compare within clusters

    This approach reduces the complexity from O(n^2) to approximately
    O(n * k) where k is the average cluster size, which is typically
    much smaller than n for molecules with diverse energies.

    Args:
        mols: List of RDKit Mol objects with 'E_tot' (Hartree) and, optionally,
            'Converged' properties. A record whose 'Converged' property is
            explicitly false is dropped; a record without the property is kept
            (it is not filtered on convergence). Records marked
            'Stereo_changed' are excluded.
        rmsd_threshold: RMSD threshold for considering structures similar (Angstrom).
        energy_cluster_window: Energy window for clustering (eV). Unchanged:
            the stored Hartree energies are converted to eV on read
            (``Auto3D.utils.energy.e_tot_ev``), so this parameter keeps its
            documented unit.

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

    # Cluster by energy
    clusters: list[list[Chem.Mol]] = []
    current_cluster: list[Chem.Mol] = [valid_mols[0]]
    current_min_e = e_tot_ev(valid_mols[0])

    for mol in valid_mols[1:]:
        e = e_tot_ev(mol)
        if e - current_min_e <= energy_cluster_window:
            current_cluster.append(mol)
        else:
            clusters.append(current_cluster)
            current_cluster = [mol]
            current_min_e = e
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
            when the heavy-atom RMSD is below ``rmsd_threshold`` AND the energies
            agree within this tolerance, so conformers that differ only in an
            O-H / N-H rotor orientation (RMSD~=0 on heavy atoms but distinct
            minima with different energies) are preserved.

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
    for mol_i in mols:
        mol_i_noH = Chem.RemoveHs(mol_i)
        e_i = _mol_energy(mol_i)
        is_unique = True

        for mol_j_noH, e_j in zip(unique_noH, unique_energies, strict=True):
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

    return unique


def _mol_energy(mol: Chem.Mol) -> float | None:
    """Return a mol's optimized energy in eV, or None.

    ``E_tot`` is stored in Hartree; the conversion lives in
    ``Auto3D.utils.energy`` so this module and ``ranking``/``utils.chemistry``
    cannot drift apart on it. ``None`` signals "no usable energy" so callers
    fall back to RMSD-only comparison.
    """
    return try_e_tot_ev(mol)
