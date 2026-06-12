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
        mols: List of RDKit Mol objects with 'E_tot' and 'Converged' properties.
        rmsd_threshold: RMSD threshold for considering structures similar (Angstrom).
        energy_cluster_window: Energy window for clustering (eV).

    Returns:
        List of unique molecules, sorted by energy (lowest first).
    """
    # Filter converged structures with valid connectivity
    valid_mols = []
    for mol in mols:
        if mol is None:
            continue
        try:
            converged = mol.GetProp('Converged').lower() == 'true'
        except KeyError:
            converged = False
        if converged and check_connectivity(mol):
            valid_mols.append(mol)

    if not valid_mols:
        return []

    # Sort by energy
    valid_mols.sort(key=lambda m: float(m.GetProp('E_tot')))

    # Cluster by energy
    clusters: list[list[Chem.Mol]] = []
    current_cluster: list[Chem.Mol] = [valid_mols[0]]
    current_min_e = float(valid_mols[0].GetProp('E_tot'))

    for mol in valid_mols[1:]:
        e = float(mol.GetProp('E_tot'))
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

        for mol_j_noH, e_j in zip(unique_noH, unique_energies):
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
    """Return a mol's optimized energy (eV) from the ``E_tot`` property, or None.

    Used by the duplicate-conformer test as an energy guard; ``None`` signals
    "no usable energy" so callers fall back to RMSD-only comparison.
    """
    try:
        return float(mol.GetProp("E_tot"))
    except (KeyError, ValueError):
        return None
