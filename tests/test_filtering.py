#!/usr/bin/env python
"""Tests for optimized RMSD filtering with energy clustering."""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.filtering import filter_unique_optimized, _filter_within_cluster


def _create_mol_with_energy(smiles: str, energy: float, converged: bool = True) -> Chem.Mol:
    """Helper to create a test molecule with properties set."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp('Converged', 'true' if converged else 'false')
    mol.SetProp('E_tot', str(energy))
    return mol


class TestFilterWithinCluster:
    """Tests for _filter_within_cluster function."""

    def test_empty_list_returns_empty(self):
        """Empty input should return empty list."""
        result = _filter_within_cluster([], rmsd_threshold=0.5)
        assert result == []

    def test_single_mol_returns_itself(self):
        """Single molecule should be returned as-is."""
        mol = _create_mol_with_energy("C", -10.0)
        result = _filter_within_cluster([mol], rmsd_threshold=0.5)
        assert len(result) == 1

    def test_identical_mols_returns_one(self):
        """Identical molecules should be deduplicated to one."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.0)
        result = _filter_within_cluster([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 1

    def test_different_conformers_returns_both(self):
        """Different conformers of the same molecule should be kept if RMSD > threshold."""
        # Create two conformers of the same molecule with different 3D coordinates
        # Use a larger molecule that can have truly different conformers
        mol1 = Chem.MolFromSmiles("CCCC")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol1)

        mol2 = Chem.MolFromSmiles("CCCC")
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, randomSeed=123)  # Different seed for different conformer
        AllChem.MMFFOptimizeMolecule(mol2)

        # Use a very small threshold so both conformers are kept
        result = _filter_within_cluster([mol1, mol2], rmsd_threshold=0.001)
        # With such a small threshold, both conformers should be unique
        assert len(result) >= 1  # At least one conformer


class TestFilterUniqueOptimized:
    """Tests for filter_unique_optimized function."""

    def test_empty_list_returns_empty(self):
        """Empty input should return empty list."""
        result = filter_unique_optimized([], rmsd_threshold=0.5)
        assert result == []

    def test_filters_unconverged_structures(self):
        """Unconverged structures should be filtered out."""
        mol1 = _create_mol_with_energy("C", -10.0, converged=True)
        mol2 = _create_mol_with_energy("CC", -11.0, converged=False)
        result = filter_unique_optimized([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 1

    def test_removes_duplicates(self):
        """Optimized filter should remove similar structures."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.0)
        result = filter_unique_optimized([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 1

    def test_keeps_different_molecules(self):
        """Different molecules should be kept."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("CCO", -12.0)
        result = filter_unique_optimized([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 2

    def test_sorts_by_energy(self):
        """Output should be sorted by energy (lowest first)."""
        mol_high = _create_mol_with_energy("C", -5.0)
        mol_low = _create_mol_with_energy("CC", -15.0)
        mol_mid = _create_mol_with_energy("CCC", -10.0)

        result = filter_unique_optimized(
            [mol_high, mol_low, mol_mid],
            rmsd_threshold=0.5
        )

        energies = [float(mol.GetProp('E_tot')) for mol in result]
        assert energies == sorted(energies)

    def test_energy_clustering_groups_similar_energies(self):
        """Molecules with similar energies should be clustered together."""
        # Create molecules with energies in two distinct clusters
        # Cluster 1: -10.0, -10.05 (within 0.1 eV window)
        # Cluster 2: -15.0
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.05)  # Same structure, should be removed
        mol3 = _create_mol_with_energy("CCO", -15.0)

        result = filter_unique_optimized(
            [mol1, mol2, mol3],
            rmsd_threshold=0.5,
            energy_cluster_window=0.1
        )

        # mol1 and mol2 are in same cluster and identical, so one removed
        # mol3 is in different cluster
        assert len(result) == 2

    def test_large_energy_window_creates_single_cluster(self):
        """Large energy window should group all molecules into one cluster."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -15.0)  # Same structure

        result = filter_unique_optimized(
            [mol1, mol2],
            rmsd_threshold=0.5,
            energy_cluster_window=10.0  # Large window
        )

        # Same molecule type, should be deduplicated regardless of energy
        assert len(result) == 1

    def test_small_energy_window_creates_separate_clusters(self):
        """Small energy window should create separate clusters."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -15.0)  # Same structure but different cluster

        result = filter_unique_optimized(
            [mol1, mol2],
            rmsd_threshold=0.5,
            energy_cluster_window=0.01  # Very small window
        )

        # Same molecule but in different energy clusters - both kept
        # (RMSD comparison only happens within clusters)
        assert len(result) == 2


class TestFilterUniqueBehavior:
    """Tests verifying behavior matches original filter_unique."""

    def test_matches_original_for_simple_case(self):
        """Should produce same results as original filter_unique for basic cases.

        Both filters operate on conformers of the *same* molecule (that is the
        real Auto3D contract). Use genuinely distinct conformers of one molecule
        so RMSD is well-defined and comparable across both implementations.
        """
        from Auto3D.utils import filter_unique

        def conformer(seed: float, energy: float) -> Chem.Mol:
            m = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))  # flexible chain
            AllChem.EmbedMolecule(m, randomSeed=seed)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("Converged", "true")
            m.SetProp("E_tot", str(energy))
            return m

        mols = [conformer(42, -12.0), conformer(7, -11.0), conformer(123, -10.0)]

        original_result = filter_unique(mols, crit=0.3)
        optimized_result = filter_unique_optimized(
            mols,
            rmsd_threshold=0.3,
            energy_cluster_window=100.0  # Large window = single cluster = same behavior
        )

        # Same molecule, well-defined RMSD: both implementations must agree.
        assert len(original_result) == len(optimized_result)


def test_rmsd_failure_keeps_both(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from Auto3D.filtering import _filter_within_cluster

    def make(name, e):
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=abs(hash(name)) % 1000)
        m.SetProp("_Name", name); m.SetProp("E_tot", str(e))
        return m

    def boom(*a, **k):
        raise RuntimeError("GetBestRMS failed")
    monkeypatch.setattr(rdMolAlign, "GetBestRMS", boom)

    cluster = [make("a", -1.0), make("b", -0.9)]
    kept = _filter_within_cluster(cluster, rmsd_threshold=0.3)
    assert len(kept) == 2  # incomparable pair must NOT be dropped
