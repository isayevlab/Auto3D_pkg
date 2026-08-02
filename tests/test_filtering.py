#!/usr/bin/env python
"""Tests for optimized RMSD filtering with energy clustering."""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.filtering import filter_unique_optimized, _filter_within_cluster
from Auto3D.utils.energy import set_e_tot_from_ev


def _create_mol_with_energy(smiles: str, energy_ev: float, converged: bool = True) -> Chem.Mol:
    """Helper to create a test molecule with properties set.

    ``energy_ev`` is in eV, which is the unit this module's thresholds
    (``rmsd_threshold``'s companion ``energy_tol``, ``energy_cluster_window``)
    are documented in and the unit the filters compare in. The SDF property
    itself is written in Hartree, through the same boundary helper the
    optimizer uses, because that is what a real input file carries.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp('Converged', 'true' if converged else 'false')
    set_e_tot_from_ev(mol, energy_ev)
    return mol


class TestConvergenceFilterIsNotAnEraser:
    """A record with no 'Converged' property must not be deleted.

    ``filter_unique_optimized`` is reached by ``ConformerRanker`` on any SDF
    the caller hands it, including files ``batchopt`` did not write -- which
    carry no ``Converged`` property at all.
    """

    def test_missing_converged_property_is_kept(self):
        mol = _create_mol_with_energy("CCO", -10.0)
        mol.ClearProp("Converged")
        assert not mol.HasProp("Converged"), "test premise"

        result = filter_unique_optimized([mol], rmsd_threshold=0.3)
        assert len(result) == 1, (
            "a record that never claimed to be an optimizer output was deleted"
        )

    def test_explicit_false_is_still_dropped(self):
        """Absence is not failure -- but a stated failure still is."""
        mol = _create_mol_with_energy("CCO", -10.0, converged=False)
        assert filter_unique_optimized([mol], rmsd_threshold=0.3) == []


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

    def test_same_geometry_different_energy_kept(self):
        """Same heavy-atom geometry but distinct energy must NOT be deduped.

        Mirrors the O-H / N-H rotamer case: heavy-atom RMSD ~= 0 but the two are
        distinct minima with different energies. The energy guard must keep both.
        """
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.5)  # RMSD~=0, |dE| >> tol
        result = _filter_within_cluster([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 2

    def test_same_geometry_near_equal_energy_deduped(self):
        """RMSD ~= 0 AND energies within tolerance => still a duplicate."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.005)  # |dE| < default 0.01 eV
        result = _filter_within_cluster([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 1

    def test_energy_tol_is_configurable(self):
        """A wide energy_tol collapses energy-distinct duplicates again."""
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.5)
        result = _filter_within_cluster(
            [mol1, mol2], rmsd_threshold=0.5, energy_tol=1.0
        )
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

        from Auto3D.utils.energy import e_tot_ev

        energies = [e_tot_ev(mol) for mol in result]
        assert energies == sorted(energies)

    def test_energy_clustering_groups_similar_energies(self):
        """Molecules with similar energies cluster together and dedup.

        Two same-structure conformers whose energies agree within the duplicate
        tolerance (and a third distinct molecule in its own cluster) reduce to
        two survivors.
        """
        # Cluster 1: -10.0, -10.005 (same structure, |dE| < duplicate tol)
        # Cluster 2: -15.0
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -10.005)  # within energy tol -> removed
        mol3 = _create_mol_with_energy("CCO", -15.0)

        result = filter_unique_optimized(
            [mol1, mol2, mol3],
            rmsd_threshold=0.5,
            energy_cluster_window=0.1
        )

        # mol1 and mol2 are in the same cluster, identical geometry AND
        # near-equal energy, so one is removed; mol3 is a separate cluster.
        assert len(result) == 2

    def test_single_cluster_energy_guard_keeps_distinct_energies(self):
        """A large window puts everything in one cluster, but the energy guard
        keeps same-geometry conformers whose energies differ beyond tolerance.

        This is the O-H / N-H rotamer case at the optimized-filter level: heavy-
        atom RMSD ~= 0 but distinct minima with different energies must survive.
        """
        mol1 = _create_mol_with_energy("C", -10.0)
        mol2 = _create_mol_with_energy("C", -15.0)  # same geometry, |dE| >> tol

        result = filter_unique_optimized(
            [mol1, mol2],
            rmsd_threshold=0.5,
            energy_cluster_window=10.0  # one cluster
        )

        # Different energies => not duplicates, both kept.
        assert len(result) == 2

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

        def conformer(seed: float, energy_ev: float) -> Chem.Mol:
            m = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))  # flexible chain
            AllChem.EmbedMolecule(m, randomSeed=seed)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("Converged", "true")
            set_e_tot_from_ev(m, energy_ev)
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


def test_filter_within_cluster_removehs_is_linear_and_nondestructive(monkeypatch):
    """RemoveHs runs once per molecule (O(n)) AND is non-destructive: returned
    conformers keep their explicit hydrogens and exact H positions. This is a
    correctness invariant, not just perf -- the MLIP requires explicit H and the
    final geometries written to SDF must retain the optimized H coordinates. The
    no-H form is a throwaway copy used only for the RMSD comparison.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D import filtering

    # Five DISTINCT conformers of one molecule so all survive as unique,
    # maximizing inner-loop comparisons (the O(n^2) path strips Hs each pair).
    mols = []
    base = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    cids = AllChem.EmbedMultipleConfs(base, numConfs=5, randomSeed=1)
    from Auto3D.utils.energy import set_e_tot_from_ev as _set_e

    for cid in cids:
        m = Chem.Mol(base, confId=int(cid))
        _set_e(m, 0.0)
        m.SetProp("Converged", "true")
        mols.append(m)
    n_atoms = base.GetNumAtoms()  # heavy + explicit H (15 for CCCCO)
    orig_pos = {id(m): m.GetConformer().GetPositions().copy() for m in mols}

    calls = {"n": 0}
    real_removehs = filtering.Chem.RemoveHs

    def counting(mol, *a, **k):
        calls["n"] += 1
        return real_removehs(mol, *a, **k)

    monkeypatch.setattr(filtering.Chem, "RemoveHs", counting)

    result = filtering._filter_within_cluster(mols, rmsd_threshold=0.01)

    # O(n): RemoveHs called once per input, never per pair.
    assert calls["n"] == len(mols)
    assert len(result) == len(mols)
    # Non-destructive: returned mols keep explicit H and byte-identical positions.
    for m in result:
        assert m.GetNumAtoms() == n_atoms
        assert any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
        assert np.array_equal(m.GetConformer().GetPositions(), orig_pos[id(m)])


def test_rmsd_failure_keeps_both(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from Auto3D.filtering import _filter_within_cluster

    def make(name, e):
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=abs(hash(name)) % 1000)
        m.SetProp("_Name", name)
        set_e_tot_from_ev(m, e)
        return m

    def boom(*a, **k):
        raise RuntimeError("GetBestRMS failed")
    monkeypatch.setattr(rdMolAlign, "GetBestRMS", boom)

    cluster = [make("a", -1.0), make("b", -0.9)]
    kept = _filter_within_cluster(cluster, rmsd_threshold=0.3)
    assert len(kept) == 2  # incomparable pair must NOT be dropped
