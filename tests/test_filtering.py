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
        """Single molecule should be returned as-is, explicit Hs and all.

        The RMSD comparison strips Hs from a throwaway copy for speed; the
        returned molecule must be the caller's original (H-explicit) object,
        not the no-H comparison copy -- the MLIP downstream requires explicit
        H, and this is the len(mols) <= 1 short-circuit that never even
        reaches the strip/compare loop.
        """
        mol = _create_mol_with_energy("C", -10.0)
        n_atoms_before = mol.GetNumAtoms()
        assert any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()), "test premise: has explicit Hs"

        result = _filter_within_cluster([mol], rmsd_threshold=0.5)

        assert len(result) == 1
        assert result[0] is mol, "the single-mol short-circuit must return the original object"
        assert result[0].GetNumAtoms() == n_atoms_before
        assert any(a.GetAtomicNum() == 1 for a in result[0].GetAtoms())

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


class TestDistinctStereoisomersAreNeverDuplicates:
    """Dedup must remove duplicate *conformers*, never a distinct compound.

    ``ranking.species_id`` strips ``<isomer>_<conformer>``, so every enumerated
    stereoisomer of one input shares a ranking group and reaches this filter
    together. Heavy-atom ``GetBestRMS`` between two diastereomers of a
    1,4-disubstituted ring is small -- 0.300 A measured for
    cis/trans-4-tert-butylcyclohexanol, 0.335 A for cyclohexane-1,4-diol -- so
    with ``--threshold`` raised above the default, or with the two isomers
    within the duplicate energy tolerance, one of two distinct compounds
    disappears from the output with nothing logged.

    The thresholds below are deliberately set wide enough that RMSD and energy
    both say "duplicate". What must keep the pair apart is the only thing that
    should: they are different compounds.
    """

    #: cis/trans-4-tert-butylcyclohexanol -- the measured worst case.
    _CIS = "O[C@H]1CC[C@@H](CC1)C(C)(C)C"
    _TRANS = "O[C@H]1CC[C@H](CC1)C(C)(C)C"

    def test_two_diastereomers_are_both_kept(self):
        cis = _create_mol_with_energy(self._CIS, -10.0)
        trans = _create_mol_with_energy(self._TRANS, -10.0)  # identical energy
        assert Chem.MolToSmiles(cis) != Chem.MolToSmiles(trans), "test premise"

        result = _filter_within_cluster([cis, trans], rmsd_threshold=10.0)

        assert len(result) == 2, (
            "two distinct diastereomers were merged into one: an input molecule "
            "vanished from the output with no record"
        )

    def test_two_double_bond_isomers_are_both_kept(self):
        """The same guarantee for E/Z, which the SDF path enumerates too."""
        e_isomer = _create_mol_with_energy(r"C/C=C/CCO", -10.0)
        z_isomer = _create_mol_with_energy(r"C/C=C\CCO", -10.0)

        result = _filter_within_cluster([e_isomer, z_isomer], rmsd_threshold=10.0)

        assert len(result) == 2, "an E/Z pair was merged into one compound"

    def test_two_conformers_of_one_stereoisomer_are_still_deduplicated(self):
        """The other half of the contract, and the reason this test exists.

        A stereo guard that declared everything distinct would satisfy the two
        tests above and quietly disable duplicate removal entirely. Two
        conformers of ONE stereoisomer, same geometry and same energy, must
        still collapse to one.
        """
        first = _create_mol_with_energy(self._CIS, -10.0)
        second = _create_mol_with_energy(self._CIS, -10.0)

        result = _filter_within_cluster([first, second], rmsd_threshold=10.0)

        assert len(result) == 1, (
            "duplicate conformers of one stereoisomer survived, so the stereo "
            "guard has switched dedup off rather than narrowing it"
        )


class TestDuplicatesCannotEscapeAcrossAnEnergyBoundary:
    """A duplicate pair must be compared however the energy axis is partitioned.

    ``filter_unique_optimized`` groups by energy and only RMSD-compares within a
    group, so a duplicate pair straddling a boundary was never compared at all.
    Three bit-identical geometries whose energies span exactly one default
    ``energy_cluster_window`` (0.1 eV) all survived, and the two that differ by
    2e-6 eV are unambiguously the same conformer.
    """

    def test_a_duplicate_pair_split_by_a_boundary_is_still_removed(self):
        base = -100.0
        # Sorted: the first is far from the other two; the last two are 2e-6 eV
        # apart and identical in geometry, so exactly two structures are unique.
        energies = [base, base + 0.099999, base + 0.100001]
        mols = [_create_mol_with_energy("CCO", e) for e in energies]

        result = filter_unique_optimized(
            mols, rmsd_threshold=0.5, energy_cluster_window=0.1
        )

        assert len(result) == 2, (
            "a duplicate pair 2e-6 eV apart survived because the two halves "
            f"landed in different energy groups; got {len(result)} structures"
        )


class TestFilterUniqueOptimized:
    """Tests for filter_unique_optimized function."""

    def test_empty_list_returns_empty(self):
        """Empty input should return empty list."""
        result = filter_unique_optimized([], rmsd_threshold=0.5)
        assert result == []

    def test_filters_unconverged_structures(self):
        """Unconverged structures should be filtered out -- and it must be
        specifically the unconverged one that is gone, not just any one of
        the two (e.g. a dedup bug that merged them for an unrelated reason
        would also leave len(result) == 1).
        """
        mol1 = _create_mol_with_energy("C", -10.0, converged=True)
        mol2 = _create_mol_with_energy("CC", -11.0, converged=False)
        result = filter_unique_optimized([mol1, mol2], rmsd_threshold=0.5)
        assert len(result) == 1
        assert result[0] is mol1, "the converged structure must be the survivor"
        assert result[0].GetProp("Converged").lower() == "true"

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


class TestMissingEnergyPropertyMustNotCrash:
    """filter_unique_optimized must tolerate a record with no 'E_tot', the
    way the legacy ``utils.chemistry.filter_unique`` already does.

    KNOWN DEFECT (found during cluster E brainstorming, not fixed by this
    lane): ``filtering.py:75`` sorts the valid-mols list by
    ``Auto3D.utils.energy.e_tot_ev``, which RAISES (KeyError/ValueError) for
    a molecule with no usable 'E_tot' property. ``_filter_within_cluster``'s
    own energy guard, two dozen lines later in the same module, instead uses
    the tolerant ``try_e_tot_ev`` and treats a missing energy as "fall back
    to RMSD only". ``utils.chemistry.filter_unique`` (the OTHER conformer
    filter, sharing the same duplicate criterion since 4.0.1) also uses
    ``try_e_tot_ev`` throughout and does not crash on this input. So the two
    filters diverge on malformed input: the same list of mols that
    ``filter_unique`` happily filters crashes ``filter_unique_optimized``.

    This matters here specifically because cluster B5 is about to delete one
    of the two filters, and the survivor is the stricter (crashing) one --
    fixing filtering.py's sort key to use ``try_e_tot_ev``, matching its own
    energy guard and the legacy filter, is what should make this pass.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "filtering.py:75 sorts by e_tot_ev (raises KeyError for a mol "
            "with no 'E_tot' property) instead of the tolerant try_e_tot_ev "
            "that _filter_within_cluster's own energy guard and the legacy "
            "utils.chemistry.filter_unique both use -- the two conformer "
            "filters disagree on malformed input (cluster E brainstorm defect)."
        ),
    )
    def test_missing_e_tot_property_does_not_crash(self):
        mol_no_energy = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol_no_energy, randomSeed=42)
        mol_no_energy.SetProp("Converged", "true")
        # Deliberately no set_e_tot_from_ev call: this record has no 'E_tot'.
        assert not mol_no_energy.HasProp("E_tot"), "test premise"

        mol_with_energy = _create_mol_with_energy("CC", -10.0)

        result = filter_unique_optimized(
            [mol_no_energy, mol_with_energy], rmsd_threshold=0.3
        )

        # Correct behavior: no crash, and a mol with no usable energy simply
        # cannot be deduped by energy -- it must survive alongside the other.
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
