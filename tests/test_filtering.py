#!/usr/bin/env python
"""Tests for the single conformer filter (RMSD dedup with energy clustering)."""

from __future__ import annotations

import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.domain.filtering import (
    DROP_REASONS,
    FilterResult,
    _filter_within_cluster,
    filter_conformers,
    filter_unique_optimized,
)
from Auto3D.foundation.utils.energy import set_e_tot_from_ev


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
    mol.SetProp("Converged", "true" if converged else "false")
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
        assert len(result) == 1, "a record that never claimed to be an optimizer output was deleted"

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
        result = _filter_within_cluster([mol1, mol2], rmsd_threshold=0.5, energy_tol=1.0)
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

        result = filter_unique_optimized(mols, rmsd_threshold=0.5, energy_cluster_window=0.1)

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

        result = filter_unique_optimized([mol_high, mol_low, mol_mid], rmsd_threshold=0.5)

        from Auto3D.foundation.utils.energy import e_tot_ev

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
            [mol1, mol2, mol3], rmsd_threshold=0.5, energy_cluster_window=0.1
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
            energy_cluster_window=10.0,  # one cluster
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
            energy_cluster_window=0.01,  # Very small window
        )

        # Same molecule but in different energy clusters - both kept
        # (RMSD comparison only happens within clusters)
        assert len(result) == 2


def _energyless(smiles: str, seed: int = 42, name: str = "") -> Chem.Mol:
    """An embedded, converged conformer carrying NO 'E_tot' property.

    This is what an SDF Auto3D's optimizer did not write can look like: a
    hand-built conformer set, or an export that names its energy field
    something else. ``ConformerRanker`` refuses such a record up front
    (``InputValidationError``), so this shape only reaches the filter through
    a direct API call -- which is exactly the caller the two filters used to
    disagree for.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("Converged", "true")
    if name:
        mol.SetProp("_Name", name)
    assert not mol.HasProp("E_tot"), "helper premise: no energy property"
    return mol


class TestMissingEnergyPropertyMustNotCrash:
    """The one conformer filter must tolerate a record with no 'E_tot'.

    ``filtering.py`` used to sort the valid-mols list by
    ``Auto3D.foundation.utils.energy.e_tot_ev``, which RAISES (KeyError/ValueError) for a
    molecule with no usable 'E_tot'. ``_filter_within_cluster``'s own energy
    guard, two dozen lines later in the same module, instead used the tolerant
    ``try_e_tot_ev`` and treated a missing energy as "fall back to RMSD only",
    as did the legacy all-pairs ``filter_unique`` throughout. So the same list
    of mols one filter happily filtered crashed the other -- and the survivor
    of the two was the crashing one.

    Tolerating the record must not mean *inventing* an energy for it: a
    missing 'E_tot' read as 0.0 would sort a garbage record to the front of a
    list of negative energies and hand it to ``top_k`` as the global minimum.
    """

    def test_missing_e_tot_property_does_not_crash(self):
        mol_no_energy = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol_no_energy, randomSeed=42)
        mol_no_energy.SetProp("Converged", "true")
        # Deliberately no set_e_tot_from_ev call: this record has no 'E_tot'.
        assert not mol_no_energy.HasProp("E_tot"), "test premise"

        mol_with_energy = _create_mol_with_energy("CC", -10.0)

        result = filter_unique_optimized([mol_no_energy, mol_with_energy], rmsd_threshold=0.3)

        # Correct behavior: no crash, and a mol with no usable energy simply
        # cannot be deduped by energy -- it must survive alongside the other.
        assert len(result) == 2

    def test_a_garbled_e_tot_is_tolerated_the_same_way(self):
        """``try_e_tot_ev`` swallows ValueError too, so a non-numeric property
        must take the same path as an absent one rather than crash."""
        mol_garbled = _energyless("CCO", seed=42)
        mol_garbled.SetProp("E_tot", "not-a-number")

        mol_with_energy = _create_mol_with_energy("CC", -10.0)

        result = filter_unique_optimized([mol_garbled, mol_with_energy], rmsd_threshold=0.3)
        assert len(result) == 2

    def test_an_energyless_record_sorts_last_whatever_the_real_energies_are(self):
        """A tolerated missing energy must not be *compared* as an energy.

        The tolerant path needs some placeholder to sort on; the danger is that
        the placeholder takes part in the comparison. A record with no ``E_tot``
        that sorts as if its energy were 0.0 lands ahead of every genuine
        structure whose energy is above 0.0 -- and the first element of the
        filter's output is what ``top_k``/``top_window`` treat as the global
        minimum: the reference ``E_rel`` is measured from, and the single
        structure a ``k=1`` request returns.

        ``E_tot`` is a user-supplied SD property, so "every real energy is a
        large negative number" is a convention, not a guarantee -- hence the
        deliberately positive energy below. The record with no energy must sort
        last regardless of where the real energies fall relative to the
        placeholder.
        """
        low = _create_mol_with_energy("CCO", -100.0)
        high = _create_mol_with_energy("CCCO", -50.0)
        positive = _create_mol_with_energy("CCCCCCO", +25.0)
        unknown = _energyless("CCCCO", seed=7)

        result = filter_unique_optimized([unknown, positive, high, low], rmsd_threshold=0.3)

        assert len(result) == 4
        assert [Chem.MolToSmiles(Chem.RemoveHs(m)) for m in result] == [
            Chem.MolToSmiles(Chem.RemoveHs(low)),
            Chem.MolToSmiles(Chem.RemoveHs(high)),
            Chem.MolToSmiles(Chem.RemoveHs(positive)),
            Chem.MolToSmiles(Chem.RemoveHs(unknown)),
        ], "the record with no energy must sort after every record that has one"

    def test_two_energyless_duplicates_still_collapse(self):
        """The inverse assertion, and the reason the two above are safe.

        A filter that declared every energy-less record distinct would pass
        both tests above while silently switching duplicate removal off for
        this whole class of input. Two identical geometries with no energy at
        all must still collapse to one: the energy guard cannot apply, so RMSD
        alone decides -- which is precisely what the legacy all-pairs filter
        did.
        """
        first = _energyless("CCO", seed=42)
        second = _energyless("CCO", seed=42)

        result = filter_unique_optimized([first, second], rmsd_threshold=0.3)

        assert len(result) == 1, (
            "two bit-identical energy-less conformers both survived, so "
            "tolerating a missing energy has disabled dedup for this input "
            "rather than falling back to RMSD only"
        )

    def test_an_energyless_record_is_compared_against_one_that_has_energy(self):
        """Energy-less records are compared against EVERYTHING, not just each
        other.

        No energy gap can prove a pair is not a duplicate when one side has no
        energy, so the partitioning that makes the filter sub-quadratic has no
        licence to separate such a record from anything. The legacy all-pairs
        filter compared it to every survivor; so must this one.
        """
        with_energy = _create_mol_with_energy("CCO", -100.0)
        without = _energyless("CCO", seed=42)
        # Same compound, same embedding seed and force field -> RMSD ~= 0.
        assert Chem.MolToSmiles(Chem.RemoveHs(with_energy)) == Chem.MolToSmiles(
            Chem.RemoveHs(without)
        ), "test premise: same compound"

        result = filter_unique_optimized(
            [with_energy, without], rmsd_threshold=0.3, energy_cluster_window=0.01
        )

        assert len(result) == 1, (
            "a record with no energy escaped comparison against a duplicate "
            "that has one, because the energy partitioning separated them"
        )


class TestTheSurvivingFilterKeepsTheLegacyVerdicts:
    """Values recorded from the legacy all-pairs ``filter_unique`` before it was
    deleted (cluster B5 phase 4a).

    Auto3D carried two conformer filters with the same duplicate criterion, each
    acting as the other's oracle, until 3.0.0. These cases were run against BOTH
    and are asserted here as literals so the surviving filter's verdicts stay
    pinned now that there is nothing left to compare against.
    """

    def test_distinct_conformers_of_one_molecule_all_survive(self):
        """Three genuinely different conformers of a flexible chain: 3 kept.

        Both filters returned 3 for this input.
        """

        def conformer(seed: int, energy_ev: float) -> Chem.Mol:
            m = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))  # flexible chain
            AllChem.EmbedMolecule(m, randomSeed=seed)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("Converged", "true")
            set_e_tot_from_ev(m, energy_ev)
            return m

        mols = [conformer(42, -12.0), conformer(7, -11.0), conformer(123, -10.0)]

        assert len(filter_unique_optimized(mols, rmsd_threshold=0.3)) == 3
        # A single cluster (huge window) must give the same answer -- that
        # equivalence is the whole justification for the energy partitioning.
        assert (
            len(filter_unique_optimized(mols, rmsd_threshold=0.3, energy_cluster_window=100.0)) == 3
        )

    def test_a_malformed_mixed_list_keeps_the_recorded_three(self):
        """The input the two filters used to DISAGREE about.

        An energy-bearing conformer, an energy-less duplicate of it, an
        energy-less record of a different compound, and a distinct third
        compound. The legacy filter kept three of the four -- merging the
        energy-less ethanol into the one that has an energy -- and so must this
        one.
        """
        a = _create_mol_with_energy("CCO", -100.0)
        a.SetProp("_Name", "ethanol_with_energy")
        b = _energyless("CCO", seed=42, name="ethanol_no_energy")
        c = _energyless("CCCCO", seed=11, name="butanol_no_energy")
        d = _create_mol_with_energy("CCCCCCO", -80.0)
        d.SetProp("_Name", "heptanol_with_energy")

        kept = filter_unique_optimized([a, b, c, d], rmsd_threshold=0.3)

        assert {m.GetProp("_Name") for m in kept} == {
            "ethanol_with_energy",
            "butanol_no_energy",
            "heptanol_with_energy",
        }

    def test_an_rmsd_threshold_sweep_straddling_a_measured_pair(self):
        """A threshold below the pair's actual RMSD keeps both; above, merges.

        Recorded from both filters. Sweeping across the *measured* RMSD -- not
        a fixed number -- is what makes this fail for a filter that ignores
        ``rmsd_threshold`` altogether, which an equal-length comparison at a
        single threshold would not.
        """
        from rdkit.Chem import rdMolAlign

        first = _create_mol_with_energy("CCCCCCCC", -100.0)  # octane
        second = Chem.Mol(first)
        AllChem.EmbedMolecule(second, randomSeed=99)
        AllChem.MMFFOptimizeMolecule(second)
        set_e_tot_from_ev(second, -100.0)  # energies agree -> RMSD decides
        second.SetProp("Converged", "true")

        rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(first), Chem.RemoveHs(second))
        assert rmsd > 0.05, "test premise: the conformers must be distinct"

        for crit, expected in ((rmsd / 2, 2), (rmsd * 2, 1)):
            kept = filter_unique_optimized([first, second], rmsd_threshold=crit)
            assert len(kept) == expected, f"at rmsd_threshold={crit}"

    def test_two_energyless_conformers_dedup_by_rmsd_alone(self):
        """Ported from the legacy filter's own suite.

        Neither record carries ``E_tot``, so the energy half of the duplicate
        criterion cannot apply and RMSD alone decides. The legacy filter kept
        one; so must this one.
        """
        mol = _energyless("CCO", seed=42)
        duplicate = Chem.Mol(mol)
        duplicate.SetProp("Converged", "true")

        assert len(filter_unique_optimized([mol, duplicate], rmsd_threshold=0.3)) == 1


class TestConvergencePropertyAbsenceFiltersLikeTrue:
    """A whole SDF that carries no 'Converged' property must filter exactly as
    the same records marked Converged=True do.

    Ported from the legacy filter's suite (it lived beside the validation tests
    because ``filter_unique`` used to live in ``utils/validation.py``). Only
    ``batchopt`` writes ``Converged``; an ``opt_geometry`` output, an
    ORCA/Gaussian export or a hand-built conformer set carries none, and
    treating that as "did not converge" deleted every record.
    """

    _SDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "example.sdf")

    def _mols(self) -> list[Chem.Mol]:
        supp = Chem.SDMolSupplier(self._SDF, removeHs=False)
        return [mol for mol in supp if mol is not None]

    def test_an_unflagged_file_keeps_what_a_flagged_one_keeps(self):
        flagged = self._mols()
        for mol in flagged:
            mol.SetProp("Converged", "True")
        expected = len(filter_unique_optimized(flagged, rmsd_threshold=0.3))
        assert expected >= 1, "test premise: the flagged file must keep something"

        unflagged = self._mols()
        for mol in unflagged:
            mol.ClearProp("Converged")
            assert not mol.HasProp("Converged")

        result = filter_unique_optimized(unflagged, rmsd_threshold=0.3)
        assert len(result) == expected, (
            f"{len(unflagged)} record(s) with no 'Converged' property kept "
            f"{len(result)}, but the same records marked Converged=True keep "
            f"{expected}"
        )

    def test_an_explicit_false_still_empties_the_selection(self):
        """The inverse: absence is not failure, but a stated failure is."""
        mols = self._mols()
        for mol in mols:
            mol.SetProp("Converged", "False")
        assert filter_unique_optimized(mols, rmsd_threshold=0.3) == []


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

    from Auto3D.domain import filtering

    # Five DISTINCT conformers of one molecule so all survive as unique,
    # maximizing inner-loop comparisons (the O(n^2) path strips Hs each pair).
    mols = []
    base = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    cids = AllChem.EmbedMultipleConfs(base, numConfs=5, randomSeed=1)
    from Auto3D.foundation.utils.energy import set_e_tot_from_ev as _set_e

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

    from Auto3D.domain.filtering import _filter_within_cluster

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


class TestTheFilterSaysWhyItDroppedThings:
    """``filter_conformers`` reports a count per reason, not just a survivor list.

    Returning a bare list is what let ``ranking`` tell a user "No structure
    converged" for a species whose conformers were every one of them dropped
    for *stereochemistry* -- a message that points at ``--opt-steps`` and
    ``--convergence-threshold`` for a problem neither can fix.
    """

    def test_each_reason_is_counted_under_its_own_name(self):
        good = _create_mol_with_energy("CCO", -100.0)
        unconverged = _create_mol_with_energy("CCO", -99.0, converged=False)
        stereo_changed = _create_mol_with_energy("CCO", -98.0)
        stereo_changed.SetProp("Stereo_changed", "true")
        broken = _create_mol_with_energy("CC", -97.0)
        conf = broken.GetConformer()
        pos = conf.GetAtomPosition(0)
        conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))
        duplicate = _create_mol_with_energy("CCO", -100.0)  # same as `good`

        result = filter_conformers(
            [None, good, unconverged, stereo_changed, broken, duplicate],
            rmsd_threshold=0.3,
        )

        assert [m is good for m in result.kept] == [True]
        assert result.dropped == {
            "unparsed": 1,
            "unconverged": 1,
            "stereochemistry": 1,
            "connectivity": 1,
            "duplicate": 1,
        }

    def test_nothing_dropped_reports_nothing(self):
        """The inverse: a clean input must not manufacture a reason.

        A result object that always carried a non-empty ``dropped`` would make
        every ranking message name a cause that did not happen.
        """
        result = filter_conformers([_create_mol_with_energy("CCO", -100.0)], rmsd_threshold=0.3)
        assert result.dropped == {}
        assert result.reasons == ()
        assert result.summary() == ""

    def test_summary_names_every_reason_that_fired_in_declared_order(self):
        result = FilterResult(
            kept=[],
            dropped={"duplicate": 3, "unconverged": 2, "stereochemistry": 1},
        )
        assert result.reasons == ("unconverged", "stereochemistry", "duplicate")
        assert result.summary() == (
            "2 marked Converged=false, "
            "1 changed stereochemistry during optimization, "
            "3 duplicates of a kept conformer"
        )

    def test_zero_counts_are_not_reported_as_reasons(self):
        result = FilterResult(kept=[], dropped={"unconverged": 0, "duplicate": 2})
        assert result.reasons == ("duplicate",)
        assert result.summary() == "2 duplicates of a kept conformer"

    def test_an_unknown_reason_is_refused_at_construction(self):
        """DROP_REASONS is the vocabulary, enforced.

        Without this, a producer misspelling a reason contributes a drop that
        ``summary()`` silently omits, so a user is told fewer conformers went
        missing than actually did.
        """
        with pytest.raises(ValueError, match="unknown filter drop reason"):
            FilterResult(kept=[], dropped={"unconvrged": 1})

    def test_every_declared_reason_has_a_phrase(self):
        """A reason with no phrase would raise KeyError from inside summary()
        -- while reporting a diagnostic, which is the worst place to fail."""
        for reason in DROP_REASONS:
            assert FilterResult(kept=[], dropped={reason: 1}).summary()

    def test_truncation_is_not_a_drop_reason(self):
        """`k` cutting the list short is selection, not a filter drop.

        Nothing is missing to explain: those conformers are valid and unique,
        they lost the ranking. Counting them would make every ``k=1`` run report
        drops it should not.
        """
        assert "truncated" not in DROP_REASONS
        distinct = [
            _create_mol_with_energy("CCO", -100.0),
            _create_mol_with_energy("CCCO", -90.0),
            _create_mol_with_energy("CCCCO", -80.0),
        ]
        result = filter_conformers(distinct, rmsd_threshold=0.3)
        assert len(result.kept) == 3
        assert result.dropped == {}
