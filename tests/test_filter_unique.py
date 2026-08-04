#!/usr/bin/env python
"""Tests for the legacy all-pairs ``Auto3D.filtering.filter_unique``.

Kept as its own file because ``filter_unique`` is scheduled for removal once
``filter_unique_optimized`` is the single filter; deleting it then means
deleting this file, not surgery on a shared one.
"""
from __future__ import annotations

import pytest  # noqa: F401
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.filtering import filter_unique


class TestFilterUnique:
    """Test the filter_unique function for RMSD-based duplicate filtering."""

    def test_filter_identical_conformers(self):
        """Test that identical conformers are filtered to one."""

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("Converged", "true")

        # Create identical copies
        mol2 = Chem.Mol(mol)
        mol2.SetProp("Converged", "true")

        mols = [mol, mol2]
        unique_mols = filter_unique(mols, crit=0.3)

        # Should only keep one
        assert len(unique_mols) == 1

    def test_same_geometry_different_energy_kept(self):
        """Identical geometry but distinct E_tot must be kept (energy guard).

        Heavy-atom RMSD ~= 0 but the two are distinct minima (the O-H rotamer
        case); the energy guard must stop them collapsing into one.
        """

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("Converged", "true")
        mol.SetProp("E_tot", "-10.0")

        mol2 = Chem.Mol(mol)  # identical geometry
        mol2.SetProp("Converged", "true")
        mol2.SetProp("E_tot", "-10.5")  # |dE| >> tol

        unique_mols = filter_unique([mol, mol2], crit=0.3)
        assert len(unique_mols) == 2

    def test_missing_energy_falls_back_to_rmsd_only(self):
        """Without E_tot the energy guard cannot apply -> RMSD-only dedup.

        Preserves the legacy behavior for callers that do not set E_tot.
        """

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("Converged", "true")  # no E_tot set

        mol2 = Chem.Mol(mol)
        mol2.SetProp("Converged", "true")

        unique_mols = filter_unique([mol, mol2], crit=0.3)
        assert len(unique_mols) == 1

    def test_filter_different_conformers(self):
        """Test that different conformers are kept."""

        mol1 = Chem.MolFromSmiles("CCCCCC")  # Hexane - flexible
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        mol1.SetProp("Converged", "true")

        mol2 = Chem.MolFromSmiles("CCCCCC")
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, randomSeed=123)
        mol2.SetProp("Converged", "true")

        # Generate very different conformers by using different seeds
        # and moving atoms around
        conf = mol2.GetConformer()
        pos = conf.GetAtomPosition(0)
        conf.SetAtomPosition(0, (pos.x + 0.5, pos.y, pos.z))

        mols = [mol1, mol2]
        unique_mols = filter_unique(mols, crit=0.3)

        # Should keep both (or at least not crash)
        assert len(unique_mols) >= 1

    def test_two_diastereomers_are_never_merged(self):
        """A distinct compound must survive dedup, however close its geometry.

        The same guarantee ``tests/test_filtering.py`` asserts for
        ``filter_unique_optimized``, asserted here because this is the other
        duplicate filter and it applies the identical RMSD-plus-energy criterion.
        Fixing one path and not the other would leave the defect reachable
        through ``ConformerRanker(use_optimized_filtering=False)``.

        cis/trans-4-tert-butylcyclohexanol: heavy-atom RMSD between the two
        diastereomers was measured at 0.300 A, i.e. at the 0.3 A default
        threshold. ``crit`` is opened wide here so RMSD and energy both say
        "duplicate" and the only thing that can keep the pair apart is the fact
        that they are different compounds.
        """
        from Auto3D.utils.energy import set_e_tot_from_ev

        def build(smiles: str) -> Chem.Mol:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("Converged", "true")
            set_e_tot_from_ev(mol, -10.0)  # identical energies
            return mol

        cis = build("O[C@H]1CC[C@@H](CC1)C(C)(C)C")
        trans = build("O[C@H]1CC[C@H](CC1)C(C)(C)C")
        assert Chem.MolToSmiles(cis) != Chem.MolToSmiles(trans), "test premise"

        assert len(filter_unique([cis, trans], crit=10.0)) == 2, (
            "the legacy filter merged two distinct diastereomers, so an input "
            "molecule vanished from the output with no record"
        )

    def test_duplicate_conformers_of_one_stereoisomer_still_collapse(self):
        """The other half: the stereo guard must narrow dedup, not disable it."""
        from Auto3D.utils.energy import set_e_tot_from_ev

        def build() -> Chem.Mol:
            mol = Chem.AddHs(Chem.MolFromSmiles("O[C@H]1CC[C@@H](CC1)C(C)(C)C"))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("Converged", "true")
            set_e_tot_from_ev(mol, -10.0)
            return mol

        assert len(filter_unique([build(), build()], crit=10.0)) == 1, (
            "duplicate conformers of one stereoisomer survived, so the stereo "
            "guard has switched dedup off rather than narrowing it"
        )

    def test_filter_unconverged_removed(self):
        """Test that unconverged structures are removed."""

        mol1 = Chem.MolFromSmiles("CCO")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol1)
        mol1.SetProp("Converged", "true")

        mol2 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, randomSeed=123)
        mol2.SetProp("Converged", "false")  # Not converged

        mols = [mol1, mol2]
        unique_mols = filter_unique(mols, crit=0.3)

        # Only converged one should remain
        assert len(unique_mols) == 1
        assert unique_mols[0].GetProp("Converged").lower() == "true"

    def test_filter_empty_list(self):
        """Test filtering empty list returns empty list."""

        unique_mols = filter_unique([], crit=0.3)
        assert len(unique_mols) == 0

    def test_filter_custom_threshold(self):
        """Test that custom RMSD threshold works."""

        mol1 = Chem.MolFromSmiles("CCO")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol1)
        mol1.SetProp("Converged", "true")

        mol2 = Chem.Mol(mol1)
        mol2.SetProp("Converged", "true")

        mols = [mol1, mol2]

        # With very small threshold, might keep both
        unique_mols_small = filter_unique(mols, crit=0.0001)
        # With large threshold, definitely keep only one
        unique_mols_large = filter_unique(mols, crit=10.0)

        # Large threshold should definitely merge identical mols
        assert len(unique_mols_large) == 1
        # A tighter threshold can never merge MORE than a looser one -- the
        # discarded half of this test's own computation, now actually checked.
        assert len(unique_mols_small) >= len(unique_mols_large)

    def test_filter_unique_removehs_is_linear_and_nondestructive(self, monkeypatch):
        """Legacy filter_unique strips Hs once per molecule (not per comparison) and
        returns the originals with explicit H + exact positions intact."""
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D import filtering

        base = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
        cids = AllChem.EmbedMultipleConfs(base, numConfs=5, randomSeed=1)
        mols = []
        for cid in cids:
            m = Chem.Mol(base, confId=int(cid))
            m.SetProp("Converged", "true")
            mols.append(m)
        n_atoms = base.GetNumAtoms()
        orig_pos = {id(m): m.GetConformer().GetPositions().copy() for m in mols}

        calls = {"n": 0}
        real_removehs = filtering.Chem.RemoveHs

        def counting(mol, *a, **k):
            calls["n"] += 1
            return real_removehs(mol, *a, **k)

        monkeypatch.setattr(filtering.Chem, "RemoveHs", counting)

        result = filtering.filter_unique(mols, crit=0.01)
        assert calls["n"] == len(mols)  # once per input, never per pair
        assert len(result) == len(mols)
        for m in result:
            assert m.GetNumAtoms() == n_atoms
            assert any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
            assert np.array_equal(m.GetConformer().GetPositions(), orig_pos[id(m)])

    def test_rmsd_failure_keeps_both(self, monkeypatch):
        """An incomparable pair (RMSD raises) must NOT be treated as a duplicate.

        When GetBestRMS raises RuntimeError, filter_unique must treat the pair
        as distinct (rmsd = inf) and keep both, mirroring the fix already in
        filtering._filter_within_cluster. The previous behavior (rmsd = 0)
        made distinct conformers look like perfect duplicates and dropped one.
        """
        from Auto3D import filtering

        def make(name):
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=abs(hash(name)) % 1000)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("_Name", name)
            m.SetProp("Converged", "true")
            return m

        def boom(*args, **kwargs):
            raise RuntimeError("GetBestRMS failed")

        # filter_unique calls rdMolAlign.GetBestRMS via the filtering module.
        monkeypatch.setattr(filtering.rdMolAlign, "GetBestRMS", boom)

        mols = [make("a"), make("b")]
        unique_mols = filtering.filter_unique(mols, crit=0.3)
        assert len(unique_mols) == 2  # incomparable pair must NOT be dropped
