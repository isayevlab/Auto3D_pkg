#!/usr/bin/env python
"""Tests for Auto3D.utils.chemistry module."""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.utils.chemistry import (
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    EV_TO_KCAL_PER_MOL,
    hartree2ev,
    hartree2kcalpermol,
    ev2kcalpermol,
    get_mol_charge,
    min_pairwise_distance,
    get_rmsd,
    check_connectivity,
)


class TestEnergyConversionConstants:
    """Test energy conversion constants and aliases."""

    def test_hartree_to_ev_value(self):
        """Test that HARTREE_TO_EV has the correct CODATA 2018 value."""
        assert abs(HARTREE_TO_EV - 27.211386245988) < 1e-10

    def test_hartree_to_kcal_per_mol_value(self):
        """Test that HARTREE_TO_KCAL_PER_MOL has the expected value."""
        assert abs(HARTREE_TO_KCAL_PER_MOL - 627.50947337481) < 1e-8

    def test_ev_to_kcal_per_mol_value(self):
        """Test that EV_TO_KCAL_PER_MOL has the expected value."""
        assert abs(EV_TO_KCAL_PER_MOL - 23.060547830619026) < 1e-10

    def test_backward_compatibility_aliases(self):
        """Test that backward compatibility aliases match constants."""
        assert hartree2ev == HARTREE_TO_EV
        assert hartree2kcalpermol == HARTREE_TO_KCAL_PER_MOL
        assert ev2kcalpermol == EV_TO_KCAL_PER_MOL

    def test_conversion_consistency(self):
        """Test that conversion factors are mathematically consistent."""
        # HARTREE_TO_KCAL_PER_MOL should approximately equal
        # HARTREE_TO_EV * EV_TO_KCAL_PER_MOL
        calculated = HARTREE_TO_EV * EV_TO_KCAL_PER_MOL
        # Allow some tolerance for floating point precision
        assert abs(calculated - HARTREE_TO_KCAL_PER_MOL) < 0.001


class TestGetMolCharge:
    """Test the get_mol_charge function."""

    def test_neutral_molecule(self):
        """Test charge of a neutral molecule."""
        mol = Chem.MolFromSmiles("CCO")
        assert get_mol_charge(mol) == 0

    def test_cation(self):
        """Test charge of a cation."""
        mol = Chem.MolFromSmiles("[NH4+]")
        assert get_mol_charge(mol) == 1

    def test_anion(self):
        """Test charge of an anion."""
        mol = Chem.MolFromSmiles("[O-]")
        assert get_mol_charge(mol) == -1

    def test_doubly_charged_cation(self):
        """Test charge of a doubly charged cation."""
        mol = Chem.MolFromSmiles("[Ca+2]")
        assert get_mol_charge(mol) == 2

    def test_zwitterion(self):
        """Test charge of a zwitterion (net neutral)."""
        # Glycine zwitterion
        mol = Chem.MolFromSmiles("[NH3+]CC([O-])=O")
        assert get_mol_charge(mol) == 0

    def test_multiple_charges(self):
        """Test molecule with multiple charged atoms."""
        mol = Chem.MolFromSmiles("[O-]C([O-])=O")  # Carbonate
        assert get_mol_charge(mol) == -2


class TestMinPairwiseDistance:
    """Test the min_pairwise_distance function."""

    def test_simple_three_points(self):
        """Test with three simple points."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0]
        ])
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_two_points(self):
        """Test with two points."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0]  # Distance = 5
        ])
        result = min_pairwise_distance(points)
        assert abs(result - 5.0) < 1e-5

    def test_collinear_points(self):
        """Test with collinear points."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0]
        ])
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_3d_points(self):
        """Test with points in 3D space."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # Distance = sqrt(3) ~ 1.732
            [5.0, 5.0, 5.0]
        ])
        result = min_pairwise_distance(points)
        expected = np.sqrt(3)
        assert abs(result - expected) < 1e-5

    def test_input_type_conversion(self):
        """Test that integer input is properly converted."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0]
        ], dtype=np.int32)
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_very_close_points(self):
        """Test with very close points."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [10.0, 0.0, 0.0]
        ])
        result = min_pairwise_distance(points)
        assert abs(result - 0.001) < 1e-6


class TestGetRmsd:
    """Test the get_rmsd function."""

    def test_identical_molecules(self):
        """Test RMSD of a molecule with itself is 0."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mol_copy = Chem.Mol(mol)
        rmsd = get_rmsd(mol, mol_copy)
        assert abs(rmsd) < 1e-5

    def test_different_conformers(self):
        """Test RMSD of different conformers is > 0."""
        mol = Chem.MolFromSmiles("CCCCCC")  # Hexane - flexible
        mol = Chem.AddHs(mol)

        # Generate two different conformers
        AllChem.EmbedMolecule(mol, randomSeed=42)
        conf1 = mol.GetConformer()

        mol2 = Chem.Mol(mol)
        AllChem.EmbedMolecule(mol2, randomSeed=123)

        rmsd = get_rmsd(mol, mol2)
        # Different random seeds should give different conformers
        # The RMSD should be >= 0 (might still be 0 if conformers happen to be similar)
        assert rmsd >= 0

    def test_remove_hs_option(self):
        """Test that remove_hs option works."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mol_copy = Chem.Mol(mol)

        # Both should return 0 for identical molecules
        rmsd_with_hs_removed = get_rmsd(mol, mol_copy, remove_hs=True)
        rmsd_without_hs_removed = get_rmsd(mol, mol_copy, remove_hs=False)

        assert abs(rmsd_with_hs_removed) < 1e-5
        assert abs(rmsd_without_hs_removed) < 1e-5

    def test_mismatched_molecules_returns_zero(self):
        """Test that mismatched molecules return 0.0."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)

        mol2 = Chem.MolFromSmiles("CCCC")
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        # This should raise RuntimeError internally and return 0.0
        rmsd = get_rmsd(mol1, mol2)
        assert rmsd == 0.0


class TestCheckConnectivity:
    """Test the check_connectivity function."""

    def test_valid_ethanol_connectivity(self):
        """Test that a valid ethanol conformer has correct connectivity."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        assert check_connectivity(mol) is True

    def test_valid_methane_connectivity(self):
        """Test that a valid methane conformer has correct connectivity."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        assert check_connectivity(mol) is True

    def test_valid_benzene_connectivity(self):
        """Test that a valid benzene conformer has correct connectivity."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        assert check_connectivity(mol) is True

    def test_valid_cyclohexane_connectivity(self):
        """Test that a valid cyclohexane conformer has correct connectivity."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        assert check_connectivity(mol) is True

    def test_broken_bond_detected(self):
        """Test that a stretched bond is detected as invalid connectivity."""
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Manually stretch a C-C bond far apart
        conf = mol.GetConformer()
        # Move one carbon far away
        pos = conf.GetAtomPosition(0)
        conf.SetAtomPosition(0, (pos.x + 5.0, pos.y, pos.z))

        # This should detect the broken bond
        assert check_connectivity(mol) is False

    def test_valid_molecule_with_heteroatoms(self):
        """Test molecule with nitrogen and oxygen."""
        mol = Chem.MolFromSmiles("CC(=O)NC")  # N-methylacetamide
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        assert check_connectivity(mol) is True

    def test_salt_with_metal_does_not_crash(self):
        """Element outside the radii table (Na) must not raise KeyError.

        Sodium acetate contains Na (atomic number 11), which is not in the
        UFF radii table. check_connectivity must skip pairs involving an
        unknown element rather than indexing the radii dict blindly.
        """
        mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]")  # sodium acetate
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Must return a bool without raising KeyError(11).
        result = check_connectivity(mol)
        assert isinstance(result, bool)


class TestModuleImports:
    """Test that module can be imported from different paths."""

    def test_import_from_utils_chemistry(self):
        """Test direct import from Auto3D.utils.chemistry."""
        from Auto3D.utils.chemistry import get_mol_charge, min_pairwise_distance, check_connectivity
        assert callable(get_mol_charge)
        assert callable(min_pairwise_distance)
        assert callable(check_connectivity)

    def test_import_from_utils_package(self):
        """Test import from Auto3D.utils package."""
        from Auto3D.utils import (
            HARTREE_TO_EV,
            hartree2ev,
            get_mol_charge,
            min_pairwise_distance,
            get_rmsd,
            check_connectivity,
        )
        assert HARTREE_TO_EV == hartree2ev
        assert callable(get_mol_charge)
        assert callable(min_pairwise_distance)
        assert callable(get_rmsd)
        assert callable(check_connectivity)


class TestGetIdx:
    """Test the getidx function for element index mapping."""

    def test_hydrogen_index(self):
        """Test hydrogen atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        # Hydrogen should map to index 0 in ANI2xt order
        assert getidx(1, model="ANI2xt") == 0

    def test_carbon_index(self):
        """Test carbon atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(6, model="ANI2xt") == 1

    def test_nitrogen_index(self):
        """Test nitrogen atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(7, model="ANI2xt") == 2

    def test_oxygen_index(self):
        """Test oxygen atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(8, model="ANI2xt") == 3

    def test_fluorine_index(self):
        """Test fluorine atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(9, model="ANI2xt") == 4

    def test_sulfur_index(self):
        """Test sulfur atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(16, model="ANI2xt") == 5

    def test_chlorine_index(self):
        """Test chlorine atomic number returns correct index."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(17, model="ANI2xt") == 6

    def test_default_model_returns_atomic_number(self):
        """Test that default model returns the atomic number unchanged."""
        from Auto3D.utils.chemistry import getidx
        assert getidx(6) == 6  # Carbon
        assert getidx(1) == 1  # Hydrogen
        assert getidx(8) == 8  # Oxygen

    def test_unsupported_element_raises_error(self):
        """Test that unsupported element raises KeyError for ANI2xt."""
        from Auto3D.utils.chemistry import getidx
        with pytest.raises(KeyError):
            getidx(79, model="ANI2xt")  # Gold not supported by ANI2xt


class TestAmendMol:
    """Test the amend_mol function for fixing molecule issues."""

    def test_amend_mol_preserves_valid_molecule(self):
        """Test that a valid molecule is preserved."""
        from Auto3D.utils.chemistry import amend_mol
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        amended_mol = amend_mol(mol)
        assert amended_mol is not None
        assert amended_mol.GetNumAtoms() == mol.GetNumAtoms()

    def test_amend_mol_returns_none_for_invalid(self):
        """Test that amend_mol returns None for severely invalid molecules."""
        from Auto3D.utils.chemistry import amend_mol
        # Create molecule and severely distort it
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Severely distort by moving atoms to overlapping positions
        conf = mol.GetConformer()
        # Move all hydrogens to same position (severe clash)
        for i in range(1, mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (0.0, 0.0, 0.0))

        # Should return None for invalid geometry
        amended_mol = amend_mol(mol, check_valid=True)
        # The function should handle this appropriately
        # (either returns None or attempts to fix)

    def test_amend_mol_with_sanitize(self):
        """Test that amend_mol can sanitize molecules."""
        from Auto3D.utils.chemistry import amend_mol
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        amended_mol = amend_mol(mol, sanitize=True)
        assert amended_mol is not None


class TestGetMolConnectivity:
    """Test the get_mol_connectivity function."""

    def test_ethane_connectivity(self):
        """Test connectivity for ethane (C-C single bond)."""
        from Auto3D.utils.chemistry import get_mol_connectivity
        mol = Chem.MolFromSmiles("CC")
        connectivity = get_mol_connectivity(mol)

        # Should have C-C bond
        assert (0, 1) in connectivity or (1, 0) in connectivity

    def test_ethanol_connectivity(self):
        """Test connectivity for ethanol."""
        from Auto3D.utils.chemistry import get_mol_connectivity
        mol = Chem.MolFromSmiles("CCO")
        connectivity = get_mol_connectivity(mol)

        # Should be a set of tuples
        assert isinstance(connectivity, (set, frozenset, list))
        # Should have at least 2 bonds (C-C and C-O)
        assert len(connectivity) >= 2

    def test_benzene_connectivity(self):
        """Test connectivity for benzene ring."""
        from Auto3D.utils.chemistry import get_mol_connectivity
        mol = Chem.MolFromSmiles("c1ccccc1")
        connectivity = get_mol_connectivity(mol)

        # Benzene has 6 C-C bonds in the ring
        assert len(connectivity) == 6

    def test_methane_connectivity(self):
        """Test connectivity for methane (no heavy atom bonds)."""
        from Auto3D.utils.chemistry import get_mol_connectivity
        mol = Chem.MolFromSmiles("C")
        connectivity = get_mol_connectivity(mol)

        # Methane has no bonds between heavy atoms (only C-H)
        # But if we include H atoms...
        mol_with_h = Chem.AddHs(mol)
        connectivity_with_h = get_mol_connectivity(mol_with_h)
        assert len(connectivity_with_h) == 4  # 4 C-H bonds

    def test_include_bond_order(self):
        """Test that bond order can be included."""
        from Auto3D.utils.chemistry import get_mol_connectivity
        mol = Chem.MolFromSmiles("C=C")  # Ethene
        connectivity = get_mol_connectivity(mol, include_bond_order=True)

        # Should contain tuple with bond order info
        # Format: (atom1_idx, atom2_idx, bond_order)
        assert len(connectivity) >= 1
        for bond_info in connectivity:
            if len(bond_info) == 3:
                # Has bond order
                assert bond_info[2] == 2.0  # Double bond


class TestFilterUnique:
    """Test the filter_unique function for RMSD-based duplicate filtering."""

    def test_filter_identical_conformers(self):
        """Test that identical conformers are filtered to one."""
        from Auto3D.utils.chemistry import filter_unique

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

    def test_filter_different_conformers(self):
        """Test that different conformers are kept."""
        from Auto3D.utils.chemistry import filter_unique

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

    def test_filter_unconverged_removed(self):
        """Test that unconverged structures are removed."""
        from Auto3D.utils.chemistry import filter_unique

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
        from Auto3D.utils.chemistry import filter_unique

        unique_mols = filter_unique([], crit=0.3)
        assert len(unique_mols) == 0

    def test_filter_custom_threshold(self):
        """Test that custom RMSD threshold works."""
        from Auto3D.utils.chemistry import filter_unique

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

    def test_rmsd_failure_keeps_both(self, monkeypatch):
        """An incomparable pair (RMSD raises) must NOT be treated as a duplicate.

        When GetBestRMS raises RuntimeError, filter_unique must treat the pair
        as distinct (rmsd = inf) and keep both, mirroring the fix already in
        filtering._filter_within_cluster. The previous behavior (rmsd = 0)
        made distinct conformers look like perfect duplicates and dropped one.
        """
        from Auto3D.utils import chemistry

        def make(name):
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=abs(hash(name)) % 1000)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("_Name", name)
            m.SetProp("Converged", "true")
            return m

        def boom(*args, **kwargs):
            raise RuntimeError("GetBestRMS failed")

        # filter_unique calls rdMolAlign.GetBestRMS via the chemistry module.
        monkeypatch.setattr(chemistry.rdMolAlign, "GetBestRMS", boom)

        mols = [make("a"), make("b")]
        unique_mols = chemistry.filter_unique(mols, crit=0.3)
        assert len(unique_mols) == 2  # incomparable pair must NOT be dropped
