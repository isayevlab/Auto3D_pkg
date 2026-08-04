#!/usr/bin/env python
"""Tests for Auto3D.utils.molprops module."""
from __future__ import annotations

from rdkit import Chem

from Auto3D.constants import MAX_CONFORMERS_CAP
from Auto3D.utils.molprops import calculate_conformer_count, get_mol_charge


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


class TestCalculateConformerCount:
    """Tests for calculate_conformer_count function."""

    def test_calculate_conformer_count_small_molecule(self):
        """Small molecule with few rotatable bonds should get reasonable count."""

        mol = Chem.MolFromSmiles("CCO")  # ethanol - 0 rotatable bonds
        count = calculate_conformer_count(mol)
        assert count >= 3  # At least num_heavy_atoms
        assert count <= MAX_CONFORMERS_CAP  # Capped at max

    def test_calculate_conformer_count_flexible_molecule(self):
        """Flexible molecule should get higher conformer count."""

        mol = Chem.MolFromSmiles("CCCCCCCC")  # octane - many rotatable bonds
        count = calculate_conformer_count(mol)
        assert count > 10  # Should be significant

    def test_calculate_conformer_count_respects_cap(self):
        """Very flexible molecules should be capped at MAX_CONFORMERS_CAP."""

        mol = Chem.MolFromSmiles("C" * 30)  # very long chain
        count = calculate_conformer_count(mol)
        assert count == MAX_CONFORMERS_CAP  # Should hit cap

    def test_calculate_conformer_count_minimum_is_heavy_atoms(self):
        """Conformer count should be at least the number of heavy atoms."""

        mol = Chem.MolFromSmiles("C")  # methane - single heavy atom
        count = calculate_conformer_count(mol)
        assert count >= 1  # At least 1 heavy atom

    def test_calculate_conformer_count_returns_int(self):
        """Result should always be an integer."""

        mol = Chem.MolFromSmiles("CCCCC")  # pentane
        count = calculate_conformer_count(mol)
        assert isinstance(count, int)

    def test_calculate_conformer_count_zero_rotatable_bonds(self):
        """Molecule with zero rotatable bonds should return at least heavy atom count."""

        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene - 0 rotatable bonds
        count = calculate_conformer_count(mol)
        # Benzene has 6 heavy atoms
        assert count >= 6

    def test_calculate_conformer_count_molecule_with_hydrogens(self):
        """Function should work correctly with molecules that have explicit hydrogens."""

        mol = Chem.MolFromSmiles("CCO")
        mol_h = Chem.AddHs(mol)
        count = calculate_conformer_count(mol_h)
        # Should still count only heavy atoms (C, C, O = 3)
        assert count >= 3
