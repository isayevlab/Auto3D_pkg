#!/usr/bin/env python
"""Tests for Auto3D.utils.chemistry module."""
import pytest
from rdkit import Chem

from Auto3D.constants import MAX_CONFORMERS_CAP


class TestCalculateConformerCount:
    """Tests for calculate_conformer_count function."""

    def test_calculate_conformer_count_small_molecule(self):
        """Small molecule with few rotatable bonds should get reasonable count."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("CCO")  # ethanol - 0 rotatable bonds
        count = calculate_conformer_count(mol)
        assert count >= 3  # At least num_heavy_atoms
        assert count <= MAX_CONFORMERS_CAP  # Capped at max

    def test_calculate_conformer_count_flexible_molecule(self):
        """Flexible molecule should get higher conformer count."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("CCCCCCCC")  # octane - many rotatable bonds
        count = calculate_conformer_count(mol)
        assert count > 10  # Should be significant

    def test_calculate_conformer_count_respects_cap(self):
        """Very flexible molecules should be capped at MAX_CONFORMERS_CAP."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("C" * 30)  # very long chain
        count = calculate_conformer_count(mol)
        assert count == MAX_CONFORMERS_CAP  # Should hit cap

    def test_calculate_conformer_count_minimum_is_heavy_atoms(self):
        """Conformer count should be at least the number of heavy atoms."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("C")  # methane - single heavy atom
        count = calculate_conformer_count(mol)
        assert count >= 1  # At least 1 heavy atom

    def test_calculate_conformer_count_returns_int(self):
        """Result should always be an integer."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("CCCCC")  # pentane
        count = calculate_conformer_count(mol)
        assert isinstance(count, int)

    def test_calculate_conformer_count_zero_rotatable_bonds(self):
        """Molecule with zero rotatable bonds should return at least heavy atom count."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene - 0 rotatable bonds
        count = calculate_conformer_count(mol)
        # Benzene has 6 heavy atoms
        assert count >= 6

    def test_calculate_conformer_count_molecule_with_hydrogens(self):
        """Function should work correctly with molecules that have explicit hydrogens."""
        from Auto3D.utils.chemistry import calculate_conformer_count

        mol = Chem.MolFromSmiles("CCO")
        mol_h = Chem.AddHs(mol)
        count = calculate_conformer_count(mol_h)
        # Should still count only heavy atoms (C, C, O = 3)
        assert count >= 3
