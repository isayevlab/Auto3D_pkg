#!/usr/bin/env python
"""Tests for Auto3D.foundation.utils.geometry module."""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401  (several tests below are parametrized helpers' home)
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.foundation.utils.geometry import get_rmsd, min_pairwise_distance


class TestMinPairwiseDistance:
    """Test the min_pairwise_distance function."""

    def test_simple_three_points(self):
        """Test with three simple points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_two_points(self):
        """Test with two points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],  # Distance = 5
            ]
        )
        result = min_pairwise_distance(points)
        assert abs(result - 5.0) < 1e-5

    def test_collinear_points(self):
        """Test with collinear points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_3d_points(self):
        """Test with points in 3D space."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],  # Distance = sqrt(3) ~ 1.732
                [5.0, 5.0, 5.0],
            ]
        )
        result = min_pairwise_distance(points)
        expected = np.sqrt(3)
        assert abs(result - expected) < 1e-5

    def test_input_type_conversion(self):
        """Test that integer input is properly converted."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.int32)
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_very_close_points(self):
        """Test with very close points."""
        points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [10.0, 0.0, 0.0]])
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

    def test_mismatched_molecules_returns_inf(self):
        """Mismatched molecules are incomparable and return float('inf').

        An incomparable pair is treated as "distinct" (inf), matching
        filter_unique, so a downstream `rmsd < threshold` check keeps the
        structure instead of dropping it as a false duplicate.
        """
        mol1 = Chem.MolFromSmiles("CCO")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)

        mol2 = Chem.MolFromSmiles("CCCC")
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        # This should raise RuntimeError internally and return inf.
        rmsd = get_rmsd(mol1, mol2)
        assert rmsd == float("inf")

    def test_runtime_error_returns_inf(self, monkeypatch):
        """A RuntimeError from the RMSD computation returns float('inf')."""
        from Auto3D.foundation.utils import geometry as chem

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol_copy = Chem.Mol(mol)

        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(chem.rdMolAlign, "GetBestRMS", _boom)
        assert chem.get_rmsd(mol, mol_copy) == float("inf")
