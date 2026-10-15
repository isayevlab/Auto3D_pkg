#!/usr/bin/env python
"""Tests for Auto3D.domain.conformer_budget module."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from Auto3D.domain.conformer_budget import calculate_conformer_count
from Auto3D.foundation.constants import MAX_CONFORMERS_CAP


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
        """Conformer count should be at least the number of heavy atoms.

        The exact value is derivable, not just a lower bound: methane has 0
        rotatable bonds, so the rotatable-bond formula term is
        ``0 ** CONFORMER_ROTATABLE_EXP == 0`` regardless of the tuned
        constants, and ``max(1, num_heavy=1, 0) == 1``. ``>= 1`` would still
        pass even if the floor-at-1 protection were dropped and the formula
        term alone (inflated by a bug) drove the count above 1.
        """

        mol = Chem.MolFromSmiles("C")  # methane - single heavy atom
        count = calculate_conformer_count(mol)
        assert count == 1

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

    @pytest.mark.parametrize(
        "smiles",
        ["CCO", "c1ccccc1O", "CC(=O)N", "c1ccncc1", "Clc1ccccc1", "C1CCOCC1", "C1CCCCCC1"],
        ids=[
            "ethanol",
            "phenol",
            "acetamide",
            "pyridine",
            "chlorobenzene",
            "oxane",
            "cycloheptane",
        ],
    )
    def test_a_rotor_free_molecule_requests_sixteen(self, smiles):
        """Ring puckering is not a rotatable bond, and the rotor term is zero
        here, so the request would fall to the heavy-atom count (3 to 7). Under
        the embedding settings this package uses, oxane keeps 1 conformer of 6
        requested and 3 of 16, cycloheptane 1 of 7 and 2 of 16: the pool is still
        growing when a heavy-atom budget stops it."""
        assert calculate_conformer_count(Chem.MolFromSmiles(smiles)) == 16

    @pytest.mark.parametrize(
        "smiles", ["C", "[H+]", "[Na+]"], ids=["methane", "proton", "sodium_ion"]
    )
    def test_a_lone_atom_still_requests_one(self, smiles):
        assert calculate_conformer_count(Chem.MolFromSmiles(smiles)) == 1

    def test_the_floor_never_binds_above_one_rotatable_bond(self):
        """At one rotor the expression already gives 16; above it the floor is
        inert, so every benchmark molecule with a rotatable bond keeps its budget."""
        import sys

        sys.path.insert(0, "benchmarks")
        from bench_optimization_perf import SMILES

        for group in SMILES.values():
            for smi in group:
                mol = Chem.MolFromSmiles(smi)
                n_rot = rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveAllHs(mol))
                if n_rot == 0:
                    continue
                n_heavy = mol.GetNumHeavyAtoms()
                formula = int(2 * 8.481 * n_rot**1.642)
                assert calculate_conformer_count(mol) == min(max(1, n_heavy, formula), 1000), smi
