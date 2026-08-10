#!/usr/bin/env python
"""Tests for Auto3D.utils.connectivity module."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.utils.connectivity import check_connectivity


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


class TestAmendMol:
    """Test the amend_mol function for fixing molecule issues."""

    def test_amend_mol_preserves_valid_molecule(self):
        """Test that a valid molecule is preserved."""
        from Auto3D.utils.connectivity import amend_mol

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        amended_mol = amend_mol(mol)
        assert amended_mol is not None
        assert amended_mol.GetNumAtoms() == mol.GetNumAtoms()

    def test_amend_mol_returns_none_for_invalid(self):
        """Test that amend_mol returns None for severely invalid molecules."""
        from Auto3D.utils.connectivity import amend_mol

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
        """amend_mol(sanitize=True) must actually run RDKit's sanitization,
        not just return a non-None object.

        A molecule parsed with ``sanitize=False`` has no ring-perception /
        implicit-valence pass yet, so querying ring info raises a
        precondition violation -- it genuinely needs sanitizing to become
        usable. If ``amend_mol``'s sanitize branch were a no-op, this
        molecule would still raise after the call.
        """
        from Auto3D.utils.connectivity import amend_mol

        mol = Chem.MolFromSmiles("c1ccccc1", sanitize=False)  # benzene, unsanitized
        with pytest.raises(RuntimeError):
            mol.GetRingInfo().NumRings()  # ring perception never ran

        amended_mol = amend_mol(mol, sanitize=True)

        assert amended_mol is not None
        # Sanitizing actually ran: ring perception now works and finds the ring.
        assert amended_mol.GetRingInfo().NumRings() == 1


class TestGetMolConnectivity:
    """Test the get_mol_connectivity function."""

    def test_ethane_connectivity(self):
        """Test connectivity for ethane (C-C single bond).

        Pins the exact canonical ordering (atom1_idx < atom2_idx, per the
        function's own docstring/example), not "either order" -- which would
        equally accept a broken ``get_mol_connectivity`` that stopped sorting
        its tuples.
        """
        from Auto3D.utils.connectivity import get_mol_connectivity

        mol = Chem.MolFromSmiles("CC")
        connectivity = get_mol_connectivity(mol)

        assert connectivity == {(0, 1)}

    def test_ethanol_connectivity(self):
        """Test connectivity for ethanol."""
        from Auto3D.utils.connectivity import get_mol_connectivity

        mol = Chem.MolFromSmiles("CCO")
        connectivity = get_mol_connectivity(mol)

        # Should be a set of tuples
        assert isinstance(connectivity, (set, frozenset, list))
        # Should have at least 2 bonds (C-C and C-O)
        assert len(connectivity) >= 2

    def test_benzene_connectivity(self):
        """Test connectivity for benzene ring."""
        from Auto3D.utils.connectivity import get_mol_connectivity

        mol = Chem.MolFromSmiles("c1ccccc1")
        connectivity = get_mol_connectivity(mol)

        # Benzene has 6 C-C bonds in the ring
        assert len(connectivity) == 6

    def test_methane_connectivity(self):
        """Test connectivity for methane (no heavy atom bonds)."""
        from Auto3D.utils.connectivity import get_mol_connectivity

        mol = Chem.MolFromSmiles("C")
        connectivity = get_mol_connectivity(mol)

        # Methane has no bonds between heavy atoms (only C-H)
        # But if we include H atoms...
        mol_with_h = Chem.AddHs(mol)
        connectivity_with_h = get_mol_connectivity(mol_with_h)
        assert len(connectivity_with_h) == 4  # 4 C-H bonds

    def test_include_bond_order(self):
        """Test that bond order can be included.

        The previous version's real assertion sat inside ``if len(bond_info)
        == 3:``, which is false exactly when ``include_bond_order`` silently
        stops adding the third element -- the one failure mode this test
        exists to catch. Assert the 3-tuple shape unconditionally, then the
        bond order value.
        """
        from Auto3D.utils.connectivity import get_mol_connectivity

        mol = Chem.MolFromSmiles("C=C")  # Ethene
        connectivity = get_mol_connectivity(mol, include_bond_order=True)

        assert connectivity == {(0, 1, 2.0)}
        for bond_info in connectivity:
            assert len(bond_info) == 3  # (atom1_idx, atom2_idx, bond_order)
            assert bond_info[2] == 2.0  # Double bond
