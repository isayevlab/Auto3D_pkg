# tests/test_padding.py
"""Tests for vectorized padding module."""
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.batch_opt.padding import pad_molecular_batch, pad_from_mols


class TestPadMolecularBatch:
    """Tests for pad_molecular_batch function."""

    def test_basic_padding_shapes(self):
        """Vectorized padding should produce correct tensor shapes."""
        coords = [
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],  # 3 atoms
            [(0, 0, 0), (1, 0, 0)],              # 2 atoms
        ]
        species = [[6, 1, 1], [8, 1]]
        charges = [0, 0]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=0.0, species_pad=-1)

        assert c.shape == (2, 3, 3)  # batch=2, max_atoms=3, xyz=3
        assert s.shape == (2, 3)
        assert q.shape == (2,)

    def test_padding_values(self):
        """Padding values should be correctly applied."""
        coords = [
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],  # 3 atoms
            [(0, 0, 0), (1, 0, 0)],              # 2 atoms
        ]
        species = [[6, 1, 1], [8, 1]]
        charges = [0, 0]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=0.0, species_pad=-1)

        # Check padding values
        assert s[1, 2].item() == -1  # padding for species
        assert torch.allclose(c[1, 2], torch.tensor([0.0, 0.0, 0.0]))

    def test_actual_values_preserved(self):
        """Actual molecular data should be preserved correctly."""
        coords = [
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            [(7.0, 8.0, 9.0)],
        ]
        species = [[6, 1], [8]]
        charges = [0, -1]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=0.0, species_pad=-1)

        # Check actual values are preserved
        assert torch.allclose(c[0, 0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(c[0, 1], torch.tensor([4.0, 5.0, 6.0]))
        assert torch.allclose(c[1, 0], torch.tensor([7.0, 8.0, 9.0]))
        assert s[0, 0].item() == 6
        assert s[0, 1].item() == 1
        assert s[1, 0].item() == 8
        assert q[0].item() == 0
        assert q[1].item() == -1

    def test_requires_grad_enabled(self):
        """Coords tensor should have requires_grad=True for force calculation."""
        coords = [[(0, 0, 0), (1, 0, 0)]]
        species = [[6, 1]]
        charges = [0]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=0.0, species_pad=-1)

        assert c.requires_grad is True

    def test_custom_padding_values(self):
        """Custom padding values should be used correctly."""
        coords = [
            [(0, 0, 0), (1, 0, 0)],
            [(0, 0, 0)],
        ]
        species = [[6, 1], [8]]
        charges = [0, 0]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=99.0, species_pad=0)

        # Check custom padding values
        assert s[1, 1].item() == 0  # custom species_pad
        assert torch.allclose(c[1, 1], torch.tensor([99.0, 99.0, 99.0]))

    def test_single_molecule(self):
        """Should work with a single molecule."""
        coords = [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]]
        species = [[6, 1, 1]]
        charges = [0]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=0.0, species_pad=-1)

        assert c.shape == (1, 3, 3)
        assert s.shape == (1, 3)
        assert q.shape == (1,)

    def test_uniform_molecule_sizes(self):
        """Should work when all molecules have the same number of atoms."""
        coords = [
            [(0, 0, 0), (1, 0, 0)],
            [(2, 0, 0), (3, 0, 0)],
        ]
        species = [[6, 1], [7, 1]]
        charges = [0, 0]
        device = torch.device("cpu")

        c, s, q = pad_molecular_batch(coords, species, charges, device,
                                       coord_pad=0.0, species_pad=-1)

        assert c.shape == (2, 2, 3)
        assert s.shape == (2, 2)
        # No padding should be applied since all molecules have same size
        assert s[0, 0].item() == 6
        assert s[0, 1].item() == 1
        assert s[1, 0].item() == 7
        assert s[1, 1].item() == 1


class TestPadFromMols:
    """Tests for pad_from_mols function."""

    def test_basic_rdkit_molecules(self):
        """Should correctly pad RDKit Mol objects."""
        # Create simple molecules
        mol1 = Chem.AddHs(Chem.MolFromSmiles("C"))  # Methane - 5 atoms
        AllChem.EmbedMolecule(mol1, randomSeed=42)

        mol2 = Chem.AddHs(Chem.MolFromSmiles("O"))  # Water - 3 atoms
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        mols = [mol1, mol2]
        device = torch.device("cpu")

        c, s, q = pad_from_mols(mols, "AIMNET", device, coord_pad=0.0, species_pad=0)

        # Methane has 5 atoms (1C + 4H), water has 3 atoms (1O + 2H)
        assert c.shape == (2, 5, 3)  # max_atoms = 5
        assert s.shape == (2, 5)
        assert q.shape == (2,)

    def test_species_values_aimnet(self):
        """AIMNET model should use atomic numbers directly."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # Methane
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q = pad_from_mols(mols, "AIMNET", device, coord_pad=0.0, species_pad=0)

        # Carbon is atomic number 6, Hydrogen is 1
        species_list = s[0].tolist()
        assert 6 in species_list  # Carbon
        assert 1 in species_list  # Hydrogen

    def test_species_values_ani2xt(self):
        """ANI2xt model should use mapped indices."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))  # Methane
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q = pad_from_mols(mols, "ANI2xt", device, coord_pad=0.0, species_pad=-1)

        # ANI2xt mapping: H->0, C->1, N->2, O->3, F->4, S->5, Cl->6
        species_list = s[0].tolist()
        assert 1 in species_list  # Carbon maps to 1
        assert 0 in species_list  # Hydrogen maps to 0

    def test_charges_extracted(self):
        """Formal charges should be extracted from molecules."""
        mol1 = Chem.AddHs(Chem.MolFromSmiles("C"))  # Neutral
        AllChem.EmbedMolecule(mol1, randomSeed=42)

        # Create a charged molecule (acetate anion simplified)
        mol2 = Chem.AddHs(Chem.MolFromSmiles("[O-]"))  # Hydroxide
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        mols = [mol1, mol2]
        device = torch.device("cpu")

        c, s, q = pad_from_mols(mols, "AIMNET", device, coord_pad=0.0, species_pad=0)

        assert q[0].item() == 0   # Methane is neutral
        assert q[1].item() == -1  # Hydroxide has -1 charge

    def test_coords_match_conformer(self):
        """Coordinates should match RDKit conformer positions."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q = pad_from_mols(mols, "AIMNET", device, coord_pad=0.0, species_pad=0)

        # Get positions from RDKit
        conf = mol.GetConformer()
        expected_positions = conf.GetPositions()

        # Compare coordinates (detach since c requires grad)
        actual_positions = c[0, :mol.GetNumAtoms()].detach().numpy()

        import numpy as np
        np.testing.assert_array_almost_equal(actual_positions, expected_positions)

    def test_requires_grad_enabled(self):
        """Coords tensor should have requires_grad=True."""
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        c, s, q = pad_from_mols(mols, "AIMNET", device, coord_pad=0.0, species_pad=0)

        assert c.requires_grad is True

    def test_ani2xt_unsupported_element_raises_valueerror(self):
        """ANI2xt only supports H,C,N,O,F,S,Cl. A phosphorus-containing
        molecule must raise a clear ValueError naming the element/model,
        not a bare KeyError."""
        # Trimethylphosphine: contains P (atomic number 15).
        mol = Chem.AddHs(Chem.MolFromSmiles("CP(C)C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        mols = [mol]
        device = torch.device("cpu")

        with pytest.raises(ValueError) as exc:
            pad_from_mols(mols, "ANI2xt", device, coord_pad=0.0, species_pad=-1)
        msg = str(exc.value)
        assert "ANI2xt" in msg and ("15" in msg or "P" in msg)
