"""Unit tests for the io/repositories module."""
from __future__ import annotations

import pytest
from rdkit import Chem

from Auto3D.io import (
    MoleculeRepository,
    SDFRepository,
    SMIRepository,
    read_molecules,
    write_molecules,
)


class TestSDFRepository:
    """Tests for SDFRepository class."""

    def test_implements_protocol(self):
        """Test that SDFRepository implements MoleculeRepository protocol."""
        repo = SDFRepository()
        assert isinstance(repo, MoleculeRepository)

    def test_read_sdf_file(self, tmp_path):
        """Test reading molecules from SDF file."""
        # Create a simple SDF file
        sdf_content = """
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
"""
        sdf_file = tmp_path / "test.sdf"
        sdf_file.write_text(sdf_content)

        repo = SDFRepository()
        mols = list(repo.read(str(sdf_file)))

        assert len(mols) == 1
        assert mols[0].GetNumAtoms() == 3

    def test_write_sdf_file(self, tmp_path):
        """Test writing molecules to SDF file."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)

        sdf_file = tmp_path / "output.sdf"
        repo = SDFRepository()
        repo.write(str(sdf_file), [mol])

        assert sdf_file.exists()

        # Read back and verify
        mols = list(repo.read(str(sdf_file)))
        assert len(mols) == 1

    def test_read_with_properties(self, tmp_path):
        """Test reading SDF with properties."""
        # Create SDF with properties
        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("_Name", "ethanol")
        mol.SetProp("test_prop", "test_value")

        sdf_file = tmp_path / "props.sdf"
        with Chem.SDWriter(str(sdf_file)) as w:
            w.write(mol)

        repo = SDFRepository()
        results = list(repo.read_with_properties(str(sdf_file)))

        assert len(results) == 1
        mol, props = results[0]
        assert "test_prop" in props
        assert props["test_prop"] == "test_value"


class TestSMIRepository:
    """Tests for SMIRepository class."""

    def test_implements_protocol(self):
        """Test that SMIRepository implements MoleculeRepository protocol."""
        repo = SMIRepository()
        assert isinstance(repo, MoleculeRepository)

    def test_read_smi_file(self, tmp_path):
        """Test reading molecules from SMILES file."""
        smi_content = "CCO ethanol\nCCC propane\n"
        smi_file = tmp_path / "test.smi"
        smi_file.write_text(smi_content)

        repo = SMIRepository()
        mols = list(repo.read(str(smi_file)))

        assert len(mols) == 2
        assert mols[0].GetProp("_Name") == "ethanol"
        assert mols[1].GetProp("_Name") == "propane"

    def test_write_smi_file(self, tmp_path):
        """Test writing molecules to SMILES file."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol1.SetProp("_Name", "ethanol")
        mol2 = Chem.MolFromSmiles("CCC")
        mol2.SetProp("_Name", "propane")

        smi_file = tmp_path / "output.smi"
        repo = SMIRepository()
        repo.write(str(smi_file), [mol1, mol2])

        assert smi_file.exists()
        content = smi_file.read_text()
        assert "ethanol" in content
        assert "propane" in content

    def test_read_raw(self, tmp_path):
        """Test reading raw SMILES strings."""
        smi_content = "CCO ethanol\nCCC propane\n"
        smi_file = tmp_path / "test.smi"
        smi_file.write_text(smi_content)

        repo = SMIRepository()
        results = list(repo.read_raw(str(smi_file)))

        assert len(results) == 2
        assert results[0] == ("CCO", "ethanol")
        assert results[1] == ("CCC", "propane")

    def test_write_raw(self, tmp_path):
        """Test writing raw SMILES strings."""
        data = [("CCO", "ethanol"), ("CCC", "propane")]

        smi_file = tmp_path / "output.smi"
        repo = SMIRepository()
        repo.write_raw(str(smi_file), data)

        assert smi_file.exists()
        content = smi_file.read_text()
        assert "CCO\tethanol" in content
        assert "CCC\tpropane" in content


class TestConvenienceFunctions:
    """Tests for read_molecules and write_molecules functions."""

    def test_read_molecules_sdf(self, tmp_path):
        """Test read_molecules with SDF file."""
        mol = Chem.MolFromSmiles("CCO")
        sdf_file = tmp_path / "test.sdf"
        with Chem.SDWriter(str(sdf_file)) as w:
            w.write(mol)

        mols = list(read_molecules(str(sdf_file)))
        assert len(mols) == 1

    def test_read_molecules_smi(self, tmp_path):
        """Test read_molecules with SMILES file."""
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol\n")

        mols = list(read_molecules(str(smi_file)))
        assert len(mols) == 1

    def test_read_molecules_unsupported_format(self, tmp_path):
        """Test read_molecules raises error for unsupported format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            list(read_molecules(str(txt_file)))

    def test_write_molecules_sdf(self, tmp_path):
        """Test write_molecules with SDF file."""
        mol = Chem.MolFromSmiles("CCO")
        sdf_file = tmp_path / "output.sdf"

        write_molecules(str(sdf_file), [mol])
        assert sdf_file.exists()

    def test_write_molecules_smi(self, tmp_path):
        """Test write_molecules with SMILES file."""
        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("_Name", "ethanol")
        smi_file = tmp_path / "output.smi"

        write_molecules(str(smi_file), [mol])
        assert smi_file.exists()

    def test_write_molecules_unsupported_format(self, tmp_path):
        """Test write_molecules raises error for unsupported format."""
        mol = Chem.MolFromSmiles("CCO")
        txt_file = tmp_path / "output.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            write_molecules(str(txt_file), [mol])
