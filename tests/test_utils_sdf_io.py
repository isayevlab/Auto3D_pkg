"""Tests for Auto3D.utils.sdf_io module."""
from pathlib import Path

import pytest  # noqa: F401  (used by the __main__ guard below)
from rdkit import Chem

from Auto3D.utils.sdf_io import (
    SDF2chunks,
    count_sdf,
    guess_file_type,
    reorder_sdf,
)

# Get the test files directory
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"



def _make_mol(name):
    """Build a tiny named RDKit mol for SDF round-trips."""

    mol = Chem.MolFromSmiles("C")
    mol.SetProp("_Name", name)
    return mol


class TestGuessFileType:
    """Tests for guess_file_type function."""

    def test_smi_extension(self):
        """Test detection of .smi files."""
        assert guess_file_type("molecules.smi") == "smi"
        assert guess_file_type("/path/to/input.smi") == "smi"

    def test_sdf_extension(self):
        """Test detection of .sdf files."""
        assert guess_file_type("molecules.sdf") == "sdf"
        assert guess_file_type("/data/output/result.sdf") == "sdf"

    def test_mol2_extension(self):
        """Test detection of .mol2 files."""
        assert guess_file_type("molecule.mol2") == "mol2"

    def test_xyz_extension(self):
        """Test detection of .xyz files."""
        assert guess_file_type("geometry.xyz") == "xyz"

    def test_complex_path(self):
        """Test with complex file paths."""
        assert guess_file_type("/home/user/data.2024/molecules.sdf") == "sdf"
        assert guess_file_type("./relative/path/file.smi") == "smi"

    def test_no_extension(self):
        """Test file without extension returns empty string."""
        assert guess_file_type("filename") == ""

    def test_hidden_file(self):
        """Test hidden files with extension."""
        assert guess_file_type(".hidden.sdf") == "sdf"


class TestSDF2chunks:
    """Tests for SDF2chunks function."""

    def test_splits_sdf_into_chunks(self):
        """Test that SDF file is split into molecule chunks."""
        sdf_path = str(FILES_DIR / "example.sdf")

        chunks = SDF2chunks(sdf_path)

        # example.sdf has 2 molecules
        assert len(chunks) == 2

        # Each chunk should end with $$$$
        for chunk in chunks:
            assert chunk[-1].strip() == "$$$$"

    def test_chunk_contains_molecule_lines(self):
        """Test that chunks contain all molecule lines."""
        sdf_path = str(FILES_DIR / "example.sdf")

        chunks = SDF2chunks(sdf_path)

        # First chunk should start with molecule name
        assert chunks[0][0].strip() == "mol1"
        assert chunks[1][0].strip() == "mol2"

    def test_preserves_all_content(self):
        """Test that all content from original file is preserved."""
        sdf_path = str(FILES_DIR / "example.sdf")

        chunks = SDF2chunks(sdf_path)

        # Reconstruct file from chunks
        reconstructed = "".join(line for chunk in chunks for line in chunk)

        with open(sdf_path) as f:
            original = f.read()

        assert reconstructed == original


class TestReorderSdf:
    """Tests for reorder_sdf function."""

    def test_reorder_sdf_from_smi(self, tmp_path):
        """Test reordering SDF file based on SMILES file order."""

        # Create source SMILES file with specific order
        smi_file = tmp_path / "source.smi"
        smi_file.write_text("CCO mol_b\nCC mol_a\nCCC mol_c\n")

        # Create SDF file with different order
        sdf_file = tmp_path / "mols.sdf"
        writer = Chem.SDWriter(str(sdf_file))
        for name in ["mol_a", "mol_c", "mol_b"]:
            mol = Chem.MolFromSmiles("C")
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()

        # Reorder
        result = reorder_sdf(str(sdf_file), str(smi_file))

        # Verify order matches source
        assert len(result) == 3
        assert result[0].GetProp("_Name") == "mol_b"
        assert result[1].GetProp("_Name") == "mol_a"
        assert result[2].GetProp("_Name") == "mol_c"

    def test_reorder_sdf_from_sdf(self, tmp_path):
        """Test reordering SDF file based on another SDF file order."""

        # Create source SDF file with specific order
        source_sdf = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source_sdf))
        for name in ["mol_x", "mol_y", "mol_z"]:
            mol = Chem.MolFromSmiles("C")
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()

        # Create target SDF file with different order
        target_sdf = tmp_path / "target.sdf"
        writer = Chem.SDWriter(str(target_sdf))
        for name in ["mol_z", "mol_x", "mol_y"]:
            mol = Chem.MolFromSmiles("C")
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()

        # Reorder
        result = reorder_sdf(str(target_sdf), str(source_sdf))

        # Verify order matches source
        assert len(result) == 3
        assert result[0].GetProp("_Name") == "mol_x"
        assert result[1].GetProp("_Name") == "mol_y"
        assert result[2].GetProp("_Name") == "mol_z"

    def test_reorder_sdf_with_tautomers(self, tmp_path):
        """Test reordering handles tautomer IDs correctly."""

        # Create source SMILES file
        smi_file = tmp_path / "source.smi"
        smi_file.write_text("CCO mol1\nCC mol2\n")

        # Create SDF file with tautomer variants
        sdf_file = tmp_path / "mols.sdf"
        writer = Chem.SDWriter(str(sdf_file))
        for name in ["mol2@taut1", "mol1@taut1", "mol1@taut2"]:
            mol = Chem.MolFromSmiles("C")
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()

        # Reorder
        result = reorder_sdf(str(sdf_file), str(smi_file))

        # Verify mol1 variants come before mol2 variants
        assert len(result) == 3
        # mol1 should be first (2 tautomers)
        assert "mol1" in result[0].GetProp("_Name")
        assert "mol1" in result[1].GetProp("_Name")
        # mol2 should be last
        assert "mol2" in result[2].GetProp("_Name")

    def test_reorder_sdf_unsupported_format(self, tmp_path, caplog):
        """Test that unsupported format returns None."""
        import logging

        xyz_file = tmp_path / "source.xyz"
        xyz_file.write_text("invalid")

        sdf_file = tmp_path / "mols.sdf"
        sdf_file.write_text("dummy")

        with caplog.at_level(logging.WARNING):
            result = reorder_sdf(str(sdf_file), str(xyz_file))

        assert result is None
        assert "Unsupported file format" in caplog.text


class TestNoneMolHardening:
    """FIX 1: None records yielded by SDMolSupplier must not crash these helpers.

    A single unparseable SDF record makes SDMolSupplier yield ``None``. The
    iterating helpers previously called ``mol.GetProp(...)`` / ``mol.GetNumAtoms()``
    on it and raised ``AttributeError``. They must skip ``None`` instead.
    """

    def test_count_sdf_skips_none_records(self, tmp_path, monkeypatch):
        """count_sdf must not count (or crash on) a None record."""

        import Auto3D.utils.sdf_io as sdf_io

        valid = _make_mol("mol_a")
        monkeypatch.setattr(
            sdf_io.Chem, "SDMolSupplier", lambda *a, **k: [valid, None]
        )

        sdf = tmp_path / "mols.sdf"
        sdf.write_text("placeholder")  # path only needs to exist for the call

        assert count_sdf(str(sdf)) == 1
    def test_reorder_sdf_skips_none_records(self, tmp_path, monkeypatch):
        """reorder_sdf must skip None records in the target SDF."""

        import Auto3D.utils.sdf_io as sdf_io

        smi = tmp_path / "source.smi"
        smi.write_text("C mol_a\nC mol_b\n")

        valid_a = _make_mol("mol_a")
        valid_b = _make_mol("mol_b")
        monkeypatch.setattr(
            sdf_io.Chem, "SDMolSupplier", lambda *a, **k: [valid_a, None, valid_b]
        )

        sdf = tmp_path / "target.sdf"
        sdf.write_text("placeholder")

        result = reorder_sdf(str(sdf), str(smi))
        names = [m.GetProp("_Name") for m in result]
        assert names == ["mol_a", "mol_b"]


class TestReorderSdfDataPreservation:
    """FIX 2: reorder_sdf must not drop unmatched molecules or truncate input."""

    def test_unmatched_mol_is_preserved(self, tmp_path):
        """A mol whose id is not in the source must still survive to disk."""

        smi = tmp_path / "source.smi"
        # source lists only mol_a and mol_b; mol_c is unmatched.
        smi.write_text("C mol_a\nC mol_b\n")

        sdf = tmp_path / "target.sdf"
        writer = Chem.SDWriter(str(sdf))
        for name in ["mol_c", "mol_b", "mol_a"]:
            writer.write(_make_mol(name))
        writer.close()

        result = reorder_sdf(str(sdf), str(smi))

        # All three mols preserved (no silent data loss).
        result_names = [m.GetProp("_Name") for m in result]
        assert set(result_names) == {"mol_a", "mol_b", "mol_c"}
        assert len(result_names) == 3
        # Matched ids appear first, in source order.
        assert result_names[0] == "mol_a"
        assert result_names[1] == "mol_b"

        # And the on-disk file must contain all three as well.
        on_disk = [m.GetProp("_Name") for m in Chem.SDMolSupplier(str(sdf))]
        assert set(on_disk) == {"mol_a", "mol_b", "mol_c"}

    def test_normal_all_matched_ordering_unchanged(self, tmp_path):
        """When every id is matched, ordering is exactly the source order."""

        smi = tmp_path / "source.smi"
        smi.write_text("C mol_b\nC mol_a\nC mol_c\n")

        sdf = tmp_path / "target.sdf"
        writer = Chem.SDWriter(str(sdf))
        for name in ["mol_a", "mol_c", "mol_b"]:
            writer.write(_make_mol(name))
        writer.close()

        result = reorder_sdf(str(sdf), str(smi))
        names = [m.GetProp("_Name") for m in result]
        assert names == ["mol_b", "mol_a", "mol_c"]


class TestSDF2chunksTrailingRecord:
    """FIX 3: a final record lacking the $$$$ terminator must not be dropped."""

    def test_trailing_record_without_terminator_preserved(self, tmp_path):
        """SDF2chunks keeps a terminator-less trailing record as its own chunk."""
        sdf = tmp_path / "ragged.sdf"
        # First record has $$$$; second record lacks it.
        sdf.write_text(
            "mol1\n  line1\n$$$$\n"
            "mol2\n  line2\n  line3\n"
        )

        chunks = SDF2chunks(str(sdf))

        assert len(chunks) == 2
        assert chunks[0][0].strip() == "mol1"
        # The trailing record's lines must be present in the final chunk.
        assert chunks[1][0].strip() == "mol2"
        joined = "".join(chunks[1])
        assert "line2" in joined
        assert "line3" in joined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
