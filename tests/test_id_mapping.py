"""Tests for Auto3D.id_mapping module (encode_ids / decode_ids)."""
from pathlib import Path

import pytest
from rdkit import Chem  # noqa: F401  (several tests import it locally too)

from Auto3D.id_mapping import decode_ids, encode_ids
from Auto3D.utils.sdf_io import count_sdf

# Get the test files directory
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"


class TestEncodeIdsSmiDenseIndex:
    """encode_ids must use a dense, gap-free index for .smi inputs."""

    def test_blank_lines_do_not_create_index_gaps(self, tmp_path):
        smi = tmp_path / "in.smi"
        # Blank lines interspersed: the old code used the file line number as the
        # index, leaving gaps. The dense record counter must yield 0,1,2.
        smi.write_text("CCO a\n\nCCC b\n\n\nCCCC c\n")

        new_path, mapping = encode_ids(str(smi))

        assert sorted(mapping.values()) == [0, 1, 2]
        ids = [line.split()[1] for line in Path(new_path).read_text().strip().split("\n")]
        assert ids == ["0", "1", "2"]


class TestEncodeDecodeIds:
    """Tests for encode_ids and decode_ids functions."""

    def test_encode_smi_file(self, tmp_path):
        """Test encoding IDs in a SMILES file."""
        input_file = tmp_path / "input.smi"
        input_file.write_text("CCO mol_alpha\nCC mol_beta\nCCC mol_gamma\n")

        new_path, mapping = encode_ids(str(input_file))

        assert mapping == {"mol_alpha": 0, "mol_beta": 1, "mol_gamma": 2}
        assert Path(new_path).name == "input_encoded.smi"

        # Check encoded file content
        content = Path(new_path).read_text()
        assert "CCO 0" in content
        assert "CC 1" in content
        assert "CCC 2" in content

    def test_encode_sdf_file(self):
        """Test encoding IDs in an SDF file."""
        sdf_path = str(FILES_DIR / "example.sdf")

        new_path, mapping = encode_ids(sdf_path)

        assert "mol1" in mapping
        assert "mol2" in mapping
        assert mapping["mol1"] == 0
        assert mapping["mol2"] == 1

        # Clean up
        Path(new_path).unlink(missing_ok=True)

    def test_encode_invalid_extension_raises(self, tmp_path):
        """Test that invalid file extension raises ValueError."""
        input_file = tmp_path / "input.xyz"
        input_file.write_text("invalid")

        with pytest.raises(ValueError, match="smi or sdf"):
            encode_ids(str(input_file))

    def test_encode_skips_blank_lines(self, tmp_path):
        """Test that blank lines in SMILES file are skipped."""
        input_file = tmp_path / "input.smi"
        input_file.write_text("CCO mol1\n\n   \nCC mol2\n")

        new_path, mapping = encode_ids(str(input_file))

        # mapping indices may not be sequential if blank lines are in between
        assert len(mapping) == 2

        # Clean up
        Path(new_path).unlink(missing_ok=True)

    def test_encode_ids_rejects_duplicate_ids(self, tmp_path):
        """Duplicate molecule IDs in a .smi file are rejected up front."""
        from Auto3D.exceptions import InputValidationError

        p = tmp_path / "dup.smi"
        p.write_text("CCO mol1\nCCC mol1\n")
        with pytest.raises(InputValidationError, match="[Dd]uplicate"):
            encode_ids(str(p))

    def test_encode_ids_rejects_missing_id(self, tmp_path):
        """A .smi row without a whitespace-separated ID is rejected."""
        from Auto3D.exceptions import InputValidationError

        p = tmp_path / "noid.smi"
        p.write_text("CCO\n")  # no whitespace-separated ID
        with pytest.raises(InputValidationError, match="ID"):
            encode_ids(str(p))

    def test_encode_ids_roundtrip_unique(self, tmp_path):
        """Unique IDs encode cleanly and appear in the mapping."""
        p = tmp_path / "ok.smi"
        p.write_text("CCO a\nCCC b\n")
        _, mapping = encode_ids(str(p))
        assert set(mapping) == {"a", "b"}

    def test_encode_ids_rejects_blank_sdf_name(self, tmp_path):
        """A molecule with a blank _Name in a .sdf file is rejected."""
        from rdkit.Chem import AllChem

        from Auto3D.exceptions import InputValidationError

        sdf = tmp_path / "blank.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", "")  # blank name
            w.write(m)
        with pytest.raises(InputValidationError):
            encode_ids(str(sdf))

    def test_encode_ids_refuses_to_overwrite_an_existing_file(self, tmp_path):
        """The `<stem>_encoded.<ext>` name belongs to the user until proven
        otherwise.

        The name is derived from the input, so `mols_encoded.smi` beside
        `mols.smi` is an ordinary thing for a user to own -- and this function
        used to open it for writing without a word. `WorkflowOrchestrator`
        now redirects the encoded copy into its own job directory (see
        `out_dir` below), but this check keeps the guarantee attached to the
        function itself, so a caller taking the default location cannot
        reintroduce the defect.

        The refusal now comes from the shared
        `Auto3D.utils.output_guard.check_output_overwrite` (hence the
        "already exists" wording) rather than a bespoke message here, so
        `encode_ids`, `decode_ids` and `tautomer.select_tautomers` state the
        same policy the same way -- and `overwrite=True` can lift it, which the
        unconditional refusal this replaced offered no way to do.
        `tests/test_output_overwrite_gates.py` covers both directions.
        """
        from Auto3D.exceptions import ConfigurationError

        p = tmp_path / "mols.smi"
        p.write_text("CCO a\n")
        users_file = tmp_path / "mols_encoded.smi"
        users_file.write_bytes(b"IRREPLACEABLE USER DATA\n")

        with pytest.raises(ConfigurationError, match="already exists"):
            encode_ids(str(p))

        assert users_file.read_bytes() == b"IRREPLACEABLE USER DATA\n"

    def test_encode_ids_writes_into_out_dir_when_given_one(self, tmp_path):
        """`out_dir` moves the encoded copy somewhere the caller owns.

        This is how the run pipeline avoids the collision above entirely: it
        passes the job directory it just created. The file name is unchanged,
        only its directory -- downstream code (`_setup_job_directory`,
        `decode_ids`) parses that name.
        """
        p = tmp_path / "mols.smi"
        p.write_text("CCO a\nCCC b\n")
        staging = tmp_path / "staging"
        staging.mkdir()

        new_path, mapping = encode_ids(str(p), out_dir=staging)

        assert Path(new_path).parent == staging
        assert Path(new_path).name == "mols_encoded.smi"
        assert mapping == {"a": 0, "b": 1}
        assert not (tmp_path / "mols_encoded.smi").exists()


class TestNoneMolHardening:
    """A None record yielded by SDMolSupplier must not crash decode_ids."""

    def test_decode_ids_skips_none_records(self, tmp_path, monkeypatch):
        """decode_ids must skip None records without raising."""

        import Auto3D.id_mapping as id_mapping

        valid = Chem.MolFromSmiles("C")
        valid.SetProp("_Name", "0")
        valid.SetProp("ID", "0_conf1")

        monkeypatch.setattr(
            id_mapping.Chem, "SDMolSupplier", lambda *a, **k: [valid, None]
        )

        # decode_ids expects a stem with at least two underscore parts.
        sdf = tmp_path / "mols_3d_encoded.sdf"
        sdf.write_text("placeholder")

        out = decode_ids(str(sdf), {"mol_a": 0})
        # Only the valid record is written; no AttributeError on the None.
        written = count_sdf(out)
        assert written == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
