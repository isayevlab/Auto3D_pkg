"""Tests for Auto3D.utils.smi_io module."""

from pathlib import Path

import pytest
from rdkit import Chem  # noqa: F401  (kept for parity with the sibling io tests)

from Auto3D.utils.smi_io import (
    combine_smi,
    hash_enumerated_smi_IDs,
    hash_taut_smi,
    iter_smi_records,
    smiles2smi,
)

# Get the test files directory
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"


class TestSmiles2Smi:
    """Tests for smiles2smi function."""

    def test_creates_file_with_inchikeys(self, tmp_path):
        """smiles2smi should create a .smi file with SMILES and InChIKey IDs."""
        smiles = ["CCO", "CCC"]
        output = tmp_path / "test.smi"

        result = smiles2smi(smiles, str(output))

        assert result == str(output)
        assert output.exists()
        content = output.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        # Each line should have SMILES and InChIKey
        for line in lines:
            parts = line.split()
            assert len(parts) == 2

    def test_returns_output_path(self, tmp_path):
        """smiles2smi should return the output file path."""
        smiles = ["CCO"]
        output = tmp_path / "output.smi"

        result = smiles2smi(smiles, str(output))

        assert result == str(output)

    def test_inchikey_format(self, tmp_path):
        """InChIKeys should have the standard 27-character format."""
        smiles = ["CCO"]
        output = tmp_path / "test.smi"

        smiles2smi(smiles, str(output))

        content = output.read_text().strip()
        parts = content.split()
        inchikey = parts[1]
        # InChIKey format: 14 chars + hyphen + 10 chars + hyphen + 1 char = 27 chars
        assert len(inchikey) == 27
        assert inchikey.count("-") == 2

    def test_preserves_smiles_string(self, tmp_path):
        """Original SMILES strings should be preserved in output."""
        smiles = ["C#N", "C=C", "[NH4+]"]
        output = tmp_path / "test.smi"

        smiles2smi(smiles, str(output))

        content = output.read_text()
        for smi in smiles:
            assert smi in content

    def test_empty_list(self, tmp_path):
        """Empty input list should create empty file."""
        output = tmp_path / "test.smi"

        result = smiles2smi([], str(output))

        assert result == str(output)
        assert output.exists()
        assert output.read_text() == ""

    def test_colliding_inchikeys_get_distinct_ids(self, tmp_path):
        """Two inputs that share an InChIKey must keep distinct IDs.

        The same molecule written two ways (here benzene) yields one InChIKey;
        without disambiguation reorder_sdf would collapse the duplicate IDs and
        silently drop the second input. Each input must get its own line/ID.
        """
        output = tmp_path / "test.smi"

        smiles2smi(["c1ccccc1", "C1=CC=CC=C1"], str(output))

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        ids = [line.split()[1] for line in lines]
        assert ids[0] != ids[1], "colliding InChIKeys must be disambiguated"
        # First keeps the bare InChIKey; the repeat is suffixed.
        assert ids[1] == f"{ids[0]}_2"

    def test_distinct_inputs_keep_bare_inchikeys(self, tmp_path):
        """Non-colliding inputs must keep their plain InChIKey IDs (no suffix)."""
        output = tmp_path / "test.smi"

        smiles2smi(["CCO", "CCC"], str(output))

        ids = [line.split()[1] for line in output.read_text().strip().split("\n")]
        assert ids[0] != ids[1]
        assert all("_" not in i for i in ids)


class TestCombineSmiOrderPreservingDedup:
    """Tests for combine_smi (order-preserving dedup).

    Named distinctly from the ``TestCombineSmi`` class below -- both defined
    ``TestCombineSmi`` until a lint audit found the second definition was
    silently shadowing this one, so pytest never collected the test below.
    """

    def test_preserves_order_and_dedups(self, tmp_path):
        f1 = tmp_path / "a.smi"
        f2 = tmp_path / "b.smi"
        f1.write_text("CCO ethanol\nCCC propane\n")
        f2.write_text("CCC propane\nCCCC butane\n")  # propane duplicated
        out = tmp_path / "combined.smi"

        combine_smi([str(f1), str(f2)], str(out))

        lines = out.read_text().strip().split("\n")
        # Deduped (propane once) and in first-seen input order.
        assert lines == ["CCO ethanol", "CCC propane", "CCCC butane"]


class TestHashEnumeratedSmiIDs:
    """Tests for hash_enumerated_smi_IDs function."""

    def test_basic_hashing(self, tmp_path):
        """Test basic hashing with simple SMILES file."""
        input_file = tmp_path / "input.smi"
        output_file = tmp_path / "output.smi"

        # Create input file with unsorted IDs
        input_file.write_text("CCO mol_b\nCC mol_a\nCCC mol_c\n")

        hash_enumerated_smi_IDs(str(input_file), str(output_file))

        # Read and verify output
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3
        # Should be sorted by ID
        assert "mol_a" in lines[0]
        assert "mol_b" in lines[1]
        assert "mol_c" in lines[2]

    def test_duplicate_id_handling(self, tmp_path):
        """Test that duplicate IDs get '_0' suffix."""
        input_file = tmp_path / "input.smi"
        output_file = tmp_path / "output.smi"

        # Create input file with duplicate IDs
        input_file.write_text("CCO mol1\nCC mol1\nCCC mol1\n")

        hash_enumerated_smi_IDs(str(input_file), str(output_file))

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Check that duplicates were renamed
        ids = [line.split()[1] for line in lines]
        assert "mol1" in ids
        assert "mol1_0" in ids
        assert "mol1_0_0" in ids

    def test_preserves_smiles(self, tmp_path):
        """Test that SMILES strings are preserved correctly."""
        input_file = tmp_path / "input.smi"
        output_file = tmp_path / "output.smi"

        input_file.write_text("C#N id1\nC=C id2\n")

        hash_enumerated_smi_IDs(str(input_file), str(output_file))

        content = output_file.read_text()
        assert "C#N" in content
        assert "C=C" in content


class TestHashTautSmi:
    """Tests for hash_taut_smi function."""

    def test_tautomer_suffix_added(self, tmp_path):
        """Test that @taut suffix is added to IDs."""
        input_file = tmp_path / "input.smi"
        output_file = tmp_path / "output.smi"

        input_file.write_text("CCO mol1\nCC mol2\n")

        hash_taut_smi(str(input_file), str(output_file))

        content = output_file.read_text()
        assert "@taut" in content

    def test_incremental_taut_suffix(self, tmp_path):
        """Test that duplicate base IDs get incrementing taut numbers."""
        input_file = tmp_path / "input.smi"
        output_file = tmp_path / "output.smi"

        # Same ID for multiple SMILES
        input_file.write_text("CCO mol1\nCC mol1\n")

        hash_taut_smi(str(input_file), str(output_file))

        lines = output_file.read_text().strip().split("\n")
        ids = [line.split()[1] for line in lines]

        # Should have different taut numbers
        assert len(set(ids)) == 2
        assert all("@taut" in id for id in ids)


class TestCombineSmi:
    """Tests for combine_smi function."""

    def test_combines_files(self, tmp_path):
        """Test that multiple SMILES files are combined."""
        file1 = tmp_path / "file1.smi"
        file2 = tmp_path / "file2.smi"
        output = tmp_path / "combined.smi"

        file1.write_text("CCO mol1\nCC mol2\n")
        file2.write_text("CCC mol3\nCCCC mol4\n")

        combine_smi([str(file1), str(file2)], str(output))

        content = output.read_text()
        assert "mol1" in content
        assert "mol2" in content
        assert "mol3" in content
        assert "mol4" in content

    def test_removes_duplicates(self, tmp_path):
        """Test that duplicate entries are removed."""
        file1 = tmp_path / "file1.smi"
        file2 = tmp_path / "file2.smi"
        output = tmp_path / "combined.smi"

        file1.write_text("CCO mol1\n")
        file2.write_text("CCO mol1\n")  # Same entry

        combine_smi([str(file1), str(file2)], str(output))

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_ignores_blank_lines(self, tmp_path):
        """Test that blank lines are ignored."""
        file1 = tmp_path / "file1.smi"
        output = tmp_path / "combined.smi"

        file1.write_text("CCO mol1\n\n\nCC mol2\n   \n")

        combine_smi([str(file1)], str(output))

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2


class TestSmiles2SmiInvalidInput:
    """FIX 4: smiles2smi must raise a clear error on an invalid SMILES."""

    def test_invalid_smiles_raises_input_validation_error(self, tmp_path):
        """An unparseable SMILES raises InputValidationError naming the SMILES."""
        from Auto3D.exceptions import InputValidationError

        out = tmp_path / "out.smi"
        with pytest.raises(InputValidationError, match=r"C\(C"):
            smiles2smi(["CCO", "C(C"], str(out))


class TestHashHelpersBlankLines:
    """FIX 5: blank / malformed lines must not crash the hashing helpers."""

    def test_hash_enumerated_skips_blank_and_extra_token_lines(self, tmp_path):
        """hash_enumerated_smi_IDs tolerates blank lines and extra tokens."""
        inp = tmp_path / "in.smi"
        inp.write_text("CCO mol1\n\n   \nCC mol2 extra_token\n")
        out = tmp_path / "out.smi"

        # Must not raise ValueError.
        hash_enumerated_smi_IDs(str(inp), str(out))

        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        ids = [ln.split()[1] for ln in lines]
        assert "mol1" in ids
        assert "mol2" in ids

    def test_hash_taut_skips_blank_and_extra_token_lines(self, tmp_path):
        """hash_taut_smi tolerates blank lines and extra tokens."""
        inp = tmp_path / "in.smi"
        inp.write_text("CCO mol1\n\nCC mol2 extra_token\n")
        out = tmp_path / "out.smi"

        hash_taut_smi(str(inp), str(out))

        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        assert all("@taut" in ln.split()[1] for ln in lines)


class TestIterSmiRecords:
    """FIX A: shared lenient .smi parser used by all 7 call sites."""

    def test_blank_lines_skipped(self, tmp_path):
        """Blank and whitespace-only lines yield no records."""
        p = tmp_path / "in.smi"
        p.write_text("CCO mol1\n\n   \nCC mol2\n")
        records = list(iter_smi_records(str(p)))
        assert [(s, i) for _ln, s, i in records] == [("CCO", "mol1"), ("CC", "mol2")]
        # line_no is 1-based and reflects the original line position.
        assert records[0][0] == 1
        assert records[1][0] == 4

    def test_three_token_line_yields_first_two(self, tmp_path):
        """A 3-token line yields only the first two tokens (extras ignored)."""
        p = tmp_path / "in.smi"
        p.write_text("CCN extra_a extra_b\n")
        records = list(iter_smi_records(str(p)))
        assert len(records) == 1
        line_no, smiles, mol_id = records[0]
        assert (smiles, mol_id) == ("CCN", "extra_a")

    def test_on_malformed_skip_skips_one_token_line_with_warning(self, tmp_path, caplog):
        """on_malformed='skip' (default) skips a 1-token line and warns."""
        import logging

        p = tmp_path / "in.smi"
        p.write_text("CCO mol1\nC1CCCCC1\nCC mol2\n")
        with caplog.at_level(logging.WARNING):
            records = list(iter_smi_records(str(p), on_malformed="skip"))
        assert [(s, i) for _ln, s, i in records] == [("CCO", "mol1"), ("CC", "mol2")]
        assert any("failed to parse" in r.message for r in caplog.records)

    def test_on_malformed_raise_raises_on_one_token_line(self, tmp_path):
        """on_malformed='raise' raises InputValidationError naming the line."""
        from Auto3D.exceptions import InputValidationError

        p = tmp_path / "in.smi"
        p.write_text("CCO mol1\nC1CCCCC1\n")
        with pytest.raises(InputValidationError, match="Line 2"):
            list(iter_smi_records(str(p), on_malformed="raise"))

    def test_invalid_on_malformed_value_raises(self, tmp_path):
        """An unknown on_malformed value raises ValueError."""
        p = tmp_path / "in.smi"
        p.write_text("CCO mol1\n")
        with pytest.raises(ValueError, match="on_malformed"):
            list(iter_smi_records(str(p), on_malformed="bogus"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
