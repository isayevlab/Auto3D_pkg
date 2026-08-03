"""Tests for Auto3D.utils.file_ops module."""
from pathlib import Path

import pytest
from rdkit import Chem

from Auto3D.utils.file_ops import (
    smiles2smi,
    guess_file_type,
    hash_enumerated_smi_IDs,
    hash_taut_smi,
    housekeeping,
    create_chunk_meta_names,
    combine_smi,
    SDF2chunks,
    encode_ids,
    decode_ids,
    reorder_sdf,
    count_sdf,
    find_ids_not_in_sdf,
    find_smiles_not_in_sdf,
    iter_smi_records,
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
        lines = content.strip().split('\n')
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
        assert inchikey.count('-') == 2

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


class TestHousekeeping:
    """Tests for housekeeping function."""

    def test_moves_files_except_output(self, tmp_path):
        """Test that files are moved except for the output file."""
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        verbose_folder = tmp_path / "verbose"
        verbose_folder.mkdir()

        # Create test files
        (job_dir / "meta1.txt").write_text("meta1")
        (job_dir / "meta2.txt").write_text("meta2")
        output_file = job_dir / "output.sdf"
        output_file.write_text("output")

        housekeeping(str(job_dir), str(verbose_folder), str(output_file))

        # Output should still be in job_dir
        assert output_file.exists()
        # Meta files should be moved
        assert (verbose_folder / "meta1.txt").exists()
        assert (verbose_folder / "meta2.txt").exists()


class TestCreateChunkMetaNames:
    """Tests for create_chunk_meta_names function."""

    def test_generates_expected_paths(self):
        """Test that all expected paths are generated."""
        result = create_chunk_meta_names("chunk1.smi", "/tmp/job")

        assert result["output"] == "/tmp/job/chunk1_3d.sdf"
        assert result["optimized_og"] == "/tmp/job/chunk1_3d0.sdf"
        assert result["output_taut"] == "/tmp/job/smi_taut.smi"
        assert result["smiles_enumerated"] == "/tmp/job/smiles_enumerated.smi"
        assert result["smiles_reduced"] == "/tmp/job/smiles_enumerated_reduced.smi"
        assert result["smiles_hashed"] == "/tmp/job/smiles_enumerated_hashed.smi"
        assert result["enumerated_sdf"] == "/tmp/job/smiles_enumerated.sdf"
        assert result["sorted_sdf"] == "/tmp/job/enumerated_sorted.sdf"
        assert result["housekeeping_folder"] == "/tmp/job/verbose"
        assert result["path"] == "chunk1.smi"
        assert result["dir"] == "/tmp/job"

    def test_handles_path_with_directory(self):
        """Test that paths with directories work correctly."""
        result = create_chunk_meta_names("/data/input/chunk1.smi", "/output/job")

        assert result["output"] == "/output/job/chunk1_3d.sdf"
        assert result["path"] == "/data/input/chunk1.smi"


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
        from rdkit import Chem
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
        """
        from Auto3D.exceptions import ConfigurationError

        p = tmp_path / "mols.smi"
        p.write_text("CCO a\n")
        users_file = tmp_path / "mols_encoded.smi"
        users_file.write_bytes(b"IRREPLACEABLE USER DATA\n")

        with pytest.raises(ConfigurationError, match="would overwrite"):
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


class TestReorderSdf:
    """Tests for reorder_sdf function."""

    def test_reorder_sdf_from_smi(self, tmp_path):
        """Test reordering SDF file based on SMILES file order."""
        from rdkit import Chem

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
        from rdkit import Chem

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
        from rdkit import Chem

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


class TestFileOpsIntegration:
    """Integration tests for file_ops module."""

    def test_create_chunks_and_housekeeping_workflow(self, tmp_path):
        """Test a typical workflow using multiple file_ops functions."""
        # Create job directory structure
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        # Create meta names
        meta = create_chunk_meta_names("input.smi", str(job_dir))

        # Verify structure
        assert "verbose" in meta["housekeeping_folder"]

        # Create verbose folder
        Path(meta["housekeeping_folder"]).mkdir()

        # Create some intermediate files
        Path(meta["smiles_enumerated"]).write_text("CCO mol1\n")
        Path(meta["output"]).write_text("fake sdf output")

        # Run housekeeping - should move enumerated but not output
        housekeeping(
            str(job_dir),
            meta["housekeeping_folder"],
            meta["output"]
        )

        # Output should still exist
        assert Path(meta["output"]).exists()
        # Enumerated should be moved to verbose folder
        assert (Path(meta["housekeeping_folder"]) / "smiles_enumerated.smi").exists()


def _make_mol(name):
    """Build a tiny named RDKit mol for SDF round-trips."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C")
    mol.SetProp("_Name", name)
    return mol


class TestNoneMolHardening:
    """FIX 1: None records yielded by SDMolSupplier must not crash these helpers.

    A single unparseable SDF record makes SDMolSupplier yield ``None``. The
    iterating helpers previously called ``mol.GetProp(...)`` / ``mol.GetNumAtoms()``
    on it and raised ``AttributeError``. They must skip ``None`` instead.
    """

    def test_count_sdf_skips_none_records(self, tmp_path, monkeypatch):
        """count_sdf must not count (or crash on) a None record."""

        import Auto3D.utils.file_ops as file_ops

        valid = _make_mol("mol_a")
        monkeypatch.setattr(
            file_ops.Chem, "SDMolSupplier", lambda *a, **k: [valid, None]
        )

        sdf = tmp_path / "mols.sdf"
        sdf.write_text("placeholder")  # path only needs to exist for the call

        assert count_sdf(str(sdf)) == 1

    def test_decode_ids_skips_none_records(self, tmp_path, monkeypatch):
        """decode_ids must skip None records without raising."""
        from rdkit import Chem

        import Auto3D.utils.file_ops as file_ops

        valid = Chem.MolFromSmiles("C")
        valid.SetProp("_Name", "0")
        valid.SetProp("ID", "0_conf1")

        monkeypatch.setattr(
            file_ops.Chem, "SDMolSupplier", lambda *a, **k: [valid, None]
        )

        # decode_ids expects a stem with at least two underscore parts.
        sdf = tmp_path / "mols_3d_encoded.sdf"
        sdf.write_text("placeholder")

        out = decode_ids(str(sdf), {"mol_a": 0})
        # Only the valid record is written; no AttributeError on the None.
        written = count_sdf(out)
        assert written == 1

    def test_find_smiles_not_in_sdf_skips_none_records(self, tmp_path, monkeypatch):
        """find_smiles_not_in_sdf must skip None SDF records."""
        from rdkit import Chem

        import Auto3D.utils.file_ops as file_ops

        valid = Chem.MolFromSmiles("C")
        valid.SetProp("_Name", "mol_a")
        monkeypatch.setattr(
            file_ops.Chem, "SDMolSupplier", lambda *a, **k: [valid, None]
        )

        smi = tmp_path / "in.smi"
        smi.write_text("C mol_a\nCC mol_b\n")
        sdf = tmp_path / "out.sdf"
        sdf.write_text("placeholder")

        bad = find_smiles_not_in_sdf(str(smi), str(sdf))
        # mol_a is present (valid mol), mol_b is missing -> reported.
        assert ("mol_b", "CC") in bad
        assert all(mol_id != "mol_a" for mol_id, _ in bad)

    def test_reorder_sdf_skips_none_records(self, tmp_path, monkeypatch):
        """reorder_sdf must skip None records in the target SDF."""

        import Auto3D.utils.file_ops as file_ops

        smi = tmp_path / "source.smi"
        smi.write_text("C mol_a\nC mol_b\n")

        valid_a = _make_mol("mol_a")
        valid_b = _make_mol("mol_b")
        monkeypatch.setattr(
            file_ops.Chem, "SDMolSupplier", lambda *a, **k: [valid_a, None, valid_b]
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
        from rdkit import Chem

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
        from rdkit import Chem

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

    def test_find_smiles_not_in_sdf_tolerates_blank_and_3token_lines(
        self, tmp_path, monkeypatch
    ):
        """find_smiles_not_in_sdf tolerates blank and 3-token .smi lines."""
        from rdkit import Chem

        import Auto3D.utils.file_ops as file_ops

        valid = Chem.MolFromSmiles("C")
        valid.SetProp("_Name", "mol_a")
        monkeypatch.setattr(
            file_ops.Chem, "SDMolSupplier", lambda *a, **k: [valid]
        )

        smi = tmp_path / "in.smi"
        # blank line + a 3-token line (first two tokens taken).
        smi.write_text("C mol_a\n\nCC mol_b extra\n")
        sdf = tmp_path / "out.sdf"
        sdf.write_text("placeholder")

        bad = find_smiles_not_in_sdf(str(smi), str(sdf))
        assert ("mol_b", "CC") in bad


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

    def test_on_malformed_skip_skips_one_token_line_with_warning(
        self, tmp_path, caplog
    ):
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


def test_housekeeping_sweep_is_per_file_robust(tmp_path, monkeypatch):
    """One unmovable file must not abandon the rest of the sweep.

    This guard used to live on a second loop that swept `oeomega_*` out of the
    *process working directory*; that loop is gone (it destroyed user files --
    see `TestHousekeepingStaysInsideTheJobDirectory` in tests/test_durability.py)
    and the OpenEye logfiles it collected now land inside the job directory,
    where this loop picks them up. The robustness property moved with them: a
    permission error, or a file that vanished under us, must leave a complete
    `verbose` folder minus that one file rather than a half-populated one plus
    a traceback out of `optim_rank_wrapper`'s blanket except.
    """
    import os

    from Auto3D.utils.file_ops import housekeeping

    job = tmp_path / "job"
    job.mkdir()
    dest = tmp_path / "verbose"
    dest.mkdir()

    # Two logfiles in the job directory; the FIRST one encountered (by
    # counter) will fail to move.
    (job / "oeomega_a.log").write_text("a")
    (job / "oeomega_b.log").write_text("b")

    real_move = __import__("shutil").move
    call_count = {"n": 0}

    def flaky_move(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the file having gone away underneath the sweep.
            if os.path.exists(src):
                os.remove(src)
            raise OSError("already gone")
        return real_move(src, dst)

    monkeypatch.setattr("Auto3D.utils.file_ops.shutil.move", flaky_move)

    housekeeping(str(job), str(dest), str(job / "out.sdf"))  # must not raise

    # Exactly one of the two logfiles must have been successfully moved: the
    # sweep continued past the failure instead of stopping on it.
    moved = list(dest.glob("oeomega_*.log"))
    assert len(moved) == 1, f"Expected 1 moved file, got {[f.name for f in moved]}"


class TestFindSmilesNotInSdfTautStripping:
    """C7: a decoded '@tautN' suffix must not cause a false "missing" report."""

    def test_taut_suffixed_output_name_matches_base_smi_id(self, tmp_path):
        """decode_ids keeps 'id@tautN' on tautomer conformers; the .smi only
        has the base id, so find_smiles_not_in_sdf must strip the suffix
        before comparing or every tautomer-derived molecule is misreported."""
        smi = tmp_path / "in.smi"
        smi.write_text("CCO mol_a\n")

        sdf = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(sdf))
        writer.write(_make_mol("mol_a@taut0"))
        writer.close()

        bad = find_smiles_not_in_sdf(str(smi), str(sdf))
        assert bad == [], f"mol_a wrongly reported missing: {bad}"


class TestFindIdsNotInSdf:
    """find_ids_not_in_sdf: the SDF-input counterpart to find_smiles_not_in_sdf."""

    def test_missing_id_is_reported(self, tmp_path):
        """An id present in the source SDF but absent from the output SDF is reported."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        for name in ["mol_a", "mol_b"]:
            writer.write(_make_mol(name))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(_make_mol("mol_a"))  # mol_b never produced a structure
        writer.close()

        bad = find_ids_not_in_sdf(str(source), str(out))
        assert bad == ["mol_b"]

    def test_no_missing_ids_returns_empty_list(self, tmp_path):
        """Every source id present in the output -> nothing reported."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        for name in ["mol_a", "mol_b"]:
            writer.write(_make_mol(name))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        for name in ["mol_b", "mol_a"]:
            writer.write(_make_mol(name))
        writer.close()

        assert find_ids_not_in_sdf(str(source), str(out)) == []

    def test_taut_suffixed_output_name_matches_base_id(self, tmp_path):
        """Same '@tautN' stripping as find_smiles_not_in_sdf, for SDF input."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        writer.write(_make_mol("mol_a"))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(_make_mol("mol_a@taut1"))
        writer.close()

        assert find_ids_not_in_sdf(str(source), str(out)) == []

    def test_an_unreadable_record_is_reported_on_the_input_side_only(
        self, tmp_path, monkeypatch
    ):
        """The two sides of the comparison are not symmetric, and must not be.

        This test used to assert ``== []`` for a source file containing an
        unreadable record, under the heading "must not crash or miscount". The
        empty list *was* the miscount: an input molecule the pipeline never saw
        was reported by nothing, and the run exited 0 claiming completeness.

        The asymmetry is the point:

        * **input side** -- a record that cannot be read is a molecule the user
          supplied and did not get back. It is reported, by position, since it has
          no ``_Name`` to report by.
        * **output side** -- a record that cannot be read yields no name to match
          against, and there is nothing better to do than skip it. Reporting it
          would invent a *missing input* out of an unreadable output.
        """
        import Auto3D.utils.file_ops as file_ops

        calls = {"n": 0}

        def fake_supplier(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return [_make_mol("mol_a"), None]  # source: one good, one unreadable
            return [None, _make_mol("mol_a")]      # output: mol_a made it

        monkeypatch.setattr(file_ops.Chem, "SDMolSupplier", fake_supplier)

        source = tmp_path / "source.sdf"
        source.write_text("placeholder")
        out = tmp_path / "out.sdf"
        out.write_text("placeholder")

        assert find_ids_not_in_sdf(str(source), str(out)) == [
            file_ops.UNPARSEABLE_RECORD_ID.format(index=1)
        ]


class TestReconciliationSeesUnparseableInputRecords:
    """A record Auto3D could not read must not vanish from the accounting.

    ``encode_ids`` skips an unparseable SDF record with a warning, so it never
    enters the run. ``find_ids_not_in_sdf`` then built its expected-ID list by
    reading **the same source SDF** and skipping the same record -- so it was in
    neither ``source_ids`` nor the output, could not appear in ``failures``, and
    ``_exit_if_incomplete`` saw ``failed_count == 0``. The run printed a success
    summary and exited **0** having processed fewer molecules than the file
    contained, which is precisely what the C7 reconciliation exists to prevent.

    Only the SDF path is affected. ``encode_ids`` reads ``.smi`` input with
    ``on_malformed="raise"``, so a malformed SMILES line aborts the run with
    ``InputValidationError`` long before reconciliation -- the same blindness
    cannot be reached through that door.
    """

    @staticmethod
    def _unparseable_record(name: str) -> str:
        """A molblock RDKit rejects: the counts line is corrupted.

        Built from a real molblock so only the one line under test is invalid;
        a wholly invented block could fail for an unrelated reason.
        """
        mol = _make_mol(name)
        lines = Chem.MolToMolBlock(mol).splitlines()
        lines[3] = "!! corrupted counts line !!"
        return "\n".join(lines)

    def test_an_unparseable_source_record_is_reported_as_a_failure(self, tmp_path):
        source = tmp_path / "source.sdf"
        good = Chem.MolToMolBlock(_make_mol("mol_a"))
        source.write_text(
            good + "$$$$\n" + self._unparseable_record("mol_b") + "\n$$$$\n"
        )
        # Confirm the premise: RDKit reads one molecule and one None.
        parsed = list(Chem.SDMolSupplier(str(source), removeHs=False))
        assert [m is None for m in parsed] == [False, True], "test premise"

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(_make_mol("mol_a"))  # the parseable one succeeded
        writer.close()

        missing = find_ids_not_in_sdf(str(source), str(out))

        assert len(missing) == 1, (
            f"a source record the pipeline could not read was left out of the "
            f"accounting entirely, so the run would exit 0 claiming every input "
            f"was processed; got {missing}"
        )
        assert "1" in missing[0], (
            f"the report must say which record could not be read, got {missing[0]!r}"
        )

    def test_a_clean_source_file_still_reports_nothing(self, tmp_path):
        """The new branch must not manufacture failures for a healthy file."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        for name in ("mol_a", "mol_b"):
            writer.write(_make_mol(name))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        for name in ("mol_a", "mol_b"):
            writer.write(_make_mol(name))
        writer.close()

        assert find_ids_not_in_sdf(str(source), str(out)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
