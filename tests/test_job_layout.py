"""Tests for Auto3D.job_layout module."""

from pathlib import Path

import pytest  # noqa: F401  (used by the __main__ guard below)

from Auto3D.job_layout import create_chunk_meta_names, housekeeping

# Get the test files directory
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"


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


class TestFileOpsIntegration:
    """Integration tests for the job-layout helpers."""

    def test_create_chunks_and_housekeeping_workflow(self, tmp_path):
        """Test a typical workflow using both job_layout functions."""
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
        housekeeping(str(job_dir), meta["housekeeping_folder"], meta["output"])

        # Output should still exist
        assert Path(meta["output"]).exists()
        # Enumerated should be moved to verbose folder
        assert (Path(meta["housekeeping_folder"]) / "smiles_enumerated.smi").exists()


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

    from Auto3D.job_layout import housekeeping

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

    monkeypatch.setattr("Auto3D.job_layout.shutil.move", flaky_move)

    housekeeping(str(job), str(dest), str(job / "out.sdf"))  # must not raise

    # Exactly one of the two logfiles must have been successfully moved: the
    # sweep continued past the failure instead of stopping on it.
    moved = list(dest.glob("oeomega_*.log"))
    assert len(moved) == 1, f"Expected 1 moved file, got {[f.name for f in moved]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
