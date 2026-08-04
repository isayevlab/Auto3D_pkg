"""Tests for Auto3D.chunk_manager module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from Auto3D.chunk_manager import ChunkManager
from Auto3D.config import Auto3DOptions


class TestChunkManagerInit:
    """Tests for ChunkManager initialization."""

    def test_init_stores_config(self, tmp_path):
        """ChunkManager should store all initialization parameters."""
        config = Auto3DOptions(path="test.smi", k=1)
        input_path = Path(tmp_path / "input.smi")

        manager = ChunkManager(
            config=config,
            input_path=input_path,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        assert manager.config is config
        assert manager.input_path == input_path
        assert manager.input_format == "smi"
        assert manager.job_dir == tmp_path
        assert manager.workflow_logger is None


class TestCalculateMemoryAndChunks:
    """Tests for ChunkManager.calculate_memory_and_chunks()."""

    def test_uses_provided_memory(self, tmp_path):
        """Should use config.memory when provided."""
        config = Auto3DOptions(path="test.smi", k=1, memory=8)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "input.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        memory_gb, chunk_size, num_jobs = manager.calculate_memory_and_chunks()

        assert memory_gb == 8
        assert num_jobs == 1  # Single job when memory is manually set

    def test_single_gpu_returns_one_job(self, tmp_path):
        """Should return num_jobs=1 for single GPU index."""
        config = Auto3DOptions(path="test.smi", k=1, use_gpu=True, gpu_idx=0)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "input.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = MagicMock(total_memory=8 * 1024**3)
            memory_gb, chunk_size, num_jobs = manager.calculate_memory_and_chunks()

        assert num_jobs == 1

    def test_multiple_gpus_returns_gpu_count(self, tmp_path):
        """Should return num_jobs equal to GPU count for multiple GPUs."""
        config = Auto3DOptions(path="test.smi", k=1, use_gpu=True, gpu_idx=[0, 1, 2])
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "input.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        with patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = MagicMock(total_memory=8 * 1024**3)
            memory_gb, chunk_size, num_jobs = manager.calculate_memory_and_chunks()

        assert num_jobs == 3

    def test_cpu_uses_system_memory(self, tmp_path):
        """Should use system memory when use_gpu=False."""
        config = Auto3DOptions(path="test.smi", k=1, use_gpu=False)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "input.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value = MagicMock(total=16 * 1024**3)
            memory_gb, chunk_size, num_jobs = manager.calculate_memory_and_chunks()

        assert memory_gb == 16
        assert num_jobs == 1

    def test_chunk_size_includes_capacity_multiplier(self, tmp_path):
        """Chunk size should be memory_gb * capacity."""
        config = Auto3DOptions(path="test.smi", k=1, memory=4, capacity=100)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "input.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        memory_gb, chunk_size, num_jobs = manager.calculate_memory_and_chunks()

        assert chunk_size == 4 * 100  # 400


class TestCreateChunkFiles:
    """Tests for ChunkManager._create_chunk_files()."""

    def test_empty_chunks_skipped(self, tmp_path):
        """Empty chunks should be skipped."""
        config = Auto3DOptions(path=str(tmp_path / "test.smi"), k=1)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        # Single molecule distributed across 3 chunks
        df = pd.DataFrame({0: ["CCO"], 1: ["ethanol"]})
        chunk_idxes = [[0], [], []]  # Only first chunk has data

        chunk_info = manager._create_chunk_files(df, chunk_idxes, 3)

        assert len(chunk_info) == 1
        assert "job1" in chunk_info[0][1]
        assert (tmp_path / "job1").exists()
        assert not (tmp_path / "job2").exists()

    def test_all_chunks_created_when_data_exists(self, tmp_path):
        """All chunks with data should be created."""
        config = Auto3DOptions(path=str(tmp_path / "test.smi"), k=1)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        # Three molecules, one per chunk
        df = pd.DataFrame({
            0: ["CCO", "CCCO", "CCCCO"],
            1: ["ethanol", "propanol", "butanol"],
        })
        chunk_idxes = [[0], [1], [2]]

        chunk_info = manager._create_chunk_files(df, chunk_idxes, 3)

        assert len(chunk_info) == 3
        assert (tmp_path / "job1").exists()
        assert (tmp_path / "job2").exists()
        assert (tmp_path / "job3").exists()

    def test_smi_format_creates_csv_files(self, tmp_path):
        """SMI format should create CSV chunk files."""
        config = Auto3DOptions(path=str(tmp_path / "test.smi"), k=1)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        df = pd.DataFrame({0: ["CCO"], 1: ["ethanol"]})
        chunk_idxes = [[0]]

        chunk_info = manager._create_chunk_files(df, chunk_idxes, 1)

        chunk_path = Path(chunk_info[0][0])
        assert chunk_path.suffix == ".smi"
        assert chunk_path.exists()

    def test_sdf_format_creates_sdf_files(self, tmp_path):
        """SDF format should create SDF chunk files."""
        config = Auto3DOptions(path=str(tmp_path / "test.sdf"), k=1)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.sdf"),
            input_format="sdf",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        # SDF data is list of line lists
        sdf_data = [["mol1\n", "  data\n", "$$$$\n"]]
        chunk_idxes = [[0]]

        chunk_info = manager._create_chunk_files(sdf_data, chunk_idxes, 1)

        chunk_path = Path(chunk_info[0][0])
        assert chunk_path.suffix == ".sdf"
        assert chunk_path.exists()

    def test_returns_path_and_dir_tuples(self, tmp_path):
        """Should return list of (chunk_path, chunk_dir) tuples."""
        config = Auto3DOptions(path=str(tmp_path / "test.smi"), k=1)
        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        df = pd.DataFrame({0: ["CCO"], 1: ["ethanol"]})
        chunk_idxes = [[0]]

        chunk_info = manager._create_chunk_files(df, chunk_idxes, 1)

        assert len(chunk_info) == 1
        chunk_path, chunk_dir = chunk_info[0]
        assert Path(chunk_path).exists()
        assert Path(chunk_dir).is_dir()


class TestPrepareChunks:
    """Tests for ChunkManager.prepare_chunks()."""

    def test_prepares_smi_chunks(self, tmp_path):
        """Should prepare chunks for SMI input."""
        # Create input file
        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("CCO ethanol\nCCCO propanol\n")

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
            memory=1,
            capacity=10,  # Small capacity to ensure single chunk
        )
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        chunk_info = manager.prepare_chunks()

        assert len(chunk_info) >= 1
        # All paths should exist
        for path, dir_ in chunk_info:
            assert Path(path).exists()
            assert Path(dir_).is_dir()

    def test_prepare_chunks_does_not_mutate_config(self, tmp_path):
        """prepare_chunks must not mutate the caller's batchsize_atoms.

        The chunk-sizing logic scales batchsize_atoms by available memory; doing
        that in place on the shared config compounds the multiplier when main()
        is called twice with the same Auto3DOptions (OOM risk).
        """
        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("CCO ethanol\nCCCO propanol\n")

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
            memory=4,  # fixed memory => deterministic multiplier
            capacity=10,
            batchsize_atoms=1024,
        )
        original = config.batchsize_atoms

        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )
        manager.prepare_chunks()

        # The caller's config must be untouched, even after a memory-scaled run.
        assert config.batchsize_atoms == original

    def test_prepare_chunks_exposes_scaled_batchsize(self, tmp_path):
        """The memory-scaled batchsize must still be available for optimization."""
        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("CCO ethanol\nCCCO propanol\n")

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
            memory=4,
            capacity=10,
            batchsize_atoms=1024,
        )
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )
        manager.prepare_chunks()

        # batchsize scaled by the 4 GB memory budget.
        assert manager.scaled_batchsize_atoms == 1024 * 4


class TestRaggedSmiAndChunkSizeClamp:
    """FIX 6: ragged .smi reads and degenerate chunk_size must not crash."""

    def test_prepare_chunks_reads_ragged_smi(self, tmp_path):
        """A .smi line with an extra whitespace token must not crash the read.

        encode_ids tolerates extra columns, so prepare_chunks must too. The old
        pd.read_csv(sep=r"\\s+") raised 'Expected 2 fields, saw 3'.
        """
        input_file = tmp_path / "test_encoded.smi"
        # second line has 3 whitespace tokens.
        input_file.write_text("CCO ethanol\nCCCO propanol extra\n")

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
            memory=1,
            capacity=10,
        )
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        chunk_info = manager.prepare_chunks()

        # Both molecules retained; only the first two columns are used.
        assert len(chunk_info) >= 1
        total_rows = 0
        for path, _ in chunk_info:
            total_rows += sum(
                1 for ln in Path(path).read_text().splitlines() if ln.strip()
            )
        assert total_rows == 2

    def test_chunk_size_clamped_to_at_least_one(self, tmp_path):
        """When memory_gb*capacity < 1, chunk_size must clamp to >= 1.

        Otherwise data_size // chunk_size explodes num_chunks (or divides by 0).
        """
        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("CCO ethanol\nCCCO propanol\nCCCCO butanol\n")

        # memory=1 GB * capacity=0.0 -> raw chunk_size 0.0 (degenerate). capacity
        # is now bounds-checked (>= 1) at construction (Task 1, C10/M27 parity),
        # so the degenerate value is set directly on the field afterward --
        # Auto3DOptions is a plain mutable dataclass and only validates at
        # __init__ -- to still exercise ChunkManager's own defensive clamp
        # below for a value that reaches it by some other means.
        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
            memory=1,
        )
        config.capacity = 0.0
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        # Must not raise ZeroDivisionError and must not explode num_chunks.
        chunk_info = manager.prepare_chunks()

        # With 3 molecules and a clamped chunk_size of 1, num_chunks stays sane
        # (no more than data_size + 1 jobs).
        assert 1 <= len(chunk_info) <= 4


class TestSmiParserCrossAgreement:
    """M59: chunk_manager's pandas-based `.smi` reader stays separate from
    ``iter_smi_records`` for performance (avoiding a pure-Python per-line loop
    over what can be a very large file), but the two must not silently drift
    apart on the input they both exist to read: an already-ID-encoded
    intermediate `.smi` file with no blank lines and no comments (chunk_manager
    calls this "encode_ids semantics" in prepare_chunks' docstring). If either
    parser's tokenizing/whitespace-splitting rule changes, this test should
    catch the divergence.
    """

    def test_prepare_chunks_agrees_with_iter_smi_records_on_well_formed_input(
        self, tmp_path
    ):
        """Both parsers must extract the same (smiles, id) pairs, in the same
        order, from a well-formed encoded .smi file."""
        from Auto3D.utils.smi_io import iter_smi_records

        rows = [
            ("CCO", "0"),
            ("CCCO", "1"),
            ("c1ccccc1", "2"),
            ("C[C@H](O)F", "3"),
            ("CCN", "4"),
        ]
        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("".join(f"{smi} {mol_id}\n" for smi, mol_id in rows))

        # memory * capacity is chosen well above len(rows) so chunk_size never
        # forces a split: exactly one chunk, holding every row in original
        # order, so the chunk file's content is directly comparable to
        # iter_smi_records' output order (round-robin distribution across
        # multiple chunks would otherwise reorder rows relative to the input).
        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"), k=1, memory=1, capacity=1000
        )
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        chunk_info = manager.prepare_chunks()
        assert len(chunk_info) == 1, "test assumes no chunk splitting"

        chunk_path, _ = chunk_info[0]
        pandas_rows = [
            tuple(line.split())
            for line in Path(chunk_path).read_text().splitlines()
            if line.strip()
        ]
        iter_rows = [
            (smi, mol_id) for _line_no, smi, mol_id in iter_smi_records(str(input_file))
        ]

        assert pandas_rows == rows
        assert iter_rows == rows
        assert pandas_rows == iter_rows

    def test_ragged_extra_column_agreement(self, tmp_path):
        """A trailing whitespace-separated column beyond SMILES+ID must be
        dropped identically by both readers (chunk_manager's usecols=[0, 1]
        vs. iter_smi_records taking only parts[0]/parts[1])."""
        from Auto3D.utils.smi_io import iter_smi_records

        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("CCO 0 inline_comment_column\nCCCO 1\n")

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"), k=1, memory=1, capacity=1000
        )
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )
        chunk_info = manager.prepare_chunks()
        assert len(chunk_info) == 1

        chunk_path, _ = chunk_info[0]
        pandas_rows = [
            tuple(line.split())
            for line in Path(chunk_path).read_text().splitlines()
            if line.strip()
        ]
        iter_rows = [
            (smi, mol_id) for _line_no, smi, mol_id in iter_smi_records(str(input_file))
        ]

        assert pandas_rows == iter_rows == [("CCO", "0"), ("CCCO", "1")]

    def test_documented_divergence_comment_lines(self, tmp_path):
        """Documents a real, deliberate divergence rather than papering over
        it: iter_smi_records treats a '#'-prefixed line as a comment and
        skips it (matching cli.commands.validate, per M25); chunk_manager's
        pd.read_csv has no `comment=` parameter and reads it as a data row.

        This is not a bug this task fixes (chunk_manager stays a separate,
        faster reader by design) -- it is recorded here so that a future
        change to either parser's comment handling is a deliberate,
        visible decision rather than a silent behavior change caught only in
        production. If this test starts failing because someone taught
        pd.read_csv to skip '#' lines, that is progress: update the
        assertion, don't just delete the test.
        """
        from Auto3D.utils.smi_io import iter_smi_records

        input_file = tmp_path / "test_encoded.smi"
        input_file.write_text("CCO 0\n# 1 2\nCCCO 3\n")

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"), k=1, memory=1, capacity=1000
        )
        manager = ChunkManager(
            config=config,
            input_path=input_file,
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )
        chunk_info = manager.prepare_chunks()
        chunk_path, _ = chunk_info[0]
        pandas_rows = [
            tuple(line.split())
            for line in Path(chunk_path).read_text().splitlines()
            if line.strip()
        ]
        iter_rows = [
            (smi, mol_id) for _line_no, smi, mol_id in iter_smi_records(str(input_file))
        ]

        # pandas reads the '#' line as data; iter_smi_records skips it.
        assert pandas_rows == [("CCO", "0"), ("#", "1"), ("CCCO", "3")]
        assert iter_rows == [("CCO", "0"), ("CCCO", "3")]
        assert pandas_rows != iter_rows


class TestLogging:
    """Tests for ChunkManager logging."""

    def test_logs_to_workflow_logger(self, tmp_path):
        """Should log to workflow_logger when provided."""
        config = Auto3DOptions(path=str(tmp_path / "test.smi"), k=1)
        mock_logger = MagicMock()

        manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=mock_logger,
        )

        manager._log_info("Test message")

        mock_logger.info.assert_called_once_with("Test message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
