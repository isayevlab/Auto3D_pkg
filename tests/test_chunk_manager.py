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
