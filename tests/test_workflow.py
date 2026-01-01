#!/usr/bin/env python
"""Tests for workflow orchestration, including multi-GPU handling."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from Auto3D.exceptions import ConfigurationError, FileFormatError, OptimizationError


class TestWorkflowExceptions:
    """Test WorkflowOrchestrator raises exceptions instead of sys.exit."""

    def test_validate_input_missing_path_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when path is None."""
        from Auto3D.workflow import WorkflowOrchestrator
        from Auto3D.config import Auto3DOptions

        config = Auto3DOptions(
            path=None,  # Missing path
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)

        with pytest.raises(ConfigurationError, match="input file path"):
            orchestrator._validate_input()

    def test_validate_input_unsupported_format_raises_file_format_error(self, tmp_path):
        """Should raise FileFormatError for unsupported input format."""
        from Auto3D.workflow import WorkflowOrchestrator
        from Auto3D.config import Auto3DOptions

        # Create a test file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("some content")

        config = Auto3DOptions(
            path=str(unsupported_file),
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)

        # Mock encode_ids to return the same path with _encoded suffix
        with patch('Auto3D.workflow.encode_ids') as mock_encode:
            mock_encode.return_value = (str(tmp_path / "test_encoded.xyz"), {})
            # Create the encoded file so it exists
            (tmp_path / "test_encoded.xyz").write_text("content")

            with pytest.raises(FileFormatError, match="not supported"):
                orchestrator._validate_input()

    def test_validate_input_missing_k_and_window_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when neither k nor window specified."""
        from Auto3D.workflow import WorkflowOrchestrator
        from Auto3D.config import Auto3DOptions

        # Create a valid .smi file
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol")

        config = Auto3DOptions(
            path=str(smi_file),
            k=None,  # Neither k nor window
            window=None,
        )

        orchestrator = WorkflowOrchestrator(config)

        # Mock encode_ids
        with patch('Auto3D.workflow.encode_ids') as mock_encode:
            mock_encode.return_value = (str(tmp_path / "test_encoded.smi"), {})
            (tmp_path / "test_encoded.smi").write_text("CCO ethanol")

            with pytest.raises(ConfigurationError, match="k or window"):
                orchestrator._validate_input()

    def test_finalize_output_no_structures_raises_optimization_error(self, tmp_path):
        """Should raise OptimizationError when no 3D structures converged."""
        from Auto3D.workflow import WorkflowOrchestrator
        from Auto3D.config import Auto3DOptions

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)
        orchestrator.job_dir = tmp_path
        orchestrator.input_path = tmp_path / "test_encoded.smi"
        orchestrator.logger = None

        # Create job directory with no output files
        (tmp_path / "job1").mkdir()
        # No *_3d.sdf files exist

        with pytest.raises(OptimizationError, match="no 3D structure converged"):
            orchestrator._finalize_output(0.0)


class TestChunkCreation:
    """Tests for chunk creation with edge cases."""

    def test_empty_chunks_skipped(self, tmp_path):
        """Empty chunks should be skipped when num_jobs > num_molecules.

        This tests the fix for issue #86 where multi-GPU with fewer molecules
        than GPUs caused OSError due to empty SDF files.
        """
        from Auto3D.chunk_manager import ChunkManager
        from Auto3D.config import Auto3DOptions
        from pathlib import Path

        # Create a minimal config - we won't run the full pipeline
        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
        )

        # Create test ChunkManager
        chunk_manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        # Create a small DataFrame (1 molecule)
        df = pd.DataFrame({0: ["CCO"], 1: ["ethanol"]})

        # Create chunk indices simulating 3 GPUs with 1 molecule
        # Only chunk 0 should have data, chunks 1 and 2 should be empty
        chunk_idxes = [[0], [], []]  # 3 chunks, only first has data

        # Run chunk creation
        chunk_info = chunk_manager._create_chunk_files(df, chunk_idxes, 3)

        # Should only have 1 chunk (empty ones skipped)
        assert len(chunk_info) == 1
        assert "job1" in chunk_info[0][1]

        # Verify job1 dir exists, job2/job3 don't
        assert (tmp_path / "job1").exists()
        assert not (tmp_path / "job2").exists()
        assert not (tmp_path / "job3").exists()

    def test_all_chunks_with_data(self, tmp_path):
        """All chunks with data should be created."""
        from Auto3D.chunk_manager import ChunkManager
        from Auto3D.config import Auto3DOptions
        from pathlib import Path

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
        )

        chunk_manager = ChunkManager(
            config=config,
            input_path=Path(tmp_path / "test_encoded.smi"),
            input_format="smi",
            job_dir=tmp_path,
            workflow_logger=None,
        )

        # Create DataFrame with 3 molecules
        df = pd.DataFrame({
            0: ["CCO", "CCCO", "CCCCO"],
            1: ["ethanol", "propanol", "butanol"]
        })

        # Create chunk indices - each chunk has one molecule
        chunk_idxes = [[0], [1], [2]]

        chunk_info = chunk_manager._create_chunk_files(df, chunk_idxes, 3)

        # Should have all 3 chunks
        assert len(chunk_info) == 3
        assert (tmp_path / "job1").exists()
        assert (tmp_path / "job2").exists()
        assert (tmp_path / "job3").exists()


class TestOptimizerEmptyInput:
    """Tests for optimizer handling of empty/missing input files."""

    def test_optimizer_handles_missing_file(self, tmp_path, caplog):
        """Optimizer should gracefully handle missing input files."""
        from Auto3D.batch_opt.batchopt import optimizing
        import logging
        import torch

        device = torch.device("cpu")
        config = {
            "opt_steps": 100,
            "opttol": 0.003,
            "patience": 100,
            "batchsize_atoms": 1024,
        }

        nonexistent = str(tmp_path / "nonexistent.sdf")
        optimizer = optimizing(nonexistent, str(tmp_path / "out.sdf"), "AIMNET", device, config)

        # Should not raise, just log warning and return
        with caplog.at_level(logging.WARNING):
            optimizer.run()

        assert "does not exist" in caplog.text

    def test_optimizer_handles_empty_file(self, tmp_path, caplog):
        """Optimizer should gracefully handle empty input files."""
        from Auto3D.batch_opt.batchopt import optimizing
        import logging
        import torch

        device = torch.device("cpu")
        config = {
            "opt_steps": 100,
            "opttol": 0.003,
            "patience": 100,
            "batchsize_atoms": 1024,
        }

        # Create empty file
        empty_sdf = tmp_path / "empty.sdf"
        empty_sdf.write_text("")

        optimizer = optimizing(str(empty_sdf), str(tmp_path / "out.sdf"), "AIMNET", device, config)

        # Should not raise, just log warning and return
        with caplog.at_level(logging.WARNING):
            optimizer.run()

        assert "empty" in caplog.text
