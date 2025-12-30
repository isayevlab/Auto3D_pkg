#!/usr/bin/env python
"""Tests for workflow orchestration, including multi-GPU handling."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestChunkCreation:
    """Tests for chunk creation with edge cases."""

    def test_empty_chunks_skipped(self, tmp_path):
        """Empty chunks should be skipped when num_jobs > num_molecules.

        This tests the fix for issue #86 where multi-GPU with fewer molecules
        than GPUs caused OSError due to empty SDF files.
        """
        from Auto3D.workflow import WorkflowOrchestrator
        from Auto3D.config import Auto3DOptions

        # Create a minimal config - we won't run the full pipeline
        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
        )

        # Create test orchestrator
        orchestrator = WorkflowOrchestrator(config)
        orchestrator.job_dir = tmp_path
        orchestrator.input_path = tmp_path / "test_encoded.smi"
        orchestrator.input_format = "smi"
        orchestrator.logger = None

        # Create a small DataFrame (1 molecule)
        df = pd.DataFrame({0: ["CCO"], 1: ["ethanol"]})

        # Create chunk indices simulating 3 GPUs with 1 molecule
        # Only chunk 0 should have data, chunks 1 and 2 should be empty
        chunk_idxes = [[0], [], []]  # 3 chunks, only first has data

        # Run chunk creation
        chunk_info = orchestrator._create_chunk_files(df, chunk_idxes, 3)

        # Should only have 1 chunk (empty ones skipped)
        assert len(chunk_info) == 1
        assert "job1" in chunk_info[0][1]

        # Verify job1 dir exists, job2/job3 don't
        assert (tmp_path / "job1").exists()
        assert not (tmp_path / "job2").exists()
        assert not (tmp_path / "job3").exists()

    def test_all_chunks_with_data(self, tmp_path):
        """All chunks with data should be created."""
        from Auto3D.workflow import WorkflowOrchestrator
        from Auto3D.config import Auto3DOptions

        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)
        orchestrator.job_dir = tmp_path
        orchestrator.input_path = tmp_path / "test_encoded.smi"
        orchestrator.input_format = "smi"
        orchestrator.logger = None

        # Create DataFrame with 3 molecules
        df = pd.DataFrame({
            0: ["CCO", "CCCO", "CCCCO"],
            1: ["ethanol", "propanol", "butanol"]
        })

        # Create chunk indices - each chunk has one molecule
        chunk_idxes = [[0], [1], [2]]

        chunk_info = orchestrator._create_chunk_files(df, chunk_idxes, 3)

        # Should have all 3 chunks
        assert len(chunk_info) == 3
        assert (tmp_path / "job1").exists()
        assert (tmp_path / "job2").exists()
        assert (tmp_path / "job3").exists()


class TestOptimizerEmptyInput:
    """Tests for optimizer handling of empty/missing input files."""

    def test_optimizer_handles_missing_file(self, tmp_path, capsys):
        """Optimizer should gracefully handle missing input files."""
        from Auto3D.batch_opt.batchopt import optimizing
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

        # Should not raise, just print warning and return
        optimizer.run()

        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_optimizer_handles_empty_file(self, tmp_path, capsys):
        """Optimizer should gracefully handle empty input files."""
        from Auto3D.batch_opt.batchopt import optimizing
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

        # Should not raise, just print warning and return
        optimizer.run()

        captured = capsys.readouterr()
        assert "empty" in captured.out
