#!/usr/bin/env python
"""Tests for workflow orchestration, including multi-GPU handling."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from Auto3D.exceptions import ConfigurationError, FileFormatError, OptimizationError


class TestWorkflowExceptions:
    """Test WorkflowOrchestrator raises exceptions instead of sys.exit."""

    def test_validate_input_missing_path_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when path is None."""
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

        config = Auto3DOptions(
            path=None,  # Missing path
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)

        with pytest.raises(ConfigurationError, match="input file path"):
            orchestrator._validate_input()

    def test_validate_input_unsupported_format_raises_file_format_error(self, tmp_path):
        """Should raise FileFormatError for unsupported input format."""
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

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
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

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

    def test_validate_input_invalid_config_raises_configuration_error(self, tmp_path):
        """An invalid config (e.g. out-of-range gpu_idx) must fail fast in
        _validate_input via check_valid_configuration, not deep in a worker."""
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol")
        config = Auto3DOptions(path=str(smi_file), k=1)
        orchestrator = WorkflowOrchestrator(config)

        with patch('Auto3D.workflow.encode_ids') as mock_encode, \
             patch(
                 'Auto3D.workflow.check_valid_configuration',
                 return_value=["GPU index 5 is invalid. Available GPUs: 1"],
             ):
            mock_encode.return_value = (str(tmp_path / "test_encoded.smi"), {})
            (tmp_path / "test_encoded.smi").write_text("CCO ethanol")

            with pytest.raises(ConfigurationError, match="GPU index 5 is invalid"):
                orchestrator._validate_input()

    def test_finalize_output_no_structures_raises_optimization_error(self, tmp_path):
        """Should raise OptimizationError when no 3D structures converged."""
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

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
        from pathlib import Path

        from Auto3D.chunk_manager import ChunkManager
        from Auto3D.config import Auto3DOptions

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
        from pathlib import Path

        from Auto3D.chunk_manager import ChunkManager
        from Auto3D.config import Auto3DOptions

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


class TestIsomerWrapperFailure:
    """Tests that isomer_wrapper emits sentinels even when generation fails."""

    def test_isomer_wrapper_emits_sentinels_on_failure(self, monkeypatch):
        """If isomer generation raises, every optimizer must still get a 'Done' sentinel."""
        import multiprocessing as mp

        from Auto3D.auto3D import isomer_wrapper
        from Auto3D.config import Auto3DOptions

        args = Auto3DOptions(path="x.smi", k=1, gpu_idx=[0, 1])
        args.input_format = "smi"
        q = mp.Manager().Queue()
        logq = mp.Manager().Queue()

        # chunk_info points at a nonexistent dir so engine.run() raises inside the worker
        with pytest.raises(Exception):
            isomer_wrapper([("/nonexistent/chunk.smi", "/nonexistent")], args, q, logq)

        drained = []
        while not q.empty():
            drained.append(q.get())
        # one "Done" per GPU even though generation failed
        assert drained.count("Done") == 2


class TestOptimizerEmptyInput:
    """Tests for optimizer handling of empty/missing input files."""

    def test_optimizer_handles_missing_file(self, tmp_path, caplog, monkeypatch):
        """Optimizer should gracefully handle missing input files."""
        import logging
        from types import SimpleNamespace

        import torch

        from Auto3D.batch_opt.batchopt import optimizing

        # This test exercises missing-file handling only -- run() returns before
        # touching the model -- so stub create_model to skip the multi-second
        # real AIMNet2 load (and stay robust to sibling tests clearing the cache).
        monkeypatch.setattr(
            "Auto3D.batch_opt.batchopt.create_model",
            lambda *a, **k: SimpleNamespace(coord_pad=0.0, species_pad=-1),
        )

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

    def test_optimizer_handles_empty_file(self, tmp_path, caplog, monkeypatch):
        """Optimizer should gracefully handle empty input files."""
        import logging
        from types import SimpleNamespace

        import torch

        from Auto3D.batch_opt.batchopt import optimizing

        # Empty-file handling returns before the model is used; stub create_model
        # to skip the real AIMNet2 load (see test_optimizer_handles_missing_file).
        monkeypatch.setattr(
            "Auto3D.batch_opt.batchopt.create_model",
            lambda *a, **k: SimpleNamespace(coord_pad=0.0, species_pad=-1),
        )

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


def test_workers_importable_from_workflow_workers():
    from Auto3D.workflow_workers import isomer_wrapper, logger_process, optim_rank_wrapper
    assert all(callable(f) for f in (isomer_wrapper, optim_rank_wrapper, logger_process))


def test_optim_rank_wrapper_isolates_failing_chunks(tmp_path, monkeypatch):
    """A chunk that raises must not kill the worker or drop chunks queued behind it.

    Previously the optimizer worker's consume loop had no per-chunk exception
    handling, so one bad chunk (a molecule the optimizer chokes on, a CUDA OOM,
    an mkdir collision, or an empty isomer SDF) killed the whole process and
    silently dropped every remaining chunk -- with the parent still reporting
    success on the partial output. Now each chunk is isolated.
    """
    import queue as queue_mod

    from Auto3D import workflow_workers as ww
    from Auto3D.config import Auto3DOptions

    attempted = []

    class _BoomOptimizing:
        def __init__(self, in_f, out_f, engine, device, config, progress_cb=None):
            self._enumerated = in_f

        def run(self):
            attempted.append(self._enumerated)
            raise RuntimeError("optimizer blew up on this chunk")

    # Replace the heavy optimizing class (which would build a real model) with
    # one that always raises, so we exercise only the loop's failure isolation.
    monkeypatch.setattr(ww, "optimizing", _BoomOptimizing)

    q: queue_mod.Queue = queue_mod.Queue()
    d1 = tmp_path / "job1"
    d1.mkdir()
    d2 = tmp_path / "job2"
    d2.mkdir()
    q.put(("enum1.sdf", str(tmp_path / "c1.smi"), str(d1), 1))
    q.put(("enum2.sdf", str(tmp_path / "c2.smi"), str(d2), 2))
    q.put("Done")
    logq: queue_mod.Queue = queue_mod.Queue()

    args = Auto3DOptions(path="x.smi", k=1, use_gpu=False)

    # Must return normally (not propagate the RuntimeError) ...
    result = ww.optim_rank_wrapper(args, q, logq, gpu_idx=0)
    # ... and BOTH chunks must have been attempted: the loop continued past the
    # first chunk's failure instead of dying on it.
    assert attempted == ["enum1.sdf", "enum2.sdf"]
    assert result == []  # neither failing chunk produced conformers


def test_unsupported_extension_rejected_before_encoding(tmp_path):
    """Bad extensions must be rejected before encode_ids writes a temp file.

    Validating the suffix after encoding raised a generic ValueError from
    encode_ids and left an orphaned *_encoded file on disk.
    """
    from Auto3D.config import Auto3DOptions
    from Auto3D.workflow import WorkflowOrchestrator

    bad = tmp_path / "mol.xyz"
    bad.write_text("stuff\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(bad), k=1))

    with patch("Auto3D.workflow.encode_ids") as enc:
        with pytest.raises(FileFormatError, match="not supported"):
            orch._validate_input()
        enc.assert_not_called()  # format is validated before any encoding

    assert not list(tmp_path.glob("*_encoded*"))


def test_encoded_input_cleaned_up_when_setup_fails(tmp_path, monkeypatch):
    """The encoded temp file must be removed even when a setup phase fails.

    encode_ids writes a *_encoded file during phase-1 setup. That setup now runs
    inside run()'s try/finally, so a failure in a later setup step (job-dir
    creation, logging start) no longer leaks the encoded file beside the input.
    """
    from Auto3D.config import Auto3DOptions
    from Auto3D.workflow import WorkflowOrchestrator

    smi = tmp_path / "mol.smi"
    smi.write_text("CCO ethanol\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(smi), k=1, use_gpu=False))

    # Fail after _validate_input has already written the encoded temp file.
    monkeypatch.setattr(
        orch, "_setup_logging", MagicMock(side_effect=RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        orch.run()

    # The encoded temp file written during _validate_input must be gone, and no
    # *_encoded file may be left orphaned next to the input.
    assert orch.input_path != Path()
    assert not orch.input_path.exists()
    assert not list(tmp_path.glob("*_encoded*"))


def test_finalize_raises_when_all_outputs_empty(tmp_path):
    import pytest

    from Auto3D.config import Auto3DOptions
    from Auto3D.exceptions import OptimizationError
    from Auto3D.workflow import WorkflowOrchestrator

    orch = WorkflowOrchestrator(Auto3DOptions(path="x.smi", k=1))
    orch.job_dir = tmp_path
    orch.input_path = tmp_path / "x_encoded.smi"
    orch.input_path.write_text("CCO 0\n")
    job = tmp_path / "job1"
    job.mkdir()
    (job / "x_3d.sdf").write_text("")  # converged nothing -> empty SDF
    orch.id_mapping = {"a": 0}

    with pytest.raises(OptimizationError):
        orch._finalize_output(start_time=0.0)


def test_run_pipeline_does_not_mutate_shared_batchsize():
    """_run_pipeline must apply the memory-scaled batchsize via a per-run config
    copy and leave the caller's shared config untouched (review #35/#36)."""
    from Auto3D.config import Auto3DOptions
    from Auto3D.workflow import WorkflowOrchestrator

    config = Auto3DOptions(path="x.smi", k=1, batchsize_atoms=1024)
    orch = WorkflowOrchestrator(config)
    # Simulate the memory scaling that _prepare_chunks would have computed.
    orch.scaled_batchsize_atoms = 1024 * 4
    orch.logging_queue = MagicMock()

    captured_configs = []

    class _FakeProcess:
        def __init__(self, target=None, args=(), **kwargs):
            self._args = args

        def start(self):
            # Record every Auto3DOptions passed to a worker (positions differ
            # between the isomer and optimization workers).
            captured_configs.extend(
                a for a in self._args if isinstance(a, Auto3DOptions)
            )

        def join(self, timeout=None):
            return None

        @property
        def exitcode(self):
            return 0

    with patch("Auto3D.workflow.mp.Manager") as mock_manager, \
         patch("Auto3D.workflow.mp.Process", _FakeProcess):
        mock_manager.return_value.Queue.return_value = MagicMock()
        orch._run_pipeline([("chunk.smi", "job1")])

    # The shared config the caller passed in must be untouched.
    assert config.batchsize_atoms == 1024
    # The optimization worker must receive the memory-scaled batchsize.
    opt_configs = [c for c in captured_configs if c.batchsize_atoms == 1024 * 4]
    assert opt_configs, "optimizer did not receive the memory-scaled batchsize"


def test_smiles2mols_uses_args_threshold(monkeypatch):
    """smiles2mols must pass args.threshold (not a hardcoded value) to the
    isomer engine, matching main()'s candidate-pool behavior (review #35/#36)."""
    import Auto3D.auto3D as auto3D_mod
    from Auto3D.config import Auto3DOptions

    captured = {}

    class _StubIsomerEngine:
        def run(self):
            return None

    def _capture_create(*, threshold, **kwargs):
        captured["threshold"] = threshold
        return _StubIsomerEngine()

    class _StubOpt:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

    class _StubRank:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return []

    monkeypatch.setattr(
        auto3D_mod.IsomerEngineFactory, "create", staticmethod(_capture_create)
    )
    monkeypatch.setattr(auto3D_mod, "optimizing", _StubOpt)
    monkeypatch.setattr(auto3D_mod, "ranking", _StubRank)
    monkeypatch.setattr(auto3D_mod, "reorder_sdf", lambda *a, **k: [])

    args = Auto3DOptions(k=1, use_gpu=False, threshold=0.27)
    auto3D_mod.smiles2mols(["CCO"], args)

    assert captured["threshold"] == 0.27
    assert captured["threshold"] != 0.03


def test_orchestrator_input_format_single_source_of_truth(tmp_path):
    """input_format lives on the config (single source); the orchestrator no
    longer keeps a redundant instance attribute that could desync."""
    from Auto3D.config import Auto3DOptions
    from Auto3D.workflow import WorkflowOrchestrator

    smi = tmp_path / "m.smi"
    smi.write_text("CCO ethanol\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(smi), k=1, use_gpu=False))
    orch._validate_input()
    assert orch.config.input_format == "smi"
    assert not hasattr(orch, "input_format")
