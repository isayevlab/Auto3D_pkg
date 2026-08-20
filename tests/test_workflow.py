#!/usr/bin/env python
"""Tests for workflow orchestration, including multi-GPU handling."""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import Auto3D.orchestration.workflow
from Auto3D.foundation.exceptions import ConfigurationError, FileFormatError, OptimizationError


class TestWorkflowExceptions:
    """Test WorkflowOrchestrator raises exceptions instead of sys.exit."""

    def test_validate_input_missing_path_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when path is None."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        config = Auto3DOptions(
            path=None,  # Missing path
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)

        with pytest.raises(ConfigurationError, match="input file path"):
            orchestrator._validate_input()

    def test_validate_input_unsupported_format_raises_file_format_error(self, tmp_path):
        """Should raise FileFormatError for unsupported input format."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        # Create a test file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("some content")

        config = Auto3DOptions(
            path=str(unsupported_file),
            k=1,
        )

        orchestrator = WorkflowOrchestrator(config)

        # No encode_ids stub: _validate_input does not encode anything. The
        # ordering guarantee (format checked before any encoding) is pinned by
        # test_unsupported_extension_rejected_before_encoding, which drives
        # run() where encoding actually lives.
        with pytest.raises(FileFormatError, match="not supported"):
            orchestrator._validate_input()

    def test_validate_input_missing_k_and_window_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when neither k nor window specified."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        # Create a valid .smi file
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol")

        config = Auto3DOptions(
            path=str(smi_file),
            k=None,  # Neither k nor window
            window=None,
        )

        orchestrator = WorkflowOrchestrator(config)

        with pytest.raises(ConfigurationError, match="k or window"):
            orchestrator._validate_input()

    def test_validate_input_invalid_config_raises_configuration_error(self, tmp_path):
        """An invalid config (e.g. out-of-range gpu_idx) must fail fast in
        _validate_input via check_valid_configuration, not deep in a worker."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol")
        config = Auto3DOptions(path=str(smi_file), k=1)
        orchestrator = WorkflowOrchestrator(config)

        with patch.object(
            Auto3D.orchestration.workflow,
            "check_valid_configuration",
            return_value=["GPU index 5 is invalid. Available GPUs: 1"],
        ):
            with pytest.raises(ConfigurationError, match="GPU index 5 is invalid"):
                orchestrator._validate_input()

    def test_finalize_output_no_structures_raises_optimization_error(self, tmp_path):
        """Should raise OptimizationError when no 3D structures converged."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

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

        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.chunk_manager import ChunkManager

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

        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.chunk_manager import ChunkManager

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
        df = pd.DataFrame({0: ["CCO", "CCCO", "CCCCO"], 1: ["ethanol", "propanol", "butanol"]})

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

        from Auto3D.entry.auto3D import isomer_wrapper
        from Auto3D.foundation.config import Auto3DOptions

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

        import torch

        from Auto3D.engines.batch_opt.batchopt import optimizing
        from tests.helpers_adapter import FakeAdapter

        device = torch.device("cpu")
        config = {
            "opt_steps": 100,
            "opttol": 0.003,
            "patience": 100,
            "batchsize_atoms": 1024,
        }

        nonexistent = str(tmp_path / "nonexistent.sdf")
        # An injected double, because `optimizing` no longer constructs its own
        # adapter -- and this test returns before the model is touched anyway.
        optimizer = optimizing(
            nonexistent,
            str(tmp_path / "out.sdf"),
            adapter=FakeAdapter(),
            device=device,
            config=config,
        )

        # Should not raise, just log warning and return
        with caplog.at_level(logging.WARNING):
            optimizer.run()

        assert "does not exist" in caplog.text

    def test_optimizer_handles_empty_file(self, tmp_path, caplog, monkeypatch):
        """Optimizer should gracefully handle empty input files."""
        import logging

        import torch

        from Auto3D.engines.batch_opt.batchopt import optimizing
        from tests.helpers_adapter import FakeAdapter

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

        optimizer = optimizing(
            str(empty_sdf),
            str(tmp_path / "out.sdf"),
            adapter=FakeAdapter(),
            device=device,
            config=config,
        )

        # Should not raise, just log warning and return
        with caplog.at_level(logging.WARNING):
            optimizer.run()

        # Pin the exact guard that fired: the empty-file message must be
        # distinguishable from the missing-file message above ("does not
        # exist"), which is also a file literally named "empty.sdf" would
        # trivially satisfy a bare "empty" in caplog.text check without ever
        # proving the *empty-file* branch (not the missing-file branch) ran.
        assert f"Input file {empty_sdf} is empty." in caplog.text
        assert "does not exist" not in caplog.text


def test_workers_importable_from_workflow_workers():
    from Auto3D.orchestration.workflow_workers import (
        isomer_wrapper,
        logger_process,
        optim_rank_wrapper,
    )

    assert all(callable(f) for f in (isomer_wrapper, optim_rank_wrapper, logger_process))


class TestAFailedChunksCauseReachesTheUser:
    """A worker's warnings and errors must not stay buried in the run log.

    Worker processes log through a ``QueueHandler`` whose only destination was a
    ``FileHandler`` in ``logger_process``. So when a chunk failed, its traceback
    went to ``<job_dir>/Auto3D.log`` and the user saw nothing: the *loss* was
    reported (reconciliation names the missing molecules, the run exits 6) but
    the *cause* was not, making a systematic bug that failed every chunk
    identically look like a batch of difficult molecules.

    These drive ``logger_process`` directly rather than spawning a real worker,
    because the behavior under test belongs to the collector: it is the one place
    that decides where a worker's records go.
    """

    @staticmethod
    def _drain(records, logging_path, capfd):
        """Run logger_process over `records`, return (stderr, log file contents).

        logger_process adds handlers to the process-wide "auto3d" logger and, in
        production, is the whole life of a dedicated process. Called in-process
        here, so its handlers are removed again afterwards -- otherwise every
        later test in the session inherits them.
        """
        import logging as logging_mod
        import queue as queue_mod

        from Auto3D.orchestration.workflow_workers import logger_process

        logger = logging_mod.getLogger("auto3d")
        before = list(logger.handlers)
        before_level = logger.level
        q: queue_mod.Queue = queue_mod.Queue()
        for record in records:
            q.put(record)
        q.put(None)
        try:
            logger_process(q, str(logging_path))
        finally:
            for handler in list(logger.handlers):
                if handler not in before:
                    logger.removeHandler(handler)
                    handler.close()
            logger.setLevel(before_level)
        return capfd.readouterr().err, Path(logging_path).read_text()

    @staticmethod
    def _record(level, message):
        import logging as logging_mod

        return logging_mod.LogRecord(
            name="auto3d",
            level=level,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )

    def test_an_error_from_a_worker_is_written_to_stderr(self, tmp_path, capfd):
        message = "job3 failed during optimization/ranking"
        err, log_text = self._drain(
            [self._record(logging.ERROR, message)], tmp_path / "Auto3D.log", capfd
        )

        assert message in err, (
            "a failed chunk's cause never reached stderr, so the user sees only "
            "that molecules are missing and not why"
        )
        assert message in log_text, "the run log must still receive it as well"

    def test_a_warning_from_a_worker_is_written_to_stderr(self, tmp_path, capfd):
        """Covers the sibling case: 'no optimized structures were produced'."""
        message = "job7: no optimized structures were produced"
        err, _ = self._drain(
            [self._record(logging.WARNING, message)], tmp_path / "Auto3D.log", capfd
        )

        assert message in err

    def test_info_stays_in_the_run_log_and_off_stderr(self, tmp_path, capfd):
        """The step-by-step narrative must not be promoted to the terminal.

        Without this, the fix above would turn every 'Optimizing on jobN' line
        into console output and bury the warnings it exists to surface -- and it
        would put chatter on the stream an interactive run draws its live panel
        on.
        """
        message = "Optimizing on job1"
        err, log_text = self._drain(
            [self._record(logging.INFO, message)], tmp_path / "Auto3D.log", capfd
        )

        assert message in log_text, "the run log must still receive INFO"
        assert message not in err, "INFO was promoted to stderr"


def test_optim_rank_wrapper_isolates_failing_chunks(tmp_path, monkeypatch):
    """A chunk that raises must not kill the worker or drop chunks queued behind it.

    Previously the optimizer worker's consume loop had no per-chunk exception
    handling, so one bad chunk (a molecule the optimizer chokes on, a CUDA OOM,
    an mkdir collision, or an empty isomer SDF) killed the whole process and
    silently dropped every remaining chunk -- with the parent still reporting
    success on the partial output. Now each chunk is isolated.
    """
    import queue as queue_mod

    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration import workflow_workers as ww

    attempted = []

    class _BoomOptimizing:
        def __init__(self, in_f, out_f, *, adapter, device, config, progress_cb=None):
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
    # No return value, by design: this function only ever runs as an
    # `mp.Process` target (workflow.py), so anything returned is discarded.
    # It used to accumulate every chunk's ranked mols into a list it then
    # returned, which held the whole run's molecules in worker memory and was
    # read by nobody. Ranked structures reach the caller through the output
    # SDF each chunk writes, not through this frame.
    assert result is None


def test_unsupported_extension_rejected_before_encoding(tmp_path):
    """Bad extensions must be rejected before encode_ids writes a temp file.

    Validating the suffix after encoding raised a generic ValueError from
    encode_ids and left an orphaned *_encoded file on disk.

    Driven through ``run()``, not ``_validate_input()``. Encoding no longer
    happens inside ``_validate_input`` at all -- it is its own phase, after
    the job directory is created -- so ``enc.assert_not_called()`` against
    ``_validate_input`` could not fail under any input whatsoever and pinned
    nothing. Against ``run()`` it can: move the format check after
    ``_encode_input`` and the mock is called.
    """
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    bad = tmp_path / "mol.xyz"
    bad.write_text("stuff\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(bad), k=1, use_gpu=False))

    with patch.object(Auto3D.orchestration.workflow, "encode_ids") as enc:
        with pytest.raises(FileFormatError, match="not supported"):
            orch.run()
        enc.assert_not_called()  # format is validated before any encoding

    # Nothing was created: no encoded file anywhere (rglob, because the
    # encoded copy's home is now a subdirectory), and no job directory --
    # the format check runs before _setup_job_directory too.
    assert not list(tmp_path.rglob("*_encoded*"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mol.xyz"]


def test_encoded_input_cleaned_up_when_setup_fails(tmp_path, monkeypatch):
    """The encoded temp file must be removed even when a setup phase fails.

    encode_ids writes a *_encoded file during phase-1 setup. That setup now runs
    inside run()'s try/finally, so a failure in a later setup step (job-dir
    creation, logging start) no longer leaks the encoded file beside the input.
    """
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    smi = tmp_path / "mol.smi"
    smi.write_text("CCO ethanol\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(smi), k=1, use_gpu=False))

    # Fail after _validate_input has already written the encoded temp file.
    monkeypatch.setattr(orch, "_setup_logging", MagicMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        orch.run()

    # The encoded temp file must be gone, and no *_encoded file may be left
    # orphaned anywhere under the input's directory. rglob, not glob: the
    # encoded copy lives in `tmp_path/<stem>_<job_name>/` now, which a
    # non-recursive glob cannot see -- it would hold with the cleanup deleted.
    assert orch.input_path != Path()
    assert not orch.input_path.exists()
    assert not list(tmp_path.rglob("*_encoded*"))


def test_finalize_raises_when_all_outputs_empty(tmp_path):
    import pytest

    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.exceptions import OptimizationError
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

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
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

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
            captured_configs.extend(a for a in self._args if isinstance(a, Auto3DOptions))

        def join(self, timeout=None):
            return None

        @property
        def exitcode(self):
            return 0

    # The seam is the orchestrator's own context, not the multiprocessing
    # module. That is the point of the change that moved it there: the
    # start method is no longer read from -- or patchable via -- global state.
    #
    # Substituting it is also what keeps this test honest about what it
    # measures. `orch.mp_context` is a real spawn context, and spawn *pickles*
    # the process target, so `_FakeProcess` closing over `captured_configs`
    # could not cross the boundary. Patching `mp.Process` used to work only
    # because the interpreter default was fork here, which silently made this a
    # test of fork behavior in a pipeline that must never fork.
    fake_context = MagicMock()
    fake_context.Process = _FakeProcess
    fake_context.Manager.return_value.Queue.return_value = MagicMock()
    orch.mp_context = fake_context

    orch._run_pipeline([("chunk.smi", "job1")])

    # The shared config the caller passed in must be untouched.
    assert config.batchsize_atoms == 1024
    # The optimization worker must receive the memory-scaled batchsize.
    opt_configs = [c for c in captured_configs if c.batchsize_atoms == 1024 * 4]
    assert opt_configs, "optimizer did not receive the memory-scaled batchsize"


def test_two_runs_do_not_reuse_job_name(tmp_path, monkeypatch):
    """A second main(args) call in the same process must not reuse the first
    run's job_name (M16).

    main() builds a fresh WorkflowOrchestrator(args) on every call but the
    two calls share the same Auto3DOptions object. Before this fix, run()
    validated and mutated that shared object directly (job_name/input_format,
    see _validate_input), so a second call would see job_name already
    non-empty and skip generating its own -- silently reusing the first
    run's job_name. run() now copies the shared config once at its own top,
    so each run's mutations land on a private copy and the shared object
    passed in is never touched.

    Exercises WorkflowOrchestrator directly (constructing it twice with the
    same shared config, exactly as two main(args) calls would) rather than
    calling main() twice end-to-end: a real run loads an optimizing model and
    forks worker processes, both disallowed on this box. Only Phase 1
    (_validate_input) is relevant to this defect, so every later phase is
    stubbed to stop the pipeline the moment it is reached -- optimizing_engine
    is pinned to 'ANI2xt' (bundled, no registry/network lookup) so even Phase
    1's real preflight_model check stays offline.
    """
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    smi = tmp_path / "mol.smi"
    smi.write_text("CCO ethanol\n")
    shared_config = Auto3DOptions(path=str(smi), k=1, use_gpu=False, optimizing_engine="ANI2xt")
    assert shared_config.job_name == ""

    class _StopAfterValidateError(Exception):
        pass

    def _stub_setup_job_directory(self):
        raise _StopAfterValidateError()

    monkeypatch.setattr(WorkflowOrchestrator, "_setup_job_directory", _stub_setup_job_directory)

    orch1 = WorkflowOrchestrator(shared_config)
    with pytest.raises(_StopAfterValidateError):
        orch1.run()
    first_job_name = orch1.config.job_name
    assert first_job_name != ""

    orch2 = WorkflowOrchestrator(shared_config)
    with pytest.raises(_StopAfterValidateError):
        orch2.run()
    second_job_name = orch2.config.job_name
    assert second_job_name != ""

    assert second_job_name != first_job_name, (
        "second run reused the first run's job_name -- the shared config "
        "object was mutated in place (M16)"
    )
    # The object the caller still holds a reference to must show no trace of
    # either run's mutation.
    assert shared_config.job_name == ""


def test_smiles2mols_uses_args_threshold(monkeypatch):
    """smiles2mols must pass args.threshold (not a hardcoded value) to the
    isomer engine, matching main()'s candidate-pool behavior (review #35/#36)."""
    import Auto3D.entry.auto3D as auto3D_mod
    from Auto3D.foundation.config import Auto3DOptions

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
        def __init__(self, in_f, out_f, *args, **kwargs):
            self._out_f = out_f

        def run(self):
            # find_smiles_not_in_sdf (C7 reconciliation, now wired into
            # smiles2mols) reads this file, so it must be a real SDF -- the
            # real ranking.run() always writes one, even a valid empty one
            # would still not parse (RDKit rejects a 0-byte SDF), so write
            # the one molecule this test actually asks for.
            from rdkit import Chem

            with Chem.SDWriter(self._out_f) as w:
                mol = Chem.MolFromSmiles("CCO")
                mol.SetProp("_Name", "stub")
                w.write(mol)
            return []

    monkeypatch.setattr(auto3D_mod.IsomerEngineFactory, "create", staticmethod(_capture_create))
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
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    smi = tmp_path / "m.smi"
    smi.write_text("CCO ethanol\n")
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(smi), k=1, use_gpu=False))
    orch._validate_input()
    assert orch.config.input_format == "smi"
    assert not hasattr(orch, "input_format")


def _encoded_mol(encoded_id):
    """A minimal mol shaped like decode_ids expects: numeric _Name + ID."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C")
    mol.SetProp("_Name", str(encoded_id))
    mol.SetProp("ID", f"{encoded_id}_conf1")
    return mol


class TestFinalizeOutputReconciliation:
    """C7: _finalize_output must compare input against output and report gaps.

    These pin the reconciliation wired into _finalize_output/_reconcile_output
    directly (the real production call site), rather than only exercising
    find_smiles_not_in_sdf/find_ids_not_in_sdf in isolation -- that is exactly
    what would fail to catch a regression back to "zero production callers".
    """

    def test_smi_input_reports_missing_id_and_sets_failures(self, tmp_path, caplog):
        """mol_c (encoded id 2) never produced a chunk output -> reported."""
        import logging

        from rdkit import Chem

        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        orig_smi = tmp_path / "orig.smi"
        # Original (decoded) ids -- this, not the encoded temp file, is what
        # reconciliation must compare against.
        orig_smi.write_text("C mol_a\nC mol_b\nC mol_c\n")

        config = Auto3DOptions(path=str(orig_smi), k=1, use_gpu=False)
        config.input_format = "smi"
        orch = WorkflowOrchestrator(config)
        orch.job_dir = tmp_path
        orch.input_path = tmp_path / "orig_encoded.smi"
        orch.input_path.write_text("C 0\nC 1\nC 2\n")
        orch.id_mapping = {"mol_a": 0, "mol_b": 1, "mol_c": 2}
        orch.logger = None

        job = tmp_path / "job1"
        job.mkdir()
        combined = job / "orig_encoded_3d.sdf"
        with Chem.SDWriter(str(combined)) as w:
            for encoded_id in (0, 1):  # id 2 (mol_c) never converged
                w.write(_encoded_mol(encoded_id))

        with caplog.at_level(logging.WARNING):
            path_output = orch._finalize_output(start_time=0.0)

        assert orch.failures == ["mol_c"], orch.failures
        assert any("mol_c" in r.message for r in caplog.records), (
            "the missing id was not logged anywhere"
        )

        produced = {m.GetProp("_Name") for m in Chem.SDMolSupplier(path_output) if m is not None}
        assert produced == {"mol_a", "mol_b"}

    def test_smi_input_reports_no_failures_when_everything_present(self, tmp_path):
        """No false positives when every input molecule made it to the output."""
        from rdkit import Chem

        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        orig_smi = tmp_path / "orig.smi"
        orig_smi.write_text("C mol_a\nC mol_b\n")

        config = Auto3DOptions(path=str(orig_smi), k=1, use_gpu=False)
        config.input_format = "smi"
        orch = WorkflowOrchestrator(config)
        orch.job_dir = tmp_path
        orch.input_path = tmp_path / "orig_encoded.smi"
        orch.input_path.write_text("C 0\nC 1\n")
        orch.id_mapping = {"mol_a": 0, "mol_b": 1}
        orch.logger = None

        job = tmp_path / "job1"
        job.mkdir()
        combined = job / "orig_encoded_3d.sdf"
        with Chem.SDWriter(str(combined)) as w:
            for encoded_id in (0, 1):
                w.write(_encoded_mol(encoded_id))

        orch._finalize_output(start_time=0.0)
        assert orch.failures == []

    def test_sdf_input_reports_missing_id_and_sets_failures(self, tmp_path):
        """SDF input must be reconciled too, not silently skipped (C7 scope)."""
        from rdkit import Chem

        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        orig_sdf = tmp_path / "orig.sdf"
        with Chem.SDWriter(str(orig_sdf)) as w:
            for name in ("mol_a", "mol_b", "mol_c"):
                mol = Chem.MolFromSmiles("C")
                mol.SetProp("_Name", name)
                w.write(mol)

        config = Auto3DOptions(path=str(orig_sdf), k=1, use_gpu=False)
        config.input_format = "sdf"
        orch = WorkflowOrchestrator(config)
        orch.job_dir = tmp_path
        orch.input_path = tmp_path / "orig_encoded.sdf"
        with Chem.SDWriter(str(orch.input_path)) as w:
            for encoded_id in (0, 1, 2):
                w.write(_encoded_mol(encoded_id))
        orch.id_mapping = {"mol_a": 0, "mol_b": 1, "mol_c": 2}
        orch.logger = None

        job = tmp_path / "job1"
        job.mkdir()
        combined = job / "orig_encoded_3d.sdf"
        with Chem.SDWriter(str(combined)) as w:
            for encoded_id in (0, 1):  # id 2 (mol_c) never converged
                w.write(_encoded_mol(encoded_id))

        orch._finalize_output(start_time=0.0)
        assert orch.failures == ["mol_c"], orch.failures

    def test_workflow_uses_the_canonical_reconciliation_functions(self, monkeypatch, tmp_path):
        """Guard against a regression to a hand-rolled duplicate: `_reconcile_output`
        must actually call the imported `find_smiles_not_in_sdf`/`find_ids_not_in_sdf`
        at its call site, not merely import them.

        An identity check on the module attribute alone
        (`workflow.find_smiles_not_in_sdf is reconciliation.find_smiles_not_in_sdf`)
        cannot catch a regression to a private reimplementation: the import
        binding survives untouched even if `_reconcile_output` is changed to
        call a different, local function instead, since nothing then
        references the import at all. Spying on the name actually resolved in
        `workflow`'s module globals at call time closes that gap: it fails if
        `_reconcile_output` stops dispatching through it, whether or not the
        unused import is still there.
        """
        import Auto3D.orchestration.workflow as workflow
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        smi_calls = []
        id_calls = []
        monkeypatch.setattr(
            workflow,
            "find_smiles_not_in_sdf",
            lambda *a, **kw: smi_calls.append((a, kw)) or [],
        )
        monkeypatch.setattr(
            workflow,
            "find_ids_not_in_sdf",
            lambda *a, **kw: id_calls.append((a, kw)) or [],
        )

        config = Auto3DOptions(path=str(tmp_path / "in.smi"), k=1, use_gpu=False)
        orch = WorkflowOrchestrator(config)
        orch.logger = None

        config.input_format = "smi"
        orch._reconcile_output(str(tmp_path / "out.sdf"))
        assert smi_calls, "find_smiles_not_in_sdf was not called from _reconcile_output"
        assert not id_calls

        smi_calls.clear()
        config.input_format = "sdf"
        orch._reconcile_output(str(tmp_path / "out.sdf"))
        assert id_calls, "find_ids_not_in_sdf was not called from _reconcile_output"
        assert not smi_calls


def test_main_propagates_orchestrator_failures_into_workflow_result(monkeypatch, tmp_path):
    """main() must surface WorkflowOrchestrator.failures on its returned
    WorkflowResult -- the carrier a later CLI fix reads to populate
    results.failures and drive a non-zero exit code. Wires main() end-to-end
    without a real pipeline run: WorkflowOrchestrator.run() (the isomer/optim/
    finalize phases) is stubbed, but the WorkflowResult construction and the
    getattr(out, "failures", ...) contract the C7 tripwire relies on are real.
    """
    from Auto3D.entry.auto3D import main
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.results import WorkflowResult
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    fake_output = str(tmp_path / "out.sdf")

    def fake_run(self):
        self.failures = ["mol_c"]
        return fake_output

    monkeypatch.setattr(WorkflowOrchestrator, "run", fake_run)

    smi = tmp_path / "in.smi"
    smi.write_text("C mol_a\nC mol_b\nC mol_c\n")
    args = Auto3DOptions(path=str(smi), k=1, use_gpu=False)

    result = main(args)

    assert isinstance(result, WorkflowResult)
    assert str(result) == fake_output
    assert result.failures == ["mol_c"]
    # getattr access, exactly as the C7 tripwire and the CLI use it.
    assert getattr(result, "failures", None) == ["mol_c"]


def test_smiles2mols_calls_find_smiles_not_in_sdf_and_reports_missing(monkeypatch, caplog):
    """smiles2mols must reconcile its SMILES input against what it produced,
    the same way main()/_finalize_output do, and the report must name the
    molecule that vanished -- proven against the real find_smiles_not_in_sdf,
    not a stand-in, so a regression to zero callers would fail this test."""
    import logging

    from rdkit import Chem
    from rdkit.Chem import inchi

    import Auto3D.entry.auto3D as auto3D_mod
    from Auto3D.foundation.config import Auto3DOptions

    ethanol_id = inchi.MolToInchiKey(Chem.MolFromSmiles("CCO"))
    written: dict[str, str] = {}

    class _StubIsomerEngine:
        def run(self):
            return None

    def _capture_create(**kwargs):
        return _StubIsomerEngine()

    class _StubOpt:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

    class _StubRank:
        def __init__(self, in_f, out_f, threshold, k=None, window=None):
            written["out_f"] = out_f

        def run(self):
            # Only ethanol "converges"; propanol vanishes mid-pipeline with no
            # trace other than what reconciliation now reports.
            with Chem.SDWriter(written["out_f"]) as w:
                mol = Chem.MolFromSmiles("CCO")
                mol.SetProp("_Name", ethanol_id)
                w.write(mol)
            return []

    monkeypatch.setattr(auto3D_mod.IsomerEngineFactory, "create", staticmethod(_capture_create))
    monkeypatch.setattr(auto3D_mod, "optimizing", _StubOpt)
    monkeypatch.setattr(auto3D_mod, "ranking", _StubRank)
    monkeypatch.setattr(auto3D_mod, "reorder_sdf", lambda *a, **k: [])

    calls = []
    real_find = auto3D_mod.find_smiles_not_in_sdf

    def spy(smi_path, sdf_path):
        result = real_find(smi_path, sdf_path)
        calls.append((smi_path, sdf_path, result))
        return result

    monkeypatch.setattr(auto3D_mod, "find_smiles_not_in_sdf", spy)

    args = Auto3DOptions(k=1, use_gpu=False)
    with caplog.at_level(logging.WARNING):
        auto3D_mod.smiles2mols(["CCO", "CCC"], args)

    assert calls, (
        "find_smiles_not_in_sdf was never called by smiles2mols -- regression "
        "to zero production callers (C7)"
    )
    _smi_path, _sdf_path, bad = calls[0]
    missing_ids = [mol_id for mol_id, _smi in bad]
    assert ethanol_id not in missing_ids
    assert len(missing_ids) == 1
    assert any(missing_ids[0] in r.message for r in caplog.records)


class TestQuietPathsNameWhatTheyDropped:
    """Two readers dropped molecules more quietly than their siblings.

    Both are the same defect: a code path that loses a molecule and says less
    about it than another path doing the identical thing, so how much the user is
    told depends on which door they came through.
    """

    def test_the_optimizer_names_each_record_it_could_not_parse(
        self, tmp_path, caplog, monkeypatch
    ):
        """`optimizing` logged only the all-records-failed case.

        A single bad record among a thousand left the output SDF shorter than the
        input with nothing said about which one -- for `opt_geometry`, that is a
        short file, the path returned, and exit 0. The only trace was RDKit's own
        C++ parse error, which names a file offset rather than a molecule.
        `SPE.calc_spe` and `ASE/thermo`'s `iter_thermo_records` both log
        per-record for exactly this; this reader did not.
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.engines.batch_opt.batchopt import optimizing
        from tests.helpers_adapter import FakeAdapter

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        mol.SetProp("_Name", "mol_a")
        block = Chem.MolToMolBlock(mol).splitlines()
        block[3] = "!! corrupted counts line !!"
        # Every record unparseable, so this returns before any model is needed --
        # the per-record warning under test happens while reading the file.
        bad_sdf = tmp_path / "bad.sdf"
        bad_sdf.write_text("\n".join(block) + "\n$$$$\n")

        config = {
            "opt_steps": 100,
            "opttol": 0.003,
            "patience": 100,
            "batchsize_atoms": 1024,
        }
        optimizer = optimizing(
            str(bad_sdf),
            str(tmp_path / "out.sdf"),
            adapter=FakeAdapter(),
            device=torch.device("cpu"),
            config=config,
        )

        with caplog.at_level(logging.WARNING):
            optimizer.run()

        assert "index 0" in caplog.text, (
            "the unparseable record was dropped without being named; only the "
            f"all-failed case was reported. Log was: {caplog.text!r}"
        )

    def test_the_parallel_embed_path_names_a_species_it_produced_nothing_for(self, caplog):
        """The serial path warns twice here; the parallel path warned not at all.

        `_embed_single` returned `[]` for an unparseable SMILES in silence, and
        `_run_parallel_embedding` had no counterpart to the serial path's
        `n_written == 0` warning. So `use_parallel_embedding` -- documented as a
        performance option -- decided whether a lost species was reported.

        The warning asserted here is the parent-side one, which is the guaranteed
        signal: a message logged inside a ProcessPoolExecutor worker depends on
        that child's logging configuration, and this one does not.
        """
        from Auto3D.domain.embedding import embed_conformers_parallel

        with caplog.at_level(logging.WARNING):
            results = list(
                embed_conformers_parallel(
                    [("this-is-not-a-smiles", "bad_mol")],
                    n_conformers=1,
                    n_workers=1,
                )
            )

        assert results == [], "test premise: an unparseable SMILES embeds nothing"
        assert "bad_mol" in caplog.text, (
            f"a species that produced no conformers was absent from the output "
            f"with nothing logged. Log was: {caplog.text!r}"
        )

    def test_a_species_that_embeds_normally_is_not_warned_about(self, caplog):
        """The new branch must not fire for a molecule that worked."""
        from Auto3D.domain.embedding import embed_conformers_parallel

        with caplog.at_level(logging.WARNING):
            results = list(
                embed_conformers_parallel([("CCO", "ethanol")], n_conformers=2, n_workers=1)
            )

        assert results, "test premise: ethanol should embed"
        assert "produced no conformers" not in caplog.text
