#!/usr/bin/env python
"""Tests for workflow orchestration, including multi-GPU handling."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import queue
import signal
import threading
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

    def test_prepare_chunks_raises_input_validation_error_on_empty_input(self, tmp_path):
        """A 0-molecule input must be diagnosed as such (exit 2, `auto3d
        validate` hint), not left to surface three phases later as a
        misleading 'no chunk produced a 3D structure output file' /
        OptimizationError (exit 1) once _finalize_output finds nothing.

        ChunkManager.prepare_chunks() returns [] silently for a 0-record
        input: every chunk ends up empty and _create_chunk_files (audit
        location chunk_manager.py:276-280) skips all of them. _prepare_chunks
        must catch that emptiness itself.
        """
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.foundation.exceptions import InputValidationError
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        smi = tmp_path / "empty.smi"
        smi.write_text("")  # 0 molecules

        config = Auto3DOptions(path=str(smi), k=1, use_gpu=False)
        config.input_format = "smi"
        orchestrator = WorkflowOrchestrator(config)
        orchestrator.job_dir = tmp_path
        orchestrator.input_path = smi
        orchestrator.logger = None

        with pytest.raises(InputValidationError, match="no molecules"):
            orchestrator._prepare_chunks()


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


@contextlib.contextmanager
def _restored_worker_globals():
    """Run a worker body in-process without leaking its process-global writes.

    Both wrappers touch state that outlives the call: ``_attach_run_log_handlers``
    adds a ``QueueHandler`` to the "auto3d" and "Auto3D" trees, and
    ``optim_rank_wrapper`` additionally runs ``configure_torch``, which writes the
    tf32 booleans and (torch >= 2.9) the ``fp32_precision`` knob -- note
    ``torch.backends.cudnn.allow_tf32`` defaults to True, so even
    ``allow_tf32=False`` changes it. Same snapshot/restore discipline as
    ``test_optim_rank_wrapper_applies_torch_config`` below.
    """
    import torch

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_matmul_fp32 = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    previous_cudnn_fp32 = getattr(torch.backends.cudnn, "fp32_precision", None)
    loggers = (logging.getLogger("auto3d"), logging.getLogger("Auto3D"))
    previous_handlers = {logger: list(logger.handlers) for logger in loggers}
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        if previous_matmul_fp32 is not None:
            torch.backends.cuda.matmul.fp32_precision = previous_matmul_fp32
        if previous_cudnn_fp32 is not None:
            torch.backends.cudnn.fp32_precision = previous_cudnn_fp32
        for logger in loggers:
            before = previous_handlers[logger]
            for handler in list(logger.handlers):
                if handler not in before:
                    logger.removeHandler(handler)
                    handler.close()
            logger.handlers[:] = before


class _InterruptingQueue:
    """A chunk queue that interrupts the worker the way a Ctrl-C does.

    A process-group SIGINT raises ``KeyboardInterrupt`` wherever the worker
    happens to be, and for an optimizer that is almost always the blocking
    ``queue.get()`` between chunks.
    """

    def get(self):
        raise KeyboardInterrupt


class TestWorkersExitQuietlyOnKeyboardInterrupt:
    """P-M12: Ctrl-C must not print a traceback per worker.

    Both wrappers convert ``KeyboardInterrupt`` into ``SystemExit(130)``, which
    ``multiprocessing``'s ``_bootstrap`` treats as a clean exit code instead of
    dumping a traceback on the shared stderr. Exercised in-process -- the
    ``multiprocessing.parent_process() is None`` guard in
    ``_exit_when_parent_dies`` makes that safe -- because the spawned-subprocess
    harness in ``tests/test_worker_lifecycle.py`` substitutes stub workers and so
    never runs these two clauses.
    """

    def test_optim_rank_wrapper_exits_130_instead_of_raising(self, tmp_path, monkeypatch):
        import Auto3D.orchestration.workflow_workers as ww
        from Auto3D.foundation.config import Auto3DOptions
        from tests.helpers_adapter import FakeAdapter

        # The adapter is built once, before the first `queue.get()` (C-2), so
        # this in-process call would otherwise load a real AIMNet2 model just to
        # reach the interrupt the test is about.
        monkeypatch.setattr(ww, "create_model", lambda *a, **k: FakeAdapter())

        args = Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, use_gpu=False)
        with _restored_worker_globals(), pytest.raises(SystemExit) as excinfo:
            ww.optim_rank_wrapper(
                args,
                args.to_optimization_config(batchsize_atoms=args.batchsize_atoms),
                _InterruptingQueue(),
                queue.Queue(),
                gpu_idx=0,
            )
        assert excinfo.value.code == 130

    def test_isomer_wrapper_exits_130_and_still_wakes_every_optimizer(self, tmp_path, monkeypatch):
        """The quiet exit must not cost the sentinels: the `finally` still runs."""
        import Auto3D.orchestration.workflow_workers as ww
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.processors import TautomerProcessor

        def _interrupt(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(TautomerProcessor, "process", _interrupt)

        args = Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, gpu_idx=[0, 1])
        args.input_format = "smi"
        chunk_queue = queue.Queue()
        with _restored_worker_globals(), pytest.raises(SystemExit) as excinfo:
            ww.isomer_wrapper(
                [(str(tmp_path / "chunk.smi"), str(tmp_path))],
                args,
                chunk_queue,
                queue.Queue(),
            )
        assert excinfo.value.code == 130

        drained = []
        while not chunk_queue.empty():
            drained.append(chunk_queue.get())
        assert drained.count("Done") == 2, (
            f"a quiet interrupt must still wake every optimizer; queue held {drained!r}"
        )


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
    from tests.helpers_adapter import FakeAdapter

    attempted = []

    class _BoomOptimizing:
        def __init__(self, in_f, out_f, *, adapter, device, config, progress_cb=None):
            self._enumerated = in_f

        def run(self):
            attempted.append(self._enumerated)
            raise RuntimeError("optimizer blew up on this chunk")

    # Replace the heavy optimizing class with one that always raises, so we
    # exercise only the loop's failure isolation. `create_model` is stubbed for
    # the same reason: since C-2 the adapter is built once before the loop, and
    # a real AIMNet2 load has nothing to do with per-chunk isolation.
    monkeypatch.setattr(ww, "optimizing", _BoomOptimizing)
    monkeypatch.setattr(ww, "create_model", lambda *a, **k: FakeAdapter())

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
    result = ww.optim_rank_wrapper(
        args, args.to_optimization_config(batchsize_atoms=args.batchsize_atoms), q, logq, gpu_idx=0
    )
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


def test_model_construction_failure_is_fatal_and_consumes_no_chunk(tmp_path, monkeypatch):
    """C-2: a model that cannot be built is not a skippable chunk.

    ``create_model`` used to run INSIDE the per-chunk ``except Exception:
    continue``, so a ``NumericalError`` from the compile probe -- or a bad
    checksum, or a missing weight file -- was logged once per chunk and the
    worker went on to "skip" every one of them. The run then ended on
    ``_finalize_output``'s convergence-failure message over an empty or partial
    SDF, naming three causes (memory, invalid SMILES, patience) that did not
    apply. Construction now happens once, before the first ``queue.get()``, and
    its failure propagates out of the worker.
    """
    import queue as queue_mod

    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.exceptions import NumericalError
    from Auto3D.orchestration import workflow_workers as ww

    def _refuse(*args_, **kwargs_):
        raise NumericalError("the compiled adapter disagrees with eager")

    monkeypatch.setattr(ww, "create_model", _refuse)
    # A tripwire, not a stub: if construction were still inside the loop, the
    # worker would swallow the NumericalError and optimize the chunk anyway.
    # ``pytest.fail`` raises a BaseException, so the per-chunk ``except
    # Exception`` cannot hide it either.
    monkeypatch.setattr(
        ww,
        "optimizing",
        lambda *args_, **kwargs_: pytest.fail("a chunk was optimized with no model"),
    )

    q: queue_mod.Queue = queue_mod.Queue()
    q.put(("enum1.sdf", str(tmp_path / "c1.smi"), str(tmp_path), 1))
    q.put(("enum2.sdf", str(tmp_path / "c2.smi"), str(tmp_path), 2))
    q.put("Done")
    args = Auto3DOptions(path="x.smi", k=1, use_gpu=False)

    with _restored_worker_globals(), pytest.raises(NumericalError):
        ww.optim_rank_wrapper(
            args,
            args.to_optimization_config(batchsize_atoms=args.batchsize_atoms),
            q,
            queue_mod.Queue(),
            gpu_idx=0,
        )

    # Nothing was taken off the queue: both chunks and the sentinel are still
    # there. (The parent tops the sentinels up itself; see _ensure_done_sentinels.)
    assert q.qsize() == 3


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

    # The streaming combine (Issue 13) writes the combined file incrementally
    # and only discovers "nothing converged" once every chunk has been
    # streamed through it -- it must still clean up the resulting empty
    # combined file before raising, matching the "no chunk produced output"
    # branch above, which never creates one at all.
    assert not (tmp_path / "x_encoded_out.sdf").exists()


def test_finalize_output_streams_chunks_in_order(tmp_path):
    """The combined output must contain every chunk's molecules, in chunk
    order, exactly as the old read-everything-then-join implementation
    produced -- pinning that streaming the combine line-by-line (Issue 13)
    did not change what gets written, only how much memory it takes.
    """
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

    job1 = tmp_path / "job1"
    job1.mkdir()
    job2 = tmp_path / "job2"
    job2.mkdir()
    with Chem.SDWriter(str(job1 / "orig_encoded_3d.sdf")) as w:
        w.write(_encoded_mol(0))
    with Chem.SDWriter(str(job2 / "orig_encoded_3d.sdf")) as w:
        w.write(_encoded_mol(1))

    path_output = orch._finalize_output(start_time=0.0)

    produced = [m.GetProp("_Name") for m in Chem.SDMolSupplier(path_output) if m is not None]
    assert produced == ["mol_a", "mol_b"]


def test_run_pipeline_does_not_mutate_shared_batchsize():
    """The memory-scaled batch size reaches the optimizer as an
    ``OptimizationConfig``, and no ``Auto3DOptions`` anywhere carries it.

    ``batchsize_atoms`` means two different things on the two classes:
    per gigabyte of measured memory on ``Auto3DOptions``, absolute on
    ``OptimizationConfig``. _run_pipeline used to bridge them with
    ``self.config.replace(batchsize_atoms=self.scaled_batchsize_atoms)`` -- an
    ``Auto3DOptions`` copy holding an absolute number in a per-gigabyte field,
    which each worker then multiplied no further only because
    ``to_optimization_config()`` happened to copy it straight across. Any worker
    that read the field for its own purpose would have read a 4x figure, and the
    copy also had to exist solely so the caller's shared config stayed at 1024
    (review #35/#36). Now the parent builds the absolute ``OptimizationConfig``
    once and hands it over as its own argument; every ``Auto3DOptions`` in every
    process keeps the per-gigabyte value.
    """
    from Auto3D.foundation.config import (
        Auto3DOptions,
        OptimizationConfig,
        optimizer_worker_indices,
    )
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    config = Auto3DOptions(path="x.smi", k=1, batchsize_atoms=1024)
    orch = WorkflowOrchestrator(config)
    # Simulate the memory scaling that _prepare_chunks would have computed.
    orch.scaled_batchsize_atoms = 1024 * 4
    orch.logging_queue = MagicMock()

    captured_configs = []
    captured_opt_configs = []

    class _FakeProcess:
        def __init__(self, target=None, args=(), **kwargs):
            self._args = args

        def start(self):
            # Record both config objects passed to a worker (positions differ
            # between the isomer and optimization workers, and only the
            # optimization workers get an OptimizationConfig at all).
            captured_configs.extend(a for a in self._args if isinstance(a, Auto3DOptions))
            captured_opt_configs.extend(a for a in self._args if isinstance(a, OptimizationConfig))

        def join(self, timeout=None):
            return None

        # _run_pipeline's `finally` terminates whatever it started, so the fake
        # has to answer the three calls _terminate_workers makes. "Already
        # finished" is the honest answer for a process whose start() only
        # recorded its arguments: terminate()/kill() are then never reached.
        def is_alive(self):
            return False

        def terminate(self):
            return None

        def kill(self):
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
    # _run_pipeline no longer calls mp_context.Manager() directly: _start_manager
    # constructs a real SyncManager against the context so it can pass an
    # initializer (see its docstring), and a real SyncManager cannot be built on
    # a MagicMock context. Redirecting the one seam keeps the fake context above
    # as the single place this test describes the multiprocessing world.
    orch._start_manager = fake_context.Manager

    orch._run_pipeline([("chunk.smi", "job1")])

    # The shared config the caller passed in must be untouched.
    assert config.batchsize_atoms == 1024
    # And so must every copy of it that crossed the spawn boundary: the scaled
    # value has no business in a per-gigabyte field.
    assert captured_configs, "no worker received an Auto3DOptions at all"
    assert [c.batchsize_atoms for c in captured_configs] == [1024] * len(captured_configs)
    # Exactly the optimizer workers -- one per index, and not the isomer worker
    # -- receive the absolute batch size, as an OptimizationConfig.
    n_optimizers = len(optimizer_worker_indices(config.use_gpu, config.gpu_idx))
    assert len(captured_opt_configs) == n_optimizers, (
        f"{len(captured_opt_configs)} workers got an OptimizationConfig, "
        f"expected {n_optimizers} (the optimizer workers and no one else)"
    )
    assert [c.batchsize_atoms for c in captured_opt_configs] == [1024 * 4] * n_optimizers, (
        "optimizer did not receive the memory-scaled batchsize"
    )


class TestAbnormalIsomerWorkerExit:
    """Issue 3: an isomer worker that dies without running its `finally`
    (SIGKILL from the OOM killer, a segfault in RDKit/Boost, os._exit -- none
    of which give Python a chance to run cleanup code) must not leave the
    optimizer workers blocked on ``queue.get()`` forever. Reproduced upstream
    as worker exitcode -9, the optimizer wedged indefinitely.

    ``_ThreadProcess`` stands in for ``mp_context.Process``, backed by a real
    thread rather than being fully inert (contrast
    ``test_run_pipeline_does_not_mutate_shared_batchsize``'s ``_FakeProcess``,
    whose ``join()`` never blocks on anything): ``p1`` and the optimizer
    stand-in genuinely run concurrently and genuinely block on a real
    ``queue.Queue.get()``. That is deliberate -- if the fix regresses, this
    test really hangs, which is why it carries its own ``pytest.mark.timeout``
    bound instead of relying on the suite's overall time limit to catch it.
    A real spawned ``multiprocessing.Process`` was avoided: the stand-in
    targets below would need to be pickled by reference for ``spawn``, which
    means being importable top-level functions in a fresh interpreter --
    fragile in a pytest worker -- for no benefit here, since the behavior
    under test (``_run_pipeline``'s supervisory logic) does not depend on
    workers being real OS processes, only on them being genuinely concurrent.
    """

    class _ThreadProcess:
        """Stands in for ``mp_context.Process``, backed by a thread.

        ``exitcode`` is set from whatever the target callable returns: the
        deliberately-abnormal isomer stand-in below returns a negative int,
        the way a real SIGKILL's exitcode would read.
        """

        def __init__(self, target=None, args=(), **kwargs):
            self._target = target
            self._args = args
            self._thread: threading.Thread | None = None
            self.exitcode = None
            self.pid = None
            self.terminated = False

        def start(self) -> None:
            def _run():
                self.exitcode = self._target(*self._args)

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            self.pid = self._thread.ident

        def is_alive(self) -> bool:
            return self._thread is not None and self._thread.is_alive()

        def join(self, timeout=None) -> None:
            if self._thread is not None:
                self._thread.join(timeout)

        def terminate(self) -> None:
            # Real threads cannot be force-killed; record the request so a
            # test could assert the backstop path was reached, if it ever is.
            self.terminated = True

    @staticmethod
    def _dying_isomer(chunk_info, config, chunk_queue, logging_queue):
        """A stand-in for isomer_wrapper that simulates a SIGKILL: exits with
        an abnormal code and never touches chunk_queue at all -- no "Done"
        sentinel, exactly as a real SIGKILL/segfault would skip
        isomer_wrapper's `finally` (workflow_workers.py:209-212).
        """
        return -9

    @staticmethod
    def _draining_optimizer(
        config, opt_config, chunk_queue, logging_queue, gpu_idx, progress_queue=None
    ):
        """A stand-in for optim_rank_wrapper's consume loop -- only the part
        under test (blocking on queue.get() until a "Done" sentinel) matters
        here; no model, no isomer/optimization work.

        The parameter list has to match production exactly, ``opt_config``
        included: _run_pipeline spawns this through the same positional
        ``args=(...)`` tuple it builds for the real worker, so a stale signature
        is a TypeError in the stand-in rather than a visible failure of what the
        test is about.
        """
        while True:
            item = chunk_queue.get()
            if item == "Done":
                return 0

    @pytest.mark.timeout(10)
    def test_run_pipeline_recovers_when_isomer_worker_dies_abnormally(self, monkeypatch):
        """Without _ensure_done_sentinels, this test hangs forever: the
        optimizer stand-in blocks on chunk_queue.get() because the dying
        isomer stand-in never puts a "Done" sentinel, and nothing else would
        ever unblock it. The pytest-timeout bound above turns a regression
        into a fast, clear failure instead of an indefinitely stuck suite.
        """
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        config = Auto3DOptions(path="x.smi", k=1, use_gpu=False)
        orch = WorkflowOrchestrator(config)
        orch.logging_queue = None

        real_chunk_queue: queue.Queue = queue.Queue()
        fake_context = MagicMock()
        fake_context.Process = self._ThreadProcess
        fake_context.Manager.return_value.Queue.return_value = real_chunk_queue
        orch.mp_context = fake_context
        # See test_run_pipeline_does_not_mutate_shared_batchsize: _start_manager
        # is the seam now, and a real SyncManager cannot be built on a MagicMock.
        orch._start_manager = fake_context.Manager

        monkeypatch.setattr(Auto3D.orchestration.workflow, "isomer_wrapper", self._dying_isomer)
        monkeypatch.setattr(
            Auto3D.orchestration.workflow, "optim_rank_wrapper", self._draining_optimizer
        )

        # Must return -- not hang -- within the pytest.mark.timeout bound.
        orch._run_pipeline([("chunk.smi", "job1")])

        # The parent must have topped the queue up with the sentinel the
        # dying isomer worker never put, and nothing more: the single
        # optimizer worker (use_gpu=False -> one worker regardless of
        # gpu_idx) consumed exactly one "Done" and returned.
        assert real_chunk_queue.empty()

    @pytest.mark.timeout(10)
    def test_supervise_with_progress_recovers_when_isomer_worker_dies_abnormally(self, monkeypatch):
        """The progress-queue supervision loop (workflow.py's
        `_supervise_with_progress`, `while any(p.is_alive())`) is the subtler
        of the two hang sites the issue names: unlike the plain path's
        sequential joins, this loop's own exit condition never goes false
        while an optimizer is blocked on queue.get() forever, so sentinel
        injection has to happen from *inside* the loop, the moment p1 is
        seen to have died -- not after it, the way the non-progress path can
        afford to. Exercised via a real progress_callback so this actually
        takes the `_supervise_with_progress` branch of `_run_pipeline`.
        """
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        config = Auto3DOptions(path="x.smi", k=1, use_gpu=False)
        orch = WorkflowOrchestrator(config, progress_callback=lambda event: None)
        orch.logging_queue = None

        # Two distinct real queues: chunk_queue and progress_queue must not
        # be the same object, or the progress-draining calls in the loop
        # would consume the "Done" sentinel meant for the optimizer.
        chunk_q: queue.Queue = queue.Queue()
        progress_q: queue.Queue = queue.Queue()
        fake_context = MagicMock()
        fake_context.Process = self._ThreadProcess
        fake_context.Manager.return_value.Queue.side_effect = [chunk_q, progress_q]
        orch.mp_context = fake_context
        # Same seam as above; the side_effect order still holds because
        # _run_pipeline asks for the chunk queue before the progress queue.
        orch._start_manager = fake_context.Manager

        monkeypatch.setattr(Auto3D.orchestration.workflow, "isomer_wrapper", self._dying_isomer)
        monkeypatch.setattr(
            Auto3D.orchestration.workflow, "optim_rank_wrapper", self._draining_optimizer
        )

        # Must return -- not hang -- within the pytest.mark.timeout bound.
        orch._run_pipeline([("chunk.smi", "job1")])

        assert chunk_q.empty()


def test_two_runs_do_not_reuse_job_name(tmp_path, monkeypatch, stub_torchani_importable):
    """A second main(args) call in the same process must not reuse the first
    run's job_name (M16).

    Uses ``stub_torchani_importable`` so this test keeps running unchanged on
    the ani=false CI leg: ``optimizing_engine="ANI2xt"`` below is picked only
    to keep Phase 1's real ``preflight_model`` offline, and this test's
    assertion is about job_name reuse, not about the ANI2xt engine itself.

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


@pytest.mark.slow
def test_smiles2mols_uses_args_threshold(monkeypatch):
    """smiles2mols must pass args.threshold (not a hardcoded value) to the
    isomer engine, matching main()'s candidate-pool behavior (review #35/#36).

    Marked slow for its wall-clock cost (5.3 s on the 2026-10-02 durations
    run), not for GPU or network needs.
    """
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
            return True  # matches optimizing.run()'s real True-on-write contract

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
            return True  # matches optimizing.run()'s real True-on-write contract

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
        """`optimizing` used to log only the all-records-failed case.

        A single bad record among a thousand left the output SDF shorter than the
        input with nothing said about which one -- for `opt_geometry`, that is a
        short file, the path returned, and exit 0. The only trace was RDKit's own
        C++ parse error, which names a file offset rather than a molecule.
        `SPE.calc_spe` and `ASE/thermo` both logged per-record for exactly this;
        this reader did not. It now reads through the same
        `Auto3D.foundation.utils.sdf_io.iter_conformer_records` those callers use
        (N-C1), whose unparseable-record message says "record %d", not
        "index %d" -- a wording change the record-policy unification review
        confirmed is the only effect on this path.
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

        assert "record 0" in caplog.text, (
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


def test_optim_rank_wrapper_applies_torch_config(tmp_path, monkeypatch):
    """N-M8: torch.backends state is process-global and does not cross the spawn
    boundary, so the worker must apply allow_tf32 itself.

    ``configure_torch(allow_tf32=True)`` writes up to four process-global
    flags (``torch.backends.cuda.matmul.allow_tf32``,
    ``torch.backends.cudnn.allow_tf32``, and -- on torch >= 2.9 -- the modern
    ``fp32_precision`` knob on both), and ``_attach_run_log_handlers`` adds a
    ``QueueHandler`` to both the "auto3d" and "Auto3D" logger trees. All of
    that is process-wide state that outlives this test unless it is put back,
    which is exactly what ``_restored_worker_globals`` above exists to do --
    this test used to carry its own inline copy of that snapshot/restore block
    (C-11).
    """
    import queue

    import torch

    import Auto3D.orchestration.workflow_workers as ww
    from Auto3D.foundation.config import Auto3DOptions
    from tests.helpers_adapter import FakeAdapter

    # Since C-2 the adapter is built before the first `queue.get()`, so even a
    # "Done"-only queue would load a real AIMNet2 model without this.
    monkeypatch.setattr(ww, "create_model", lambda *a, **k: FakeAdapter())

    with _restored_worker_globals():
        torch.backends.cuda.matmul.allow_tf32 = False
        args = Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, use_gpu=False, allow_tf32=True)
        q = queue.Queue()
        q.put("Done")
        ww.optim_rank_wrapper(
            args,
            args.to_optimization_config(batchsize_atoms=args.batchsize_atoms),
            q,
            queue.Queue(),
            gpu_idx=0,
        )
        assert torch.backends.cuda.matmul.allow_tf32 is True


def test_shutdown_logging_removes_handler_and_stops_manager(tmp_path):
    """P-M10: one handler and one SyncManager leaked per run."""
    import logging
    import multiprocessing as mp

    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    def _managers():
        # A Manager's server is a SpawnProcess whose .name is "SyncManager-N";
        # type(c).__name__ would never match (the reviewer's repro printed .name).
        return sum(1 for c in mp.active_children() if c.name.startswith("SyncManager"))

    root = logging.getLogger("auto3d")
    handlers_before, managers_before = list(root.handlers), _managers()
    orch = WorkflowOrchestrator(Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, use_gpu=False))
    orch.job_dir = tmp_path
    try:
        orch._setup_logging()
        assert len(root.handlers) == len(handlers_before) + 1
        assert _managers() == managers_before + 1
    finally:
        orch._shutdown_logging()
    assert list(root.handlers) == handlers_before
    # Other tests may have leaked a Manager; compare to baseline, not zero.
    assert _managers() == managers_before


@contextlib.contextmanager
def _sigterm_disposition_restored():
    """Put the process's SIGTERM disposition back, whatever these tests did to it."""
    previous = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def _unused_sigterm_handler(signum, frame):  # pragma: no cover - never delivered
    raise AssertionError("this handler must never run")


class TestSigtermRaises:
    """C-4/A-3: the scoped SIGTERM handler had no direct test at all.

    ``_run_pipeline``'s worker-termination ``finally`` is the whole point of
    turning SIGTERM into ``SystemExit``: under the default disposition the
    process dies outright and the optimizer workers grind through the rest of
    the queue on the GPU. The three cases below are the install, and the two
    situations in which installing would be wrong.
    """

    def test_the_handler_is_installed_and_the_default_put_back(self):
        from Auto3D.foundation.constants import EXIT_TERMINATED
        from Auto3D.orchestration.workflow import _raise_on_sigterm, _sigterm_raises

        with _sigterm_disposition_restored():
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with _sigterm_raises():
                assert signal.getsignal(signal.SIGTERM) is _raise_on_sigterm
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL

        # The code the handler raises is the shell's "killed by SIGTERM".
        with pytest.raises(SystemExit) as excinfo:
            _raise_on_sigterm(signal.SIGTERM, None)
        assert excinfo.value.code == EXIT_TERMINATED == 143

    def test_a_host_that_already_owns_sigterm_keeps_its_handler(self):
        """A-3(a): a non-default disposition means something to somebody.

        A service supervisor draining requests, an outer CLI, a notebook
        kernel: replacing its handler for the duration of one ``main()`` is not
        a library's call to make. Nothing is lost by declining -- every worker
        arms PR_SET_PDEATHSIG and a parent-sentinel watchdog of its own, so the
        workers still die with the parent however the parent goes.
        """
        from Auto3D.orchestration.workflow import _sigterm_raises

        with _sigterm_disposition_restored():
            signal.signal(signal.SIGTERM, _unused_sigterm_handler)
            with _sigterm_raises():
                assert signal.getsignal(signal.SIGTERM) is _unused_sigterm_handler
            assert signal.getsignal(signal.SIGTERM) is _unused_sigterm_handler

    def test_off_the_main_thread_it_is_a_no_op_and_leaves_nothing_behind(self):
        """``signal.signal`` raises ValueError anywhere but the main thread, and
        Auto3D is a library that may well be called from someone else's worker
        thread -- so the block must yield quietly and touch nothing.

        A-3(b): this is also the branch that used to be able to leave Auto3D's
        own handler installed after ``run()`` returned.
        """
        from Auto3D.orchestration.workflow import _sigterm_raises

        before = signal.getsignal(signal.SIGTERM)
        escaped: list[BaseException] = []

        def _body():
            try:
                with _sigterm_raises():
                    pass
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                escaped.append(exc)

        thread = threading.Thread(target=_body)
        thread.start()
        thread.join(timeout=10)

        assert not thread.is_alive(), "the context manager hung off the main thread"
        assert escaped == [], f"_sigterm_raises raised off the main thread: {escaped}"
        assert signal.getsignal(signal.SIGTERM) is before


class _StubbornProcess:
    """A process double that outlives its first join, like a wedged worker.

    C-7: the escalation branches -- ``_shutdown_logging``'s ``terminate()`` and
    ``_terminate_workers``' ``kill()`` -- only run when ``is_alive()`` is still
    True after a bounded join, which never happens in a test that uses a real
    (and therefore promptly exiting) process.

    Args:
        alive_for: how many ``is_alive()`` calls report True before it reports
            False. ``None`` means "never dies", which is what drives
            ``_terminate_workers`` all the way to ``kill()``.
    """

    def __init__(self, alive_for: int | None = None) -> None:
        self._alive_for = alive_for
        self.calls: list[str] = []
        self._is_alive_calls = 0

    def is_alive(self) -> bool:
        self._is_alive_calls += 1
        if self._alive_for is None:
            return True
        return self._is_alive_calls <= self._alive_for

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(f"join({timeout})")

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")


class TestWedgedChildrenAreEscalated:
    """C-7: the terminate/kill branches, with a process that will not go."""

    def test_shutdown_logging_terminates_a_logger_that_outlives_its_join(self, tmp_path):
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        orch = WorkflowOrchestrator(Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, use_gpu=False))
        # Only the logger process: no queue, no handler, no Manager, so this
        # exercises exactly the join -> still alive -> terminate escalation.
        stubborn = _StubbornProcess(alive_for=1)
        orch._logger_p = stubborn
        orch.logging_queue = None
        orch._log_handler = None
        orch._logging_manager = None

        orch._shutdown_logging()

        assert stubborn.calls == ["join(10)", "terminate", "join(5)"]
        assert orch._logger_p is None

    def test_terminate_workers_kills_a_worker_that_survives_terminate(self, tmp_path):
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        orch = WorkflowOrchestrator(Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, use_gpu=False))
        never_dies = _StubbornProcess()  # is_alive() is True forever
        shut_down: list[str] = []

        class _Manager:
            def shutdown(self):
                shut_down.append("shutdown")

        orch._terminate_workers([never_dies], [_Manager()])

        assert never_dies.calls == ["terminate", "join(5)", "kill", "join(5)"]
        # The Managers go LAST, after the workers holding proxies into them.
        assert shut_down == ["shutdown"]

    def test_a_manager_that_is_already_gone_does_not_raise_out_of_the_finally(self, tmp_path):
        """``_terminate_workers`` runs from a ``finally``; a dead Manager socket
        must not become the exception the run reports."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.orchestration.workflow import WorkflowOrchestrator

        orch = WorkflowOrchestrator(Auto3DOptions(path=str(tmp_path / "x.smi"), k=1, use_gpu=False))

        class _DeadManager:
            def shutdown(self):
                raise BrokenPipeError("server already gone")

        orch._terminate_workers([], [_DeadManager()])  # must not raise
