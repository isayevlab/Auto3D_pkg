"""Workflow orchestration for Auto3D conformer generation pipeline.

``WorkflowOrchestrator`` is the sequencer: it owns ``__init__``, ``run`` and
every setup/teardown step around the three phases below, composing three
mixins that hold the rest of the implementation. ``run_logging.py`` holds
the queue-backed log handler, the version lookup and the mixin methods that
write through (``_log_both``), shut down and log the banner/parameters for
the run's own log file (``RunLoggingMixin``). ``supervisor.py`` holds the worker
join, exit-check and progress-drain methods (``SupervisorMixin``).
``finalize.py`` holds the output combination, reorder/decode and
reconciliation methods (``FinalizeMixin``). Three different reasons keep the
rest of the implementation here rather than in a mixin: ``_validate_input``,
``_encode_input`` and the worker-construction half of ``_run_pipeline``
resolve bare module-global names (``check_valid_configuration``,
``check_input``, ``preflight_model``, ``encode_ids``, ``isomer_wrapper``,
``optim_rank_wrapper``) that tests patch on this module; ``_setup_logging``
constructs ``Process(target=logger_process, ...)``, and ``logger_process``
must not be imported into ``run_logging.py`` (that would risk a cycle once a
later change makes ``workflow_workers.py`` import ``run_logging``);
``_start_manager`` is shared by both ``_setup_logging`` and ``_run_pipeline``
and stays here by ruling, not because anything patches it.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import shutil
import signal
import threading
import time
from datetime import datetime
from multiprocessing.managers import SyncManager
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING

from Auto3D.domain.id_mapping import encode_ids
from Auto3D.engines.models.preflight import preflight_model
from Auto3D.engines.torch_config import TorchConfig, configure_torch
from Auto3D.foundation.config import Auto3DOptions
from Auto3D.foundation.constants import EXIT_TERMINATED
from Auto3D.foundation.exceptions import ConfigurationError, FileFormatError, InputValidationError
from Auto3D.foundation.process_lifecycle import _exit_when_parent_dies
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.orchestration.chunk_manager import ChunkManager
from Auto3D.orchestration.finalize import FinalizeMixin
from Auto3D.orchestration.job_layout import optimizer_worker_indices
from Auto3D.orchestration.pipeline.input_checks import check_input, check_valid_configuration
from Auto3D.orchestration.run_logging import RunLoggingMixin, _DropOnFullQueueHandler
from Auto3D.orchestration.supervisor import SupervisorMixin
from Auto3D.orchestration.workflow_workers import (
    ProgressEvent,
    isomer_wrapper,
    logger_process,
    optim_rank_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from logging import LogRecord
    from multiprocessing import Queue

logger = get_logger(__name__)


def _raise_on_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into a normal unwind so every ``finally`` runs.

    :data:`~Auto3D.foundation.constants.EXIT_TERMINATED` (143 = 128 + SIGTERM) is
    the shell's own convention for "killed by SIGTERM", so a caller that reads
    the exit code sees exactly what it would have seen from the default
    disposition -- the difference is only that the workers were terminated on
    the way out (P-C2).
    """
    raise SystemExit(EXIT_TERMINATED)


@contextlib.contextmanager
def _sigterm_raises() -> Iterator[None]:
    """Make SIGTERM raise ``SystemExit(143)`` for the duration of the block.

    The default disposition for SIGTERM kills the process outright: no
    ``finally`` runs, so ``_run_pipeline``'s worker-termination block never
    executes and the optimizer workers are left to finish the whole remaining
    queue on the GPU (P-C2, P-M12). Raising instead gives the parent an
    ordinary unwind, which is all that block needs.

    Two conditions, and the block is a no-op unless BOTH hold:

    * **The main thread.** ``signal.signal`` raises ``ValueError`` anywhere
      else, and Auto3D is a library that may well be called from someone
      else's worker thread.
    * **SIGTERM is still at its default disposition.** A host that has
      installed its own SIGTERM handler -- a service supervisor draining
      requests, a test runner, a notebook kernel, an outer CLI of its own --
      means something by it, and a library call has no business replacing it
      for the duration of one ``main()``. Nothing is lost by declining:
      every worker arms ``PR_SET_PDEATHSIG`` plus a parent-sentinel watchdog
      of its own (``Auto3D.foundation.process_lifecycle``), so the workers
      still die with the parent however the parent goes (A-3).

    When the handler IS installed, the previous disposition -- ``SIG_DFL`` by
    the guard above -- is put back on the way out, so one ``main()`` call
    never changes how the next SIGTERM behaves.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous = signal.getsignal(signal.SIGTERM)
    if previous != signal.SIG_DFL:
        yield
        return

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        yield
    finally:
        # `previous` is SIG_DFL by the guard above, so this restores exactly
        # what was found; written as a restore rather than as a literal
        # SIG_DFL so it cannot drift away from the install condition. The
        # ``None`` arm (a handler installed from C, not representable in
        # Python, which ``signal.signal`` refuses) is unreachable for the same
        # reason -- and SIG_DFL would be the right answer for it anyway,
        # because leaving Auto3D's handler installed after the block is the
        # one outcome this must never produce.
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


class WorkflowOrchestrator(RunLoggingMixin, SupervisorMixin, FinalizeMixin):
    """Orchestrates the Auto3D conformer generation pipeline.

    This class encapsulates the workflow logic previously in the main() function,
    providing better separation of concerns and testability. The implementation
    is composed from ``RunLoggingMixin``, ``SupervisorMixin`` and
    ``FinalizeMixin`` (see the module docstring); this class itself keeps only
    ``__init__``, ``run`` and the setup/teardown sequencing around them.

    Example:
        >>> config = Auto3DOptions(path="input.smi", k=5)
        >>> orchestrator = WorkflowOrchestrator(config)
        >>> output_path = orchestrator.run()
    """

    def __init__(
        self,
        config: Auto3DOptions,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """Initialize the orchestrator with configuration.

        Args:
            config: Auto3D configuration options.
            progress_callback: Optional callable invoked (in the main process)
                with per-step optimizer progress events (see
                ``Auto3D.orchestration.workflow_workers.ProgressEvent`` for the payload
                schema) for a live display. When None (the default, and every
                library/test caller) the pipeline runs exactly as before -- no
                progress queue, plain blocking joins.
        """
        self.config = config
        self.progress_callback = progress_callback
        #: The context every process and manager below is started from.
        #:
        #: Explicit rather than global. The workers run PyTorch, and forking a
        #: process that has already initialized a CUDA context yields a broken
        #: context in the child -- the worker crashes and the run produces no
        #: output ("no 3D structure converged"). ``main()`` used to guarantee
        #: spawn by calling ``mp.set_start_method("spawn", force=True)``, which
        #: reconfigured the *caller's* interpreter for the rest of its life, and
        #: needed ``force=True`` because a default-context pool elsewhere may
        #: already have locked the global method to the platform default.
        #:
        #: Reading the context from here instead makes both problems go away at
        #: once: the guarantee is local, and what the global method happens to be
        #: is no longer any of this class's business.
        self.mp_context = mp.get_context("spawn")
        self.job_name: str = ""
        self.job_dir: Path = Path()
        self.input_path: Path = Path()
        self.id_mapping: dict[str, str] = {}
        # Input molecule IDs reconciled away as missing from the final output
        # (C7), populated by _finalize_output. Read by main() to build the
        # WorkflowResult.failures carrier.
        self.failures: list[str] = []
        self.logging_queue: Queue[LogRecord | None] | None = None
        self.logger: logging.Logger | None = None
        self._logger_p: BaseProcess | None = None
        self._logging_manager: SyncManager | None = None
        self._log_handler: _DropOnFullQueueHandler | None = None
        # Absolute atom batch size for optimization: the config's per-gigabyte
        # `batchsize_atoms` scaled by the memory ChunkManager measures, set in
        # _prepare_chunks and handed to the workers as an OptimizationConfig in
        # _run_pipeline. Defaults to the unscaled config value, i.e. to what one
        # gigabyte would buy.
        self.scaled_batchsize_atoms: int = config.batchsize_atoms

    def run(self) -> str:
        """Execute the full conformer generation pipeline.

        Returns:
            Path to the output SDF file.

        Raises:
            ConfigurationError: If configuration is invalid.
            FileFormatError: If input file format is not supported.
            OptimizationError: If no structures converge.
        """
        # Copy the caller's config once, up front (M16). _validate_input
        # below mutates self.config (job_name, input_format); without this
        # copy those mutations land on the exact object the caller still
        # holds, so a second main(args) call with the same config in one
        # process would see self.config.job_name already non-empty and
        # reuse the first run's job_name instead of generating its own.
        # This is what makes _run_pipeline's own `replace()` comment further
        # down ("the caller's shared config is never mutated") true for the
        # whole run, not just that one local copy.
        self.config = self.config.replace()

        start_time = time.time()

        # Configure PyTorch settings (TF32, cuDNN benchmark)
        torch_config = TorchConfig(allow_tf32=self.config.allow_tf32)
        configure_torch(torch_config)

        # Scoped to this run only (see _sigterm_raises): SIGTERM has to reach
        # _run_pipeline's `finally` as an exception, or the workers outlive the
        # parent that was told to stop.
        with _sigterm_raises():
            try:
                # Phase 1: Validation and setup. Kept inside the try so the encoded
                # temp file written by _encode_input is cleaned up even when a
                # later setup step (logging) raises. The job directory is created
                # BEFORE the encoding, because that is where the encoded file goes
                # -- see _encode_input.
                self._validate_input()
                self._setup_job_directory()
                try:
                    self._encode_input()
                except BaseException:
                    # encode_ids is the last step that can still *reject* the run:
                    # it raises InputValidationError on a duplicate ID, a blank
                    # molecule name, or a malformed .smi row. Moving it after the
                    # mkdir() (which is what lets it write the encoded copy into a
                    # provably new directory) therefore made a rejected run leave
                    # an empty `<stem>_<timestamp>/` beside the user's input --
                    # one more on every retry -- and, for .sdf input, a partial
                    # `<stem>_encoded.sdf` inside it, because Chem.SDWriter opens
                    # before the duplicate is seen. Removing the directory here
                    # restores the property _validate_input's docstring names: a
                    # rejected run leaves no trace on disk.
                    #
                    # Unconditionally safe, and only because _setup_job_directory
                    # used a bare mkdir(): the directory is provably new, so
                    # nothing in it can predate this run. `except BaseException`
                    # (matching ASE/geometry.py's staging cleanup) so a
                    # KeyboardInterrupt mid-encode cleans up too, and
                    # ignore_errors so a cleanup failure never masks the real
                    # rejection the user needs to see.
                    shutil.rmtree(self.job_dir, ignore_errors=True)
                    raise
                self._setup_logging()

                # Phase 2: Prepare chunks
                chunk_info = self._prepare_chunks()

                # Phase 3: Run pipeline
                self._run_pipeline(chunk_info)

                # Phase 4: Combine and finalize
                output_path = self._finalize_output(start_time)

                return output_path
            finally:
                # Always flush the daemon logger and remove the temporary encoded
                # input file, even when a phase raises partway through.
                #
                # What makes this unlink safe is NOT the is_file() test -- that
                # only distinguishes a file from input_path's Path() default (the
                # cwd, a directory), and an earlier version of this comment
                # wrongly claimed it "guards against ever unlinking anything but a
                # real encoded input". It cannot: a file is a file, and when the
                # encoded copy was written beside the user's input this line
                # deleted whatever happened to be named `<stem>_encoded.<ext>`
                # there, including a file the user owned. Safety comes from
                # _encode_input writing into self.job_dir, which _setup_job_directory
                # created with a bare mkdir() (no exist_ok) moments earlier: the
                # directory is provably new, so nothing inside it predates this run.
                self._shutdown_logging()
                if self.input_path.is_file():
                    self.input_path.unlink()

    def _validate_input(self) -> None:
        """Validate the input configuration. Writes nothing.

        Every configuration, format and model check happens here, before
        ``_setup_job_directory`` creates a directory and ``_encode_input``
        writes a file. One further class of rejection cannot be made here at
        all: duplicate molecule IDs, blank names and malformed ``.smi`` rows
        are only detectable while reading the records, which is
        ``encode_ids``' job, and ``encode_ids`` must run *after* the job
        directory exists because that is where it writes. ``run()`` therefore
        removes that freshly created directory when ``_encode_input`` raises,
        which is what keeps the overall property true: a rejected run leaves
        no trace on disk at all.

        Also resolves the optimizing engine name and verifies the model is
        obtainable (see ``preflight_model``), so a bad model name, a cold
        cache with no network, a corrupted cached file, or an unwritable
        cache directory all fail here -- before any worker is forked --
        instead of surfacing as an opaque failure deep inside a spawned
        worker.

        Raises:
            ConfigurationError: If path is None, k/window not specified, or
                the configuration (including the optimizing engine name) is
                otherwise invalid.
            FileFormatError: If input file format is not supported.
            ModelLoadError: If the optimizing model could not be obtained or
                loaded.
            DependencyError: If a required optional dependency is missing.
        """
        if self.config.path is None:
            raise ConfigurationError("Please specify the input file path.")

        # Validate file format BEFORE encoding so an unsupported extension
        # raises FileFormatError (not a generic ValueError from encode_ids) and
        # no encoded temp file is written for input we are about to reject. The
        # encoded file keeps the source suffix, so this format is authoritative.
        # casefold (M7): a chemically valid `mols.SDF` or `mols.SMI` must not
        # be refused the way `mols.xyz` is.
        input_format = Path(self.config.path).suffix[1:].casefold()  # Remove leading dot
        if input_format not in ("smi", "sdf"):
            raise FileFormatError(
                f"Input file type is not supported. Only .smi and .sdf are supported. "
                f"But the input file is {input_format}."
            )

        # Store format on the config -- the single source of truth for downstream
        # consumers (chunk manager, isomer workers), surviving replace()/pickling.
        self.config.input_format = input_format

        # Validate output selection
        if not self.config.k and not self.config.window:
            raise ConfigurationError(
                "Either k or window needs to be specified. "
                "Usually, setting '--k=1' satisfies most needs."
            )

        # Generate job name if not provided
        if self.config.job_name == "":
            self.config.job_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        # Fail fast on an invalid configuration (notably an out-of-range gpu_idx).
        # Without this the bad index only surfaces deep inside a spawned worker as
        # an opaque "no structure converged". check_valid_configuration already
        # validates the index against torch.cuda.device_count(); reuse it.
        config_errors = check_valid_configuration(self.config)
        if config_errors:
            raise ConfigurationError("Invalid configuration:\n  - " + "\n  - ".join(config_errors))

        check_input(self.config)

        # Resolve the engine name and verify the model is obtainable HERE, in
        # the parent, before any worker is forked (C8/M22). Every worker
        # builds its own copy of the model regardless (spawned processes
        # share no memory with the parent), so this is purely diagnostic: a
        # cold cache with no network, a corrupted cached file, or an
        # unwritable cache directory would otherwise surface only inside
        # optim_rank_wrapper's blanket per-chunk except, as an opaque "no 3D
        # structure converged".
        preflight_model(self.config.optimizing_engine)

    def _setup_job_directory(self) -> None:
        """Create the job directory for output files.

        Derived from ``self.config.path`` (the user's input) rather than from
        ``self.input_path`` (the encoded copy, which no longer exists at this
        point and now lives *inside* this directory anyway).

        ``mkdir()`` is deliberately bare -- no ``exist_ok``, no ``parents`` --
        so a name collision fails the run instead of merging this run's files
        into an existing directory. Every later phase depends on that: it is
        what lets ``_encode_input`` write into this directory without an
        existence check, and what lets ``run()``'s ``finally`` unlink from it.
        """
        input_file = Path(self.config.path).resolve()
        self.job_name = f"{input_file.stem}_{self.config.job_name}"
        self.job_dir = input_file.parent / self.job_name

        self.job_dir.mkdir()

    def _encode_input(self) -> None:
        """Write the encoded copy of the input into this run's job directory.

        The pipeline replaces every molecule ID with a dense integer index and
        works on that copy (``decode_ids`` restores the originals at the end).
        That copy used to be written beside the user's input as
        ``<stem>_encoded.<ext>`` -- a name the user may already be using --
        with no existence check, and ``run()``'s ``finally`` then unlinked it.
        A user with ``mols_encoded.smi`` next to ``mols.smi`` lost it: silently
        overwritten, then deleted.

        Writing into ``self.job_dir`` fixes both halves at once, and does it
        without refusing any run the user could previously make: the directory
        was created by ``_setup_job_directory``'s bare ``mkdir()`` immediately
        before this call, so it is provably new and nothing in it can belong
        to the user.
        """
        encoded_path, self.id_mapping = encode_ids(self.config.path, out_dir=self.job_dir)
        self.input_path = Path(encoded_path)

    def _start_manager(self) -> SyncManager:
        """Start a ``SyncManager`` server process that dies with this process.

        ``self.mp_context.Manager()`` is exactly this minus the initializer:
        it constructs the same ``SyncManager(ctx=...)`` and calls ``start()``
        with no arguments. The initializer is the whole point of doing it by
        hand -- ``BaseManager.start`` runs it inside the server process before
        the server accepts anything, which is the only hook there is for
        arming ``_exit_when_parent_dies`` there.

        A Manager server otherwise has no death detection of any kind (unlike
        the workers, which call ``_exit_when_parent_dies`` themselves), so a
        SIGKILLed parent left its Manager processes serving a queue nobody
        would ever read again -- the last orphans of a killed run (P-C2).
        ``_exit_when_parent_dies`` returns immediately when there is no parent
        process, so it is inert if this is ever called in-process.

        Imported from ``Auto3D.foundation.process_lifecycle``, not from
        ``workflow_workers`` (I1): under ``spawn`` the server unpickles the
        initializer by importing its defining module, so a reference into
        ``workflow_workers`` dragged torch and rdkit into every Manager
        server process just to be able to call this one stdlib-only
        function. ``process_lifecycle`` has no Auto3D imports at all, so the
        server now imports only that.
        """
        manager = SyncManager(ctx=self.mp_context)
        manager.start(initializer=_exit_when_parent_dies)
        return manager

    def _setup_logging(self) -> None:
        """Initialize logging infrastructure."""
        logging_path = self.job_dir / "Auto3D.log"
        self._logging_manager = self._start_manager()
        self.logging_queue = self._logging_manager.Queue(999)

        # Start logging process
        logger_p = self.mp_context.Process(
            target=logger_process,
            args=(self.logging_queue, str(logging_path)),
            daemon=True,
        )
        logger_p.start()
        self._logger_p = logger_p

        # Configure main process logger
        self.logger = logging.getLogger("auto3d")
        self._log_handler = _DropOnFullQueueHandler(self.logging_queue)
        self.logger.addHandler(self._log_handler)
        self.logger.setLevel(logging.INFO)

        # Log banner
        self._log_banner()
        self._log_parameters()

    def _prepare_chunks(self) -> list[tuple[str, str]]:
        """Prepare input chunks for parallel processing.

        Returns:
            List of (chunk_path, chunk_dir) tuples.
        """
        chunk_manager = ChunkManager(
            config=self.config,
            input_path=self.input_path,
            input_format=self.config.input_format,
            job_dir=self.job_dir,
            workflow_logger=self.logger,
        )
        chunk_info = chunk_manager.prepare_chunks()
        # Capture the memory-scaled batch size for the optimization workers.
        # prepare_chunks() does not mutate the shared config, so we thread the
        # scaled value through to the OptimizationConfig built in _run_pipeline.
        self.scaled_batchsize_atoms = chunk_manager.scaled_batchsize_atoms

        if not chunk_info:
            # A 0-record input (an empty .smi, or a .smi/.sdf that parses but
            # yields no rows) makes every chunk empty; prepare_chunks() skips
            # each one (chunk_manager.py's _create_chunk_files) and returns []
            # silently. Left unchecked, that [] reaches _run_pipeline, no
            # worker ever writes a "*_3d.sdf", and the failure only surfaces
            # in _finalize_output as "no chunk produced a 3D structure output
            # file" -- a message that says "pre-flight passed" and points at
            # memory/opt_steps/SMILES validity, none of which is the actual
            # cause. Raising here, with the real cause named, turns that into
            # exit 2 (with the `auto3d validate` hint) instead of exit 1.
            raise InputValidationError(
                f"Input file {self.config.path} contains no molecules; there is nothing to process."
            )
        return chunk_info

    def _run_pipeline(self, chunk_info: list[tuple[str, str]]) -> None:
        """Run the isomer generation and optimization pipeline.

        Args:
            chunk_info: List of (chunk_path, chunk_dir) tuples.
        """
        # Declared before the `try` -- and mutated only inside it -- so that a
        # failure partway through setup (a Manager server that fails to start,
        # `to_optimization_config()` refusing the scaled batch size,
        # `Process(...)` construction itself raising) still reaches the
        # `finally` with whatever was actually created so far, instead of
        # leaking it (M3). `_terminate_workers` tolerates both being empty.
        managers: list[SyncManager] = []
        # Every process that actually started, so the `finally` below only ever
        # touches processes it can legally join: Process.join() asserts on one
        # that was never started, and a start() that fails partway (EAGAIN under
        # process pressure) must surface its own error rather than an
        # AssertionError raised from the cleanup path on top of it.
        started: list[BaseProcess] = []
        try:
            # Both Managers below are owned for the duration of this method and
            # shut down by _terminate_workers in the `finally`: their server
            # processes are children of this process too, and an abandoned one
            # keeps a queue (and everything it references) alive with nothing
            # left to consume it.
            chunk_manager = self._start_manager()
            chunk_queue: Queue[tuple[str, str, str, int] | str] = chunk_manager.Queue()
            managers.append(chunk_manager)

            # Process-safe channel for live progress events, created only when a
            # progress callback was supplied (interactive `auto3d run`). When
            # None, the optimizer workers emit nothing and the supervise loop
            # below falls back to plain blocking joins -- the default/library
            # path is unchanged.
            progress_queue = None
            if self.progress_callback is not None:
                progress_manager = self._start_manager()
                progress_queue = progress_manager.Queue()
                managers.append(progress_manager)

            # The optimizer's configuration, carrying the ABSOLUTE batch size
            # for this run: built once, here, in the only process that measured
            # the memory it was scaled by. The workers used to be handed an
            # Auto3DOptions copy with the scaled number written back into the
            # per-gigabyte `batchsize_atoms` field, and each then rebuilt this
            # object per chunk from that copy -- a field whose documented
            # meaning no longer held inside a worker, and a measurement no
            # worker could repeat (P-M11, R4). self.config (itself already a
            # private copy made at the top of run(), see M16) keeps the
            # per-gigabyte value, and so does every copy of it that crosses the
            # spawn boundary below.
            opt_config = self.config.to_optimization_config(
                batchsize_atoms=self.scaled_batchsize_atoms
            )

            # Create isomer generation process
            p1 = self.mp_context.Process(
                target=isomer_wrapper,
                args=(chunk_info, self.config, chunk_queue, self.logging_queue),
            )

            # Create optimization processes: one per GPU when running on GPU
            # with a list of indices, a single worker otherwise. A CPU run
            # with a list of gpu_idx must NOT spawn N processes all contending
            # for the same cores (N model loads -> OOM risk);
            # optimizer_worker_indices collapses that to one, and the isomer
            # worker derives its sentinel count the same way.
            p2s: list[BaseProcess] = []
            for idx in optimizer_worker_indices(self.config.use_gpu, self.config.gpu_idx):
                p2s.append(
                    self.mp_context.Process(
                        target=optim_rank_wrapper,
                        args=(
                            self.config,
                            opt_config,
                            chunk_queue,
                            self.logging_queue,
                            idx,
                            progress_queue,
                        ),
                    )
                )

            # Start all processes
            p1.start()
            started.append(p1)
            for p2 in p2s:
                p2.start()
                started.append(p2)

            # Wait for completion and supervise exit codes.
            #
            # Two-layer guarantee against an isomer worker that dies without
            # running its `finally` (SIGKILL from the OOM killer, a segfault in
            # RDKit/Boost, os._exit -- none of which give Python a chance to run
            # cleanup code): layer one is workflow_workers.isomer_wrapper's own
            # `finally`, which puts one "Done" sentinel per optimizer on every
            # exit Python *does* get to run cleanup for -- including an unhandled
            # exception, since a `finally` still runs on the way to exit(1).
            # Layer two is `_ensure_done_sentinels` (supervisor.py, reached
            # through `_join_workers` just below): when `p1.exitcode`
            # comes back neither 0 nor None, layer one cannot be assumed to have
            # run, so the parent -- which holds the very same queue proxy --
            # tops the queue up itself, using the identical
            # `optimizer_worker_indices` count both the isomer worker and the
            # spawn loop above use (a doubled-up sentinel from a worker that
            # *did* clean up normally is harmless: every optimizer has already
            # exited by the time it would be consumed). `_join_optimizer_bounded`
            # backstops that second layer with a bounded join + terminate() --
            # but ONLY when `_isomer_worker_was_signal_killed`, not for every
            # nonzero exit code; see that helper and `_ABNORMAL_EXIT_JOIN_TIMEOUT`
            # for why a plain unhandled exception (a positive exit code, `finally`
            # already ran) must keep blocking indefinitely rather than risk
            # cutting off an optimizer still doing legitimate, possibly
            # long-running work on the backlog queued before the failure.
            if progress_queue is not None:
                self._supervise_with_progress(p1, p2s, progress_queue, chunk_queue)
            else:
                self._join_workers(p1, p2s, chunk_queue, top_up_sentinels=True)
        finally:
            self._terminate_workers(started, managers)

    def _terminate_workers(self, processes: list[BaseProcess], managers: list[SyncManager]) -> None:
        """Stop every child this run started. Safe after a normal join.

        Runs on every exit from ``_run_pipeline``, including SIGTERM (via the
        handler ``run()`` installs), KeyboardInterrupt, and any exception
        between stages. Without it a dead parent left the optimizer workers
        running through the whole remaining queue on the GPU (P-C2, P-M12).

        A no-op on the ordinary path: every process has already been joined by
        the time this runs, so ``is_alive()`` is False for all of them and the
        joins return immediately. The Managers are shut down last, *after* the
        workers are gone -- the reverse order tears the queue's server down
        underneath a worker still holding a proxy, which is how a Ctrl-C used
        to end in a ``BrokenPipeError`` traceback from each worker.

        ``self._logging_manager`` is deliberately absent: it outlives this
        method (``_shutdown_logging`` flushes the run log in ``run()``'s own
        ``finally``, after this) and is owned there.
        """
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        for p in processes:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for m in managers:
            try:
                m.shutdown()
            except Exception:
                # Already gone, never fully started, or its socket is dead:
                # none of that is worth raising out of a `finally` over.
                pass
