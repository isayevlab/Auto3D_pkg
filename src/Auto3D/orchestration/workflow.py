"""Workflow orchestration for Auto3D conformer generation pipeline."""

from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from logging.handlers import QueueHandler
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING

from Auto3D.domain.id_mapping import decode_ids, encode_ids
from Auto3D.engines.model_factory import ModelFactory
from Auto3D.engines.models.preflight import preflight_model
from Auto3D.foundation.config import Auto3DOptions, optimizer_worker_indices
from Auto3D.foundation.exceptions import (
    ConfigurationError,
    FileFormatError,
    InputValidationError,
    OptimizationError,
)
from Auto3D.foundation.torch_config import TorchConfig, configure_torch
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.reconciliation import find_ids_not_in_sdf, find_smiles_not_in_sdf
from Auto3D.foundation.utils.sdf_io import reorder_sdf
from Auto3D.orchestration.chunk_manager import ChunkManager
from Auto3D.orchestration.pipeline.input_checks import check_input, check_valid_configuration
from Auto3D.orchestration.workflow_workers import (
    ProgressEvent,
    isomer_wrapper,
    logger_process,
    optim_rank_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import LogRecord
    from multiprocessing import Queue

logger = get_logger(__name__)

# Backstop timeout (seconds) for joining an optimizer worker after the isomer
# worker has been found to have been killed by a signal -- see
# _isomer_worker_was_signal_killed/_join_optimizer_bounded. Deliberately NOT
# used for every nonzero exit code: an unhandled exception in isomer_wrapper
# still runs its own `finally` (a `finally` always executes while the
# exception unwinds the stack, before multiprocessing's bootstrap ever calls
# sys.exit(1)) before the process exits with a *positive* code, so the
# sentinels are already genuinely in the queue and any optimizer still
# running is doing legitimate work on the finite backlog of chunks queued
# before the failure -- one chunk at a high opt_steps can easily run longer
# than this window on its own, and bounding that join would silently drop a
# real result. A *negative* exitcode is the one case worth bounding: the
# process was killed by a signal (SIGKILL from the OOM killer, a segfault),
# so nothing -- including this worker's own cleanup -- ran, and an optimizer
# that is *also* wedged by the same pressure must not be allowed to hang the
# run a second, unrecoverable way.
_ABNORMAL_EXIT_JOIN_TIMEOUT = 600


def _package_version() -> str:
    """Read the installed ``Auto3D`` distribution version.

    Used only for the run-log banner (see ``_log_banner``). This duplicates
    ``Auto3D.__init__._detect_version`` rather than reading
    ``Auto3D.__version__`` off the package -- the orchestrator (core layer)
    used to do ``import Auto3D`` purely to reach that one attribute, which
    was flagged as a needless self-import of this module's own package root
    (review Minor #35). ``importlib.metadata`` is the single source of truth
    either way; this just reads it directly instead of through the parent
    package's namespace.
    """
    try:
        return _dist_version("Auto3D")
    except PackageNotFoundError:
        return "unknown"


class _DropOnFullQueueHandler(QueueHandler):
    """A ``QueueHandler`` that drops a record instead of raising when full.

    ``_setup_logging`` bounds the logging queue at 999 items on purpose: it
    is a ``Manager().Queue()`` shared by every worker process, and an
    unbounded one would let a runaway burst of log records grow it without
    limit if ``logger_process`` (the sole consumer) ever falls behind. The
    stock ``QueueHandler.enqueue()`` calls ``queue.put_nowait()``, and stdlib
    ``logging.Handler.handleError()`` reacts to the resulting ``queue.Full``
    by printing a full traceback to stderr -- once per dropped record, so a
    genuine burst does not just lose N records quietly, it also prints N
    tracebacks while doing it, which is worse than the loss itself. This
    override keeps the bound (unlike simply removing it, which would trade a
    bounded memory ceiling for an unbounded one under the very same burst)
    and makes the drop itself -- not a wall of tracebacks about the drop --
    the only visible symptom, which is exactly what a full queue already
    means.
    """

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Exception:
            # queue.Full (the burst this class exists for) and any other put
            # failure alike: a lost log record must never crash, or spam
            # stderr about, the run it exists to describe.
            pass


class WorkflowOrchestrator:
    """Orchestrates the Auto3D conformer generation pipeline.

    This class encapsulates the workflow logic previously in the main() function,
    providing better separation of concerns and testability.

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
        # Memory-scaled atom batch size for optimization, set in _prepare_chunks.
        # Defaults to the unscaled config value.
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
        input_format = Path(self.config.path).suffix[1:]  # Remove leading dot
        if input_format not in ("smi", "sdf"):
            raise FileFormatError(
                f"Input file type is not supported. Only .smi and .sdf are supported. "
                f"But the input file is {input_format}."
            )

        # Store format on the config -- the single source of truth for downstream
        # consumers (chunk manager, isomer workers), surviving replace()/pickling.
        self.config["input_format"] = input_format

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

    def _setup_logging(self) -> None:
        """Initialize logging infrastructure."""
        logging_path = self.job_dir / "Auto3D.log"
        self.logging_queue = self.mp_context.Manager().Queue(999)

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
        self.logger.addHandler(_DropOnFullQueueHandler(self.logging_queue))
        self.logger.setLevel(logging.INFO)

        # Log banner
        self._log_banner()
        self._log_parameters()

    def _shutdown_logging(self) -> None:
        """Flush and stop the daemon logging process.

        Sends the poison-pill sentinel on the logging queue and joins the
        logger process so its file handler flushes before the run returns.
        Safe to call multiple times and when logging was never started.
        """
        if self.logging_queue is not None:
            try:
                # Bounded (see _setup_logging's Queue(999)) precisely so a
                # burst of worker log records cannot grow it without limit --
                # which means a plain blocking put(None) here could itself
                # block on that same bound if logger_process has fallen
                # behind or died, hanging this method's caller (run()'s
                # `finally`) forever. A short timeout turns that hang into a
                # logged warning; queue.Full (like any other put failure) is
                # swallowed the same way below, since failing to enqueue the
                # shutdown sentinel must never stop the run itself returning.
                self.logging_queue.put(None, timeout=5)
            except Exception:
                logger.warning("Failed to enqueue logger shutdown sentinel.")

        if self._logger_p is not None:
            self._logger_p.join(timeout=10)
            self._logger_p = None

    def _log_banner(self) -> None:
        """Log the Auto3D ASCII art banner."""
        if self.logger is None:
            return

        self.logger.info(
            f"         _              _             _____   ____  \n"
            f"        / \\     _   _  | |_    ___   |___ /  |  _ \\ \n"
            f"       / _ \\   | | | | | __|  / _ \\    |_ \\  | | | |\n"
            f"      / ___ \\  | |_| | | |_  | (_) |  ___) | | |_| |\n"
            f"     /_/   \\_\\  \\__,_|  \\__|  \\___/  |____/  |____/  {_package_version()}\n"
            f"              // Generating low-energy 3D structures"
        )

    def _log_parameters(self) -> None:
        """Log input parameters."""
        if self.logger is None:
            return

        self.logger.info("=" * 80)
        self.logger.info("                               INPUT PARAMETERS")
        self.logger.info("=" * 80)
        for key, val in self.config.items():
            self.logger.info(f"{key}: {val}")

        self.logger.info("=" * 80)
        self.logger.info("                               RUNNING PROCESS")
        self.logger.info("=" * 80)

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
        # prepare_chunks() no longer mutates the shared config, so we thread the
        # scaled value through to a per-run config copy in _run_pipeline.
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
                f"Input file {self.config.path} contains no molecules; "
                "there is nothing to process."
            )
        return chunk_info

    def _run_pipeline(self, chunk_info: list[tuple[str, str]]) -> None:
        """Run the isomer generation and optimization pipeline.

        Args:
            chunk_info: List of (chunk_path, chunk_dir) tuples.
        """
        chunk_queue: Queue[tuple[str, str, str, int] | str] = self.mp_context.Manager().Queue()

        # Process-safe channel for live progress events, created only when a
        # progress callback was supplied (interactive `auto3d run`). When None,
        # the optimizer workers emit nothing and the supervise loop below falls
        # back to plain blocking joins -- the default/library path is unchanged.
        progress_queue = (
            self.mp_context.Manager().Queue() if self.progress_callback is not None else None
        )

        # Per-run config carrying the memory-scaled batch size for optimization.
        # Built with dataclasses.replace so self.config (itself already a
        # private copy made at the top of run(), see M16) is left holding the
        # unscaled value -- only the optimizer workers get the scaled one.
        opt_config = self.config.replace(batchsize_atoms=self.scaled_batchsize_atoms)

        # Create isomer generation process
        p1 = self.mp_context.Process(
            target=isomer_wrapper,
            args=(chunk_info, self.config, chunk_queue, self.logging_queue),
        )

        # Create optimization processes: one per GPU when running on GPU with a
        # list of indices, a single worker otherwise. A CPU run with a list of
        # gpu_idx must NOT spawn N processes all contending for the same cores
        # (N model loads -> OOM risk); optimizer_worker_indices collapses that to
        # one, and the isomer worker derives its sentinel count the same way.
        p2s: list[BaseProcess] = []
        for idx in optimizer_worker_indices(self.config.use_gpu, self.config.gpu_idx):
            p2s.append(
                self.mp_context.Process(
                    target=optim_rank_wrapper,
                    args=(opt_config, chunk_queue, self.logging_queue, idx, progress_queue),
                )
            )

        # Start all processes
        p1.start()
        for p2 in p2s:
            p2.start()

        # Wait for completion and supervise exit codes.
        #
        # Two-layer guarantee against an isomer worker that dies without
        # running its `finally` (SIGKILL from the OOM killer, a segfault in
        # RDKit/Boost, os._exit -- none of which give Python a chance to run
        # cleanup code): layer one is workflow_workers.isomer_wrapper's own
        # `finally`, which puts one "Done" sentinel per optimizer on every
        # exit Python *does* get to run cleanup for -- including an unhandled
        # exception, since a `finally` still runs on the way to exit(1).
        # Layer two is `_ensure_done_sentinels` below: when `p1.exitcode`
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
            p1.join()
            self._check_exit(p1, "Isomer generation")
            self._ensure_done_sentinels(p1.exitcode, chunk_queue)
            degraded = self._isomer_worker_was_signal_killed(p1.exitcode)
            for p2 in p2s:
                if degraded:
                    self._join_optimizer_bounded(p2)
                else:
                    p2.join()
                self._check_exit(p2, "Optimization")

    def _ensure_done_sentinels(
        self, p1_exitcode: int | None, chunk_queue: Queue[tuple[str, str, str, int] | str]
    ) -> None:
        """Supply the "Done" sentinels isomer_wrapper's `finally` may have missed.

        See the two-layer comment in `_run_pipeline`. Tops the queue up
        whenever `p1_exitcode` is neither 0 nor None -- i.e. every case that
        is not "the isomer worker ran to completion or was never started at
        all" -- regardless of *how* it failed: unlike
        `_isomer_worker_was_signal_killed` below, this must not narrow to
        signal deaths only. A positive exit code (an unhandled exception)
        already means `isomer_wrapper`'s own `finally` ran and the sentinels
        it put are already in the queue, but topping the queue up again here
        is harmless (every optimizer has already exited by the time a
        doubled-up sentinel would be consumed) -- so there is no reason to
        skip it, and every reason to: an `os._exit()` deep in a dependency
        can still exit with a small positive code while skipping `finally`
        exactly like a signal would, and this is the one guarantee that must
        never depend on telling that case apart from a normal exception.
        """
        if p1_exitcode not in (0, None):
            n_optimizers = len(optimizer_worker_indices(self.config.use_gpu, self.config.gpu_idx))
            for _ in range(n_optimizers):
                chunk_queue.put("Done")

    @staticmethod
    def _isomer_worker_was_signal_killed(p1_exitcode: int | None) -> bool:
        """Whether `p1_exitcode` indicates the isomer worker was killed by a
        signal (SIGKILL from the OOM killer, a segfault) rather than exiting
        through ordinary Python control flow.

        On POSIX, `multiprocessing.Process.exitcode` is negative (the
        negated signal number) for a signal death and non-negative
        otherwise: 0 for success, and a positive code -- typically 1 -- for
        an unhandled exception, because multiprocessing's own bootstrap
        catches it and calls `sys.exit(1)` only *after* the exception has
        already unwound the stack and run every `finally` along the way.
        A negative exitcode is therefore the one reliable signal that
        cleanup did NOT run, which is what `_join_optimizer_bounded`'s
        bounded-join-then-terminate is for -- see
        `_ABNORMAL_EXIT_JOIN_TIMEOUT`. It intentionally does not try to catch
        an `os._exit()` with a small positive code (indistinguishable here
        from a normal exception): `_ensure_done_sentinels` already covers
        that case for the sentinel guarantee, and bounding the join for
        every positive code as well would risk terminating a healthy
        optimizer mid-chunk on the much more common unhandled-exception path.
        """
        return p1_exitcode is not None and p1_exitcode < 0

    def _join_optimizer_bounded(self, p2: BaseProcess) -> None:
        """Join one optimizer worker, backstopped by a bounded timeout.

        Only used once `_isomer_worker_was_signal_killed` is True: at that
        point the run is already in a degraded state, and this bounds how
        long a possibly-also-wedged optimizer can hang it. Not reached on
        the ordinary path, nor on a plain unhandled-exception exit -- see
        `_ABNORMAL_EXIT_JOIN_TIMEOUT`.
        """
        p2.join(timeout=_ABNORMAL_EXIT_JOIN_TIMEOUT)
        if p2.is_alive():
            logger.error(
                "Optimization process (pid=%s) did not exit within %ss of the "
                "isomer worker's abnormal exit; terminating it.",
                p2.pid,
                _ABNORMAL_EXIT_JOIN_TIMEOUT,
            )
            p2.terminate()
            p2.join()

    def _check_exit(self, proc: BaseProcess, label: str) -> None:
        """Log a warning if a worker process exited abnormally."""
        if proc.exitcode not in (0, None):
            logger.error(
                "%s process exited with code %s; output may be incomplete.",
                label,
                proc.exitcode,
            )

    def _log_both(self, msg: str, *, warning: bool = False) -> None:
        """Emit ``msg`` to both the module logger and this run's Auto3D.log.

        ``self.logger`` (the per-run file logger ``_setup_logging`` attaches)
        is ``None`` until logging starts, so every call site used to guard the
        per-run copy with its own ``if self.logger:`` right next to an
        identical module-level call -- four times (``_finalize_output``,
        ``_reconcile_output``, and twice in ``_log_timing``), three at
        ``info`` and one at ``warning``.
        """
        (logger.warning if warning else logger.info)(msg)
        if self.logger:
            (self.logger.warning if warning else self.logger.info)(msg)

    def _supervise_with_progress(
        self,
        p1: BaseProcess,
        p2s: list[BaseProcess],
        progress_queue: Queue[ProgressEvent],
        chunk_queue: Queue[tuple[str, str, str, int] | str],
    ) -> None:
        """Drain progress events to the callback while workers run, then join.

        Used only when a progress callback is set. The progress queue is an
        unbounded Manager queue, so workers never block on put even if draining
        lags; after the workers exit we drain any buffered events, then join
        (bounded only in the same degraded case _run_pipeline handles) and run
        the same exit-code checks as the default path.

        The `any(p.is_alive() for p in procs)` loop below is, on its own, the
        same hang `_run_pipeline`'s two-layer comment describes: if the isomer
        worker (`p1`) dies without running its `finally`, every optimizer
        blocks on `queue.get()` forever, so `p1` is the only process that ever
        goes not-alive and this loop never exits. Layer two therefore has to
        run from *inside* the loop here, the moment `p1` is seen to have
        exited -- not after it, the way the non-progress path can afford to.
        """
        import queue as _queue

        procs = [p1, *p2s]
        sentinels_checked = False
        while any(p.is_alive() for p in procs):
            if not sentinels_checked and not p1.is_alive():
                self._ensure_done_sentinels(p1.exitcode, chunk_queue)
                sentinels_checked = True
            try:
                event = progress_queue.get(timeout=0.2)
            except _queue.Empty:
                continue
            self._emit_progress(event)
        if not sentinels_checked:
            # Every process (including p1) was already done by the time the
            # while-condition above was first evaluated.
            self._ensure_done_sentinels(p1.exitcode, chunk_queue)
        # Drain events buffered after the liveness check.
        while True:
            try:
                event = progress_queue.get_nowait()
            except _queue.Empty:
                break
            self._emit_progress(event)

        p1.join()
        self._check_exit(p1, "Isomer generation")
        degraded = self._isomer_worker_was_signal_killed(p1.exitcode)
        for p2 in p2s:
            if degraded:
                self._join_optimizer_bounded(p2)
            else:
                p2.join()
            self._check_exit(p2, "Optimization")

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Forward one progress event to the callback; never raise."""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            logger.debug("Progress callback failed for event %r", event, exc_info=True)

    def _finalize_output(self, start_time: float) -> str:
        """Combine outputs and finalize the pipeline.

        Args:
            start_time: Pipeline start time for duration calculation.

        Returns:
            Path to the final output SDF file.
        """
        # Combine all job outputs using pathlib glob
        output_files = list(self.job_dir.glob("job*/*_3d.sdf"))
        # Computed once, up front: both failure messages below name it, and
        # at most one of the two `raise`s below it can ever execute.
        log_path = self.job_dir / "Auto3D.log"

        if not output_files:
            raise OptimizationError(
                "No chunk produced a 3D structure output file, so no 3D "
                "structure converged. The model was already verified "
                "obtainable before any chunk was processed (pre-flight "
                "passed), ruling out a cold cache, network failure, or "
                "corrupted download as the cause. Likely causes: "
                "insufficient memory for the batch size used, input SMILES "
                "that do not encode valid chemical structures, or "
                "optimization settings (opt_steps/patience) too aggressive "
                f"for these molecules. See {log_path} for the per-chunk "
                "errors already recorded during this run."
            )

        # Combine output data, streaming each chunk file into the combined
        # path one at a time rather than reading every chunk into one shared
        # list and then joining that whole list into a second string before
        # writing -- the previous approach held three live copies of the
        # combined output at once (the list of lines, the joined string, and
        # the OS write buffer), measured at 2.70x the file's own size (53 MB
        # peak for a 19.6 MB SDF); a 2 GB output would have meant roughly
        # 5.4 GB resident in the orchestrator. Iterating line-by-line while
        # writing keeps the $$$$-terminator check exactly as it was (`any`
        # line, across every chunk, stripped, equal to "$$$$") without ever
        # holding more than the current line in memory, and sidesteps the
        # line-splitting-across-a-read-boundary edge case a raw
        # shutil.copyfileobj byte-chunk copy would have to guard against
        # separately to answer the same question.
        path_combined = self.job_dir / f"{self.input_path.stem}_out.sdf"
        found_terminator = False
        with path_combined.open("w") as combined_fh:
            for file_path in output_files:
                with file_path.open("r") as chunk_fh:
                    for line in chunk_fh:
                        if line.strip() == "$$$$":
                            found_terminator = True
                        combined_fh.write(line)

        if not found_terminator:
            # Nothing converged: leave no partial/empty output file behind,
            # matching the "no chunk produced output" branch above, which
            # never creates path_combined at all.
            path_combined.unlink()
            raise OptimizationError(
                "No 3D structure converged. Every chunk produced an output "
                "file, but none of them contain a converged structure. The "
                "model was already verified obtainable before any chunk was "
                "processed (pre-flight passed), ruling out a cold cache, "
                "network failure, or corrupted download as the cause. "
                "Likely causes: every input molecule failed geometry "
                "optimization (opt_steps/patience too aggressive for these "
                "molecules), or the energy window/top-k filtering removed "
                f"all conformers. See {log_path} for the per-chunk errors "
                "already recorded during this run."
            )

        # Log timing
        self._log_timing(start_time)

        # Reorder and decode IDs
        reorder_sdf(str(path_combined), str(self.input_path))
        path_output = decode_ids(str(path_combined), self.id_mapping)

        # Cleanup temporary files (input_path is unlinked in run()'s finally)
        path_combined.unlink()

        self._log_both(f"Output path: {path_output}")

        # Reconcile inputs against outputs (C7): a molecule that vanished
        # mid-pipeline must leave a trace. Compare the ORIGINAL input
        # (self.config.path -- untouched, user-facing IDs) against the
        # DECODED output (path_output), not self.input_path (the encoded temp
        # file, whose IDs are encode_ids' numeric indices) or path_combined
        # (still numeric) -- either of those would make every molecule look
        # missing.
        self._reconcile_output(path_output)

        # Clear model cache to free GPU memory
        ModelFactory.clear_cache()

        return path_output

    def _reconcile_output(self, path_output: str) -> None:
        """Compare the original input against the final output SDF (C7).

        Populates ``self.failures`` with every input molecule ID absent from
        ``path_output``, and logs them (both to the module logger and, when
        available, this run's own Auto3D.log). ``main()`` reads
        ``self.failures`` off the orchestrator to build the returned
        ``WorkflowResult.failures``.

        Args:
            path_output: Path to the final, decoded output SDF.
        """
        if self.config.input_format == "smi":
            missing = find_smiles_not_in_sdf(self.config.path, path_output)
            self.failures = [mol_id for mol_id, _smiles in missing]
        elif self.config.input_format == "sdf":
            # find_smiles_not_in_sdf reads its expected-IDs list from a .smi
            # file, which does not exist for SDF input; find_ids_not_in_sdf is
            # the SDF-native equivalent (reads _Name directly from the source
            # SDF), so SDF input gets the same reconciliation coverage instead
            # of silently skipping it.
            self.failures = find_ids_not_in_sdf(self.config.path, path_output)
        else:
            # _validate_input already rejects any other extension before this
            # point is ever reached.
            self.failures = []

        if self.failures:
            msg = (
                f"{len(self.failures)} input molecule(s) produced no output "
                f"and were not reported anywhere else: {sorted(self.failures)}"
            )
            self._log_both(msg, warning=True)

    def _log_timing(self, start_time: float) -> None:
        """Log pipeline execution time.

        Args:
            start_time: Pipeline start time.
        """
        self._log_both("Energy unit: Hartree if implicit.")

        elapsed_minutes = int((time.time() - start_time) / 60)

        if elapsed_minutes <= 60:
            msg = f"Program running time: {elapsed_minutes + 1} minute(s)"
        else:
            hours = elapsed_minutes // 60
            remaining = elapsed_minutes - hours * 60
            msg = f"Program running time: {hours} hour(s) and {remaining} minute(s)"

        self._log_both(msg)
