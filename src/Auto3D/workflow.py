"""Workflow orchestration for Auto3D conformer generation pipeline."""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import replace
from datetime import datetime
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import Auto3D
from Auto3D.chunk_manager import ChunkManager
from Auto3D.config import Auto3DOptions, optimizer_worker_indices
from Auto3D.exceptions import ConfigurationError, FileFormatError, OptimizationError
from Auto3D.model_factory import ModelFactory
from Auto3D.models.preflight import preflight_model
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import check_input, check_valid_configuration, reorder_sdf
from Auto3D.utils.file_ops import (
    decode_ids,
    encode_ids,
    find_ids_not_in_sdf,
    find_smiles_not_in_sdf,
)
from Auto3D.utils.logging_config import get_logger
from Auto3D.workflow_workers import (
    isomer_wrapper,
    logger_process,
    optim_rank_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import LogRecord
    from multiprocessing import Queue

logger = get_logger(__name__)


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
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        """Initialize the orchestrator with configuration.

        Args:
            config: Auto3D configuration options.
            progress_callback: Optional callable invoked (in the main process)
                with per-step optimizer progress events for a live display. When
                None (the default, and every library/test caller) the pipeline
                runs exactly as before -- no progress queue, plain blocking joins.
        """
        self.config = config
        self.progress_callback = progress_callback
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
        self._logger_p: mp.Process | None = None
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
        self.config = replace(self.config)

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
            self._encode_input()
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

        Every check that can reject the run happens here, before
        ``_setup_job_directory`` creates a directory and ``_encode_input``
        writes a file, so a rejected run leaves no trace on disk at all.

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
        config_errors = check_valid_configuration(
            path=self.config.path,
            k=self.config.k,
            window=self.config.window,
            use_gpu=self.config.use_gpu,
            gpu_idx=self.config.gpu_idx,
            optimizing_engine=self.config.optimizing_engine,
            isomer_engine=self.config.isomer_engine,
            opt_steps=self.config.opt_steps,
            enumerate_tautomer=self.config.enumerate_tautomer,
            tauto_engine=self.config.tauto_engine,
        )
        if config_errors:
            raise ConfigurationError(
                "Invalid configuration:\n  - " + "\n  - ".join(config_errors)
            )

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
        encoded_path, self.id_mapping = encode_ids(
            self.config.path, out_dir=self.job_dir
        )
        self.input_path = Path(encoded_path)

    def _setup_logging(self) -> None:
        """Initialize logging infrastructure."""
        logging_path = self.job_dir / "Auto3D.log"
        self.logging_queue = mp.Manager().Queue(999)

        # Start logging process
        logger_p = mp.Process(
            target=logger_process,
            args=(self.logging_queue, str(logging_path)),
            daemon=True,
        )
        logger_p.start()
        self._logger_p = logger_p

        # Configure main process logger
        self.logger = logging.getLogger("auto3d")
        self.logger.addHandler(QueueHandler(self.logging_queue))
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
                self.logging_queue.put(None)
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
            f"     /_/   \\_\\  \\__,_|  \\__|  \\___/  |____/  |____/  {Auto3D.__version__}\n"
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
        return chunk_info

    def _run_pipeline(self, chunk_info: list[tuple[str, str]]) -> None:
        """Run the isomer generation and optimization pipeline.

        Args:
            chunk_info: List of (chunk_path, chunk_dir) tuples.
        """
        chunk_queue: Queue[tuple[str, str, str, int] | str] = mp.Manager().Queue()

        # Process-safe channel for live progress events, created only when a
        # progress callback was supplied (interactive `auto3d run`). When None,
        # the optimizer workers emit nothing and the supervise loop below falls
        # back to plain blocking joins -- the default/library path is unchanged.
        progress_queue = (
            mp.Manager().Queue() if self.progress_callback is not None else None
        )

        # Per-run config carrying the memory-scaled batch size for optimization.
        # Built with dataclasses.replace so self.config (itself already a
        # private copy made at the top of run(), see M16) is left holding the
        # unscaled value -- only the optimizer workers get the scaled one.
        opt_config = replace(
            self.config, batchsize_atoms=self.scaled_batchsize_atoms
        )

        # Create isomer generation process
        p1 = mp.Process(
            target=isomer_wrapper,
            args=(chunk_info, self.config, chunk_queue, self.logging_queue),
        )

        # Create optimization processes: one per GPU when running on GPU with a
        # list of indices, a single worker otherwise. A CPU run with a list of
        # gpu_idx must NOT spawn N processes all contending for the same cores
        # (N model loads -> OOM risk); optimizer_worker_indices collapses that to
        # one, and the isomer worker derives its sentinel count the same way.
        p2s: list[mp.Process] = []
        for idx in optimizer_worker_indices(
            self.config.use_gpu, self.config.gpu_idx
        ):
            p2s.append(
                mp.Process(
                    target=optim_rank_wrapper,
                    args=(opt_config, chunk_queue, self.logging_queue, idx,
                          progress_queue),
                )
            )

        # Start all processes
        p1.start()
        for p2 in p2s:
            p2.start()

        # Wait for completion and supervise exit codes. The isomer worker emits
        # a "Done" sentinel per optimizer in a `finally`, so even if it crashed
        # the optimizers will drain and exit rather than block on queue.get().
        if progress_queue is not None:
            self._supervise_with_progress(p1, p2s, progress_queue)
        else:
            p1.join()
            self._check_exit(p1, "Isomer generation")
            for p2 in p2s:
                p2.join()
                self._check_exit(p2, "Optimization")

    def _check_exit(self, proc: mp.Process, label: str) -> None:
        """Log a warning if a worker process exited abnormally."""
        if proc.exitcode not in (0, None):
            logger.error(
                "%s process exited with code %s; output may be incomplete.",
                label, proc.exitcode,
            )

    def _supervise_with_progress(
        self, p1: mp.Process, p2s: list[mp.Process], progress_queue: Queue[dict]
    ) -> None:
        """Drain progress events to the callback while workers run, then join.

        Used only when a progress callback is set. The progress queue is an
        unbounded Manager queue, so workers never block on put even if draining
        lags; after the workers exit we drain any buffered events, then join
        (immediate) and run the same exit-code checks as the default path.
        """
        import queue as _queue

        procs = [p1, *p2s]
        while any(p.is_alive() for p in procs):
            try:
                event = progress_queue.get(timeout=0.2)
            except _queue.Empty:
                continue
            self._emit_progress(event)
        # Drain events buffered after the liveness check.
        while True:
            try:
                event = progress_queue.get_nowait()
            except _queue.Empty:
                break
            self._emit_progress(event)

        p1.join()
        self._check_exit(p1, "Isomer generation")
        for p2 in p2s:
            p2.join()
            self._check_exit(p2, "Optimization")

    def _emit_progress(self, event: dict) -> None:
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

        if not output_files:
            log_path = self.job_dir / "Auto3D.log"
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

        # Combine output data
        combined_data: list[str] = []
        for file_path in output_files:
            combined_data.extend(file_path.read_text().splitlines(keepends=True))

        if not any(line.strip() == "$$$$" for line in combined_data):
            log_path = self.job_dir / "Auto3D.log"
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

        # Write combined output
        path_combined = self.job_dir / f"{self.input_path.stem}_out.sdf"
        path_combined.write_text("".join(combined_data))

        # Log timing
        self._log_timing(start_time)

        # Reorder and decode IDs
        reorder_sdf(str(path_combined), str(self.input_path))
        path_output = decode_ids(str(path_combined), self.id_mapping)

        # Cleanup temporary files (input_path is unlinked in run()'s finally)
        path_combined.unlink()

        logger.info(f"Output path: {path_output}")
        if self.logger:
            self.logger.info(f"Output path: {path_output}")

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
            logger.warning(msg)
            if self.logger:
                self.logger.warning(msg)

    def _log_timing(self, start_time: float) -> None:
        """Log pipeline execution time.

        Args:
            start_time: Pipeline start time.
        """
        logger.info("Energy unit: Hartree if implicit.")
        if self.logger:
            self.logger.info("Energy unit: Hartree if implicit.")

        elapsed_minutes = int((time.time() - start_time) / 60)

        if elapsed_minutes <= 60:
            msg = f"Program running time: {elapsed_minutes + 1} minute(s)"
        else:
            hours = elapsed_minutes // 60
            remaining = elapsed_minutes - hours * 60
            msg = f"Program running time: {hours} hour(s) and {remaining} minute(s)"

        logger.info(msg)
        if self.logger:
            self.logger.info(msg)
