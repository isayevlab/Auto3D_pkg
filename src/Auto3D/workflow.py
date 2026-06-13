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
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import check_input, check_valid_configuration, reorder_sdf
from Auto3D.utils.file_ops import decode_ids, encode_ids
from Auto3D.utils.logging_config import get_logger
from Auto3D.workflow_workers import (
    isomer_wrapper,
    logger_process,
    optim_rank_wrapper,
)

if TYPE_CHECKING:
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

    def __init__(self, config: Auto3DOptions) -> None:
        """Initialize the orchestrator with configuration.

        Args:
            config: Auto3D configuration options.
        """
        self.config = config
        self.job_name: str = ""
        self.job_dir: Path = Path()
        self.input_path: Path = Path()
        self.id_mapping: dict[str, str] = {}
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
        start_time = time.time()

        # Configure PyTorch settings (TF32, cuDNN benchmark)
        torch_config = TorchConfig(allow_tf32=self.config.allow_tf32)
        configure_torch(torch_config)

        try:
            # Phase 1: Validation and setup. Kept inside the try so the encoded
            # temp file written by _validate_input is cleaned up even when a
            # later setup step (job dir creation, logging) raises.
            self._validate_input()
            self._setup_job_directory()
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
            # input_path stays at its Path() default (the cwd, a directory)
            # until _validate_input assigns the encoded file, so is_file()
            # guards against ever unlinking anything but a real encoded input.
            self._shutdown_logging()
            if self.input_path.is_file():
                self.input_path.unlink()

    def _validate_input(self) -> None:
        """Validate input configuration and prepare encoded input file.

        Raises:
            ConfigurationError: If path is None or k/window not specified.
            FileFormatError: If input file format is not supported.
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

        # Encode IDs for internal processing
        encoded_path, self.id_mapping = encode_ids(self.config.path)
        self.input_path = Path(encoded_path)

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

    def _setup_job_directory(self) -> None:
        """Create the job directory for output files."""
        # Remove '_encoded' suffix from stem
        job_basename = self.input_path.stem[:-8]  # Remove '_encoded'
        self.job_name = f"{job_basename}_{self.config.job_name}"
        self.job_dir = self.input_path.resolve().parent / self.job_name

        self.job_dir.mkdir()

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

        # Per-run config carrying the memory-scaled batch size for optimization.
        # Built with dataclasses.replace so the caller's shared config is never
        # mutated (review findings #35/#36).
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
                    args=(opt_config, chunk_queue, self.logging_queue, idx),
                )
            )

        # Start all processes
        p1.start()
        for p2 in p2s:
            p2.start()

        # Wait for completion and supervise exit codes. The isomer worker emits
        # a "Done" sentinel per optimizer in a `finally`, so even if it crashed
        # the optimizers will drain and exit rather than block on queue.get().
        p1.join()
        if p1.exitcode not in (0, None):
            logger.error(
                "Isomer generation process exited with code %s; "
                "output may be incomplete.",
                p1.exitcode,
            )

        for p2 in p2s:
            p2.join()
            if p2.exitcode not in (0, None):
                logger.error(
                    "Optimization process exited with code %s; "
                    "output may be incomplete.",
                    p2.exitcode,
                )

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
            raise OptimizationError(
                "The optimization engine did not run, or no 3D structure converged.\n"
                "The reason might be one of the following:\n"
                "1. Allocated memory is not enough;\n"
                "2. The input SMILES encodes invalid chemical structures;\n"
                "3. Patience is too small"
            )

        # Combine output data
        combined_data: list[str] = []
        for file_path in output_files:
            combined_data.extend(file_path.read_text().splitlines(keepends=True))

        if not any(line.strip() == "$$$$" for line in combined_data):
            raise OptimizationError(
                "No 3D structure converged. None of the input molecules produced "
                "an optimized conformer. Check input validity, memory, and patience settings."
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

        # Clear model cache to free GPU memory
        ModelFactory.clear_cache()

        return path_output

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
