"""Workflow orchestration for Auto3D conformer generation pipeline."""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from datetime import datetime
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import Auto3D
from Auto3D.chunk_manager import ChunkManager
from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import ConfigurationError, FileFormatError, OptimizationError
from Auto3D.model_factory import ModelFactory
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import check_input, reorder_sdf
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
        self.input_format: str = ""
        self.id_mapping: dict[str, str] = {}
        self.logging_queue: Queue[LogRecord | None] | None = None
        self.logger: logging.Logger | None = None
        self._logger_p: mp.Process | None = None

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

        # Phase 1: Validation and setup
        self._validate_input()
        self._setup_job_directory()
        self._setup_logging()

        try:
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
            self._shutdown_logging()
            if self.input_path.exists():
                self.input_path.unlink()

    def _validate_input(self) -> None:
        """Validate input configuration and prepare encoded input file.

        Raises:
            ConfigurationError: If path is None or k/window not specified.
            FileFormatError: If input file format is not supported.
        """
        if self.config.path is None:
            raise ConfigurationError("Please specify the input file path.")

        # Encode IDs for internal processing
        encoded_path, self.id_mapping = encode_ids(self.config.path)
        self.input_path = Path(encoded_path)

        # Validate file format
        self.input_format = self.input_path.suffix[1:]  # Remove leading dot
        if self.input_format not in ("smi", "sdf"):
            raise FileFormatError(
                f"Input file type is not supported. Only .smi and .sdf are supported. "
                f"But the input file is {self.input_format}."
            )

        # Store format in config for downstream use
        self.config["input_format"] = self.input_format

        # Validate output selection
        if not self.config.k and not self.config.window:
            raise ConfigurationError(
                "Either k or window needs to be specified. "
                "Usually, setting '--k=1' satisfies most needs."
            )

        # Generate job name if not provided
        if self.config.job_name == "":
            self.config.job_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

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
            input_format=self.input_format,
            job_dir=self.job_dir,
            workflow_logger=self.logger,
        )
        return chunk_manager.prepare_chunks()

    def _run_pipeline(self, chunk_info: list[tuple[str, str]]) -> None:
        """Run the isomer generation and optimization pipeline.

        Args:
            chunk_info: List of (chunk_path, chunk_dir) tuples.
        """
        chunk_queue: Queue[tuple[str, str, str, int] | str] = mp.Manager().Queue()

        # Create isomer generation process
        p1 = mp.Process(
            target=isomer_wrapper,
            args=(chunk_info, self.config, chunk_queue, self.logging_queue),
        )

        # Create optimization processes (one per GPU or single for CPU)
        p2s: list[mp.Process] = []
        if isinstance(self.config.gpu_idx, int):
            p2s.append(
                mp.Process(
                    target=optim_rank_wrapper,
                    args=(self.config, chunk_queue, self.logging_queue, self.config.gpu_idx),
                )
            )
        else:
            for idx in self.config.gpu_idx:
                p2s.append(
                    mp.Process(
                        target=optim_rank_wrapper,
                        args=(self.config, chunk_queue, self.logging_queue, idx),
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
