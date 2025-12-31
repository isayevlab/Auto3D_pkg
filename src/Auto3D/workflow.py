"""Workflow orchestration for Auto3D conformer generation pipeline."""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from datetime import datetime
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import psutil
import torch

import Auto3D
from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import ConfigurationError, FileFormatError, OptimizationError
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import check_input, reorder_sdf
from Auto3D.utils.file_ops import SDF2chunks, decode_ids, encode_ids

if TYPE_CHECKING:
    from logging import LogRecord
    from multiprocessing import Queue


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

        # Phase 2: Prepare chunks
        chunk_info = self._prepare_chunks()

        # Phase 3: Run pipeline
        self._run_pipeline(chunk_info)

        # Phase 4: Combine and finalize
        output_path = self._finalize_output(start_time)

        return output_path

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
        from Auto3D.auto3D import logger_process

        logging_path = self.job_dir / "Auto3D.log"
        self.logging_queue = mp.Manager().Queue(999)

        # Start logging process
        logger_p = mp.Process(
            target=logger_process,
            args=(self.logging_queue, str(logging_path)),
            daemon=True,
        )
        logger_p.start()

        # Configure main process logger
        self.logger = logging.getLogger("auto3d")
        self.logger.addHandler(QueueHandler(self.logging_queue))
        self.logger.setLevel(logging.INFO)

        # Log banner
        self._log_banner()
        self._log_parameters()

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

    def _calculate_memory_and_chunks(self) -> tuple[int, int, int]:
        """Calculate available memory and chunk configuration.

        Returns:
            Tuple of (memory_gb, chunk_size, num_jobs).
        """
        num_jobs = 1

        if self.config.memory is not None:
            memory_gb = int(self.config.memory)
        elif self.config.use_gpu:
            if isinstance(self.config.gpu_idx, int):
                gpu_idx = self.config.gpu_idx
            else:
                gpu_idx = self.config.gpu_idx[0]
                num_jobs = len(self.config.gpu_idx)
            memory_gb = int(
                math.ceil(
                    torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
                )
            )
        else:
            memory_gb = int(psutil.virtual_memory().total / (1024**3))

        chunk_size = memory_gb * self.config.capacity
        return memory_gb, chunk_size, num_jobs

    def _prepare_chunks(self) -> list[tuple[str, str]]:
        """Prepare input chunks for parallel processing.

        Returns:
            List of (chunk_path, chunk_dir) tuples.
        """
        memory_gb, chunk_size, num_jobs = self._calculate_memory_and_chunks()

        # Update batchsize based on memory
        self.config.batchsize_atoms = self.config.batchsize_atoms * memory_gb

        # Read input data
        if self.input_format == "smi":
            df = pd.read_csv(str(self.input_path), sep=r"\s+", header=None)
        else:  # sdf
            df = SDF2chunks(str(self.input_path))

        data_size = len(df)
        num_chunks = max(int(data_size // chunk_size + 1), num_jobs)

        print(f"The available memory is {memory_gb} GB.", flush=True)
        print(f"The task will be divided into {num_chunks} jobs.", flush=True)

        if self.logger:
            self.logger.info(f"The available memory is {memory_gb} GB.")
            self.logger.info(f"The task will be divided into {num_chunks} jobs.")

        # Calculate chunk indices (round-robin distribution)
        chunk_idxes: list[list[int]] = [[] for _ in range(num_chunks)]
        for i in range(num_chunks):
            idx = i
            while idx < data_size:
                chunk_idxes[i].append(idx)
                idx += num_chunks

        # Create chunk files
        return self._create_chunk_files(df, chunk_idxes, num_chunks)

    def _create_chunk_files(
        self,
        df: pd.DataFrame | list,
        chunk_idxes: list[list[int]],
        num_chunks: int,
    ) -> list[tuple[str, str]]:
        """Create individual chunk files for parallel processing.

        Args:
            df: Input data (DataFrame for SMI, list for SDF).
            chunk_idxes: Indices for each chunk.
            num_chunks: Number of chunks to create.

        Returns:
            List of (chunk_path, chunk_dir) tuples.
        """
        chunk_info: list[tuple[str, str]] = []
        basename = self.input_path.stem

        for i in range(num_chunks):
            # Skip empty chunks (can happen with multi-GPU and few molecules)
            if not chunk_idxes[i]:
                print(f"Job{i + 1}, number of inputs: 0 (skipped)", flush=True)
                if self.logger:
                    self.logger.info(f"Job{i + 1}, number of inputs: 0 (skipped)")
                continue

            chunk_dir = self.job_dir / f"job{i + 1}"
            chunk_dir.mkdir()

            if self.input_format == "smi":
                chunk_path = chunk_dir / f"{basename}_{i + 1}.smi"
                df_chunk = df.iloc[chunk_idxes[i], :]
                df_chunk.to_csv(str(chunk_path), header=None, index=None, sep=" ")
                count = len(df_chunk)
            else:  # sdf
                chunk_path = chunk_dir / f"{basename}_{i + 1}.sdf"
                chunks = [df[j] for j in chunk_idxes[i]]
                chunk_path.write_text("".join(line for chunk in chunks for line in chunk))
                count = len(chunks)

            print(f"Job{i + 1}, number of inputs: {count}", flush=True)
            if self.logger:
                self.logger.info(f"Job{i + 1}, number of inputs: {count}")

            chunk_info.append((str(chunk_path), str(chunk_dir)))

        return chunk_info

    def _run_pipeline(self, chunk_info: list[tuple[str, str]]) -> None:
        """Run the isomer generation and optimization pipeline.

        Args:
            chunk_info: List of (chunk_path, chunk_dir) tuples.
        """
        from Auto3D.auto3D import isomer_wraper, optim_rank_wrapper

        chunk_queue: Queue[tuple[str, str, str, int] | str] = mp.Manager().Queue()

        # Create isomer generation process
        p1 = mp.Process(
            target=isomer_wraper,
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

        # Wait for completion
        p1.join()
        for p2 in p2s:
            p2.join()

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

        # Write combined output
        path_combined = self.job_dir / f"{self.input_path.stem}_out.sdf"
        path_combined.write_text("".join(combined_data))

        # Log timing
        self._log_timing(start_time)

        # Reorder and decode IDs
        reorder_sdf(str(path_combined), str(self.input_path))
        path_output = decode_ids(str(path_combined), self.id_mapping)

        # Cleanup temporary files
        self.input_path.unlink()
        path_combined.unlink()

        print(f"Output path: {path_output}", flush=True)
        if self.logger:
            self.logger.info(f"Output path: {path_output}")

        # Shutdown logging
        if self.logging_queue:
            self.logging_queue.put(None)
            time.sleep(3)

        return path_output

    def _log_timing(self, start_time: float) -> None:
        """Log pipeline execution time.

        Args:
            start_time: Pipeline start time.
        """
        print("Energy unit: Hartree if implicit.", flush=True)
        if self.logger:
            self.logger.info("Energy unit: Hartree if implicit.")

        elapsed_minutes = int((time.time() - start_time) / 60)

        if elapsed_minutes <= 60:
            msg = f"Program running time: {elapsed_minutes + 1} minute(s)"
        else:
            hours = elapsed_minutes // 60
            remaining = elapsed_minutes - hours * 60
            msg = f"Program running time: {hours} hour(s) and {remaining} minute(s)"

        print(msg, flush=True)
        if self.logger:
            self.logger.info(msg)


def run_workflow(config: Auto3DOptions) -> str:
    """Convenience function to run the workflow.

    Args:
        config: Auto3D configuration options.

    Returns:
        Path to the output SDF file.
    """
    orchestrator = WorkflowOrchestrator(config)
    return orchestrator.run()
