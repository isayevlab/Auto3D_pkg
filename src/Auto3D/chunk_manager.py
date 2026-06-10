"""Chunk management for Auto3D parallel processing.

This module provides the ChunkManager class for handling the division
of input data into chunks for parallel processing.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import psutil
import torch

from Auto3D.utils.file_ops import SDF2chunks
from Auto3D.utils.logging_config import get_logger

if TYPE_CHECKING:
    import logging

    from Auto3D.config import Auto3DOptions

logger = get_logger(__name__)


class ChunkManager:
    """Manages the division of input data into chunks for parallel processing.

    This class encapsulates the logic for calculating memory availability,
    determining chunk sizes, and creating chunk files for parallel execution.

    Args:
        config: Auto3D configuration options.
        input_path: Path to the input file.
        input_format: Format of input file ('smi' or 'sdf').
        job_dir: Directory for job output.
        workflow_logger: Optional logger for workflow-level logging.
    """

    def __init__(
        self,
        config: Auto3DOptions,
        input_path: Path,
        input_format: str,
        job_dir: Path,
        workflow_logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.input_path = input_path
        self.input_format = input_format
        self.job_dir = job_dir
        self.workflow_logger = workflow_logger
        # Memory-scaled atom batch size, computed in prepare_chunks(). Kept on
        # the manager (not written back to the shared config) so that calling
        # main() twice with the same Auto3DOptions does not compound the
        # multiplier (OOM risk). Defaults to the unscaled config value.
        self.scaled_batchsize_atoms: int = config.batchsize_atoms

    def calculate_memory_and_chunks(self) -> tuple[int, int, int]:
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

    def prepare_chunks(self) -> list[tuple[str, str]]:
        """Prepare input chunks for parallel processing.

        Returns:
            List of (chunk_path, chunk_dir) tuples.
        """
        memory_gb, chunk_size, num_jobs = self.calculate_memory_and_chunks()

        # Scale batchsize by available memory. Store on the manager rather than
        # mutating self.config: the config is shared with the caller and the
        # optimization workers, and mutating it in place would compound the
        # multiplier on repeated main() calls (review findings #35/#36).
        self.scaled_batchsize_atoms = self.config.batchsize_atoms * memory_gb

        # Read input data
        if self.input_format == "smi":
            df = pd.read_csv(str(self.input_path), sep=r"\s+", header=None)
        else:  # sdf
            df = SDF2chunks(str(self.input_path))

        data_size = len(df)
        num_chunks = max(int(data_size // chunk_size + 1), num_jobs)

        self._log_info(f"The available memory is {memory_gb} GB.")
        self._log_info(f"The task will be divided into {num_chunks} jobs.")

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
                self._log_info(f"Job{i + 1}, number of inputs: 0 (skipped)")
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

            self._log_info(f"Job{i + 1}, number of inputs: {count}")
            chunk_info.append((str(chunk_path), str(chunk_dir)))

        return chunk_info

    def _log_info(self, message: str) -> None:
        """Log message to both module logger and workflow logger.

        Args:
            message: Message to log.
        """
        logger.info(message)
        if self.workflow_logger:
            self.workflow_logger.info(message)
