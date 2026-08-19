"""Chunk management for Auto3D parallel processing.

This module provides the ChunkManager class for handling the division
of input data into chunks for parallel processing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import psutil

from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.sdf_io import SDF2chunks

if TYPE_CHECKING:
    import logging

    from Auto3D.foundation.config import Auto3DOptions

logger = get_logger(__name__)

# Ceiling for the memory-scaled `batchsize_atoms` multiplier (audit M36). A
# bare `batchsize_atoms * memory_gb` has no upper bound: on an 80 GB GPU the
# documented default (1024 atoms/GB) scales to 81,920 atoms in a single NN
# forward call, which -- combined with `optimizing.BUCKET_MAX_COUNT = 1024`
# molecules per bucket -- can mean one flattened AIMNet2 call over an entire
# max-size bucket. 1024 * 16 matches `EnForce_ANI`'s own default
# `batchsize_atoms` (batch_opt/model_wrapper.py), i.e. the ceiling already
# considered a safe single-call size everywhere else `EnForce_ANI` is
# constructed without an explicit override.
_MAX_SCALED_BATCHSIZE_ATOMS = 1024 * 16


def _gpu_free_memory_gb(gpu_idx: int) -> int | None:
    """Free memory (GB, floor) for one GPU, without initializing a CUDA context.

    This orchestrator process never runs a model -- it only sizes chunks
    before workers are spawned. Two torch.cuda calls look like the obvious
    way to answer "how much memory does this GPU have", and both are wrong
    here:

    * ``torch.cuda.get_device_properties()`` calls ``torch.cuda._lazy_init()``
      directly (see ``torch/cuda/__init__.py``), which runs
      ``torch._C._cuda_init()`` and brings up the full CUDA runtime state
      (caching allocator, streams, RNG) for the calling process.
    * ``torch.cuda.mem_get_info()`` does not call ``_lazy_init()`` itself, but
      it reaches the CUDA Runtime's ``cudaMemGetInfo``, which -- like nearly
      every runtime call that touches a specific device -- lazily creates
      that device's primary CUDA context if one does not already exist. It
      swaps "total memory" for "free memory" (the other half of this finding)
      without avoiding the context-creation hazard.

    ``nvidia-smi`` is a separate process built on NVML, Nvidia's *management*
    library, which is deliberately decoupled from the CUDA driver/runtime for
    exactly this reason -- it is the same property ``torch.cuda.
    is_available()``/``device_count()`` use internally to offer an
    NVML-based check that "will NOT poison fork" (see the ``_nvml_based_avail``
    path in ``torch/cuda/__init__.py``). Shelling out to it is the only way
    this process can answer the question without paying the cost the question
    is trying to measure.

    ``gpu_idx`` is a **CUDA-visible** index, which is not an ``nvidia-smi``
    index. PyTorch numbers devices after ``CUDA_VISIBLE_DEVICES`` remapping,
    while ``nvidia-smi -i`` numbers physical cards: under
    ``CUDA_VISIBLE_DEVICES=4,5``, ``gpu_idx=0`` is physical GPU 4, so passing 0
    straight through would report a different card's free memory and size every
    chunk from it. Shared multi-GPU boxes -- the environment this scaling exists
    for -- are exactly where that is set, so the index is translated first. An
    entry may be a UUID rather than an ordinal (``CUDA_VISIBLE_DEVICES=GPU-abc...``),
    which ``-i`` also accepts, so entries are passed through as-is.

    Returns:
        Free memory in whole GB (floored, minimum 1), or ``None`` if
        ``nvidia-smi`` is unavailable, times out, or returns something
        unparsable (no NVIDIA driver, non-NVIDIA GPU, sandboxed/minimal
        container), or if ``CUDA_VISIBLE_DEVICES`` does not contain
        ``gpu_idx``. Never raises, and never touches ``torch.cuda``.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        smi_id = str(gpu_idx)
    else:
        entries = [e.strip() for e in visible.split(",") if e.strip()]
        if gpu_idx >= len(entries):
            # The caller asked for a device CUDA cannot see. Reporting some
            # other card's memory would be worse than declining to guess;
            # get_device() raises GPUError for this case anyway.
            return None
        smi_id = entries[gpu_idx]

    try:
        result = subprocess.run(
            [
                exe,
                "-i",
                smi_id,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        free_mib = float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return max(1, int(free_mib / 1024))


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
            free_gb = _gpu_free_memory_gb(gpu_idx)
            if free_gb is None:
                # nvidia-smi unavailable/unparsable (no NVIDIA driver,
                # non-NVIDIA GPU, minimal container). Fall back to a
                # conservative default instead of reaching for torch.cuda,
                # which would trade an unmeasured multiplier for the exact
                # parent-process CUDA context this function exists to avoid
                # (audit M36). 1 GB keeps batchsize_atoms/chunk_size at their
                # unscaled defaults, matching what an explicit --memory=1
                # would do.
                self._log_info(
                    "nvidia-smi unavailable; could not detect GPU memory. "
                    "Using unscaled batchsize_atoms/chunk_size. Pass "
                    "`memory=<GB>` to size chunks explicitly."
                )
                memory_gb = 1
            else:
                memory_gb = free_gb
        else:
            memory_gb = int(psutil.virtual_memory().total / (1024**3))

        # Clamp to at least 1: a fractional/zero capacity would otherwise make
        # data_size // chunk_size explode num_chunks or raise ZeroDivisionError.
        chunk_size = max(1, int(memory_gb * self.config.capacity))
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
        #
        # Clamped to _MAX_SCALED_BATCHSIZE_ATOMS (audit M36) but never below
        # the caller's own unscaled batchsize_atoms: the cap bounds what
        # memory-scaling can produce, it does not second-guess an explicit,
        # already-large user setting that made no use of scaling at all
        # (e.g. a fixed `memory=1`).
        scaled = self.config.batchsize_atoms * memory_gb
        self.scaled_batchsize_atoms = max(
            self.config.batchsize_atoms,
            min(scaled, _MAX_SCALED_BATCHSIZE_ATOMS),
        )

        # Read input data
        if self.input_format == "smi":
            # Robust read matching encode_ids semantics: take only the first two
            # whitespace-separated columns, ignore any extra columns, and skip
            # blank lines. The python engine + usecols tolerates ragged rows
            # (an extra token) that the default C engine rejects with
            # "Expected 2 fields, saw 3".
            df = pd.read_csv(
                str(self.input_path),
                sep=r"\s+",
                header=None,
                names=[0, 1],
                usecols=[0, 1],
                engine="python",
                skip_blank_lines=True,
                dtype=str,
            )
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
