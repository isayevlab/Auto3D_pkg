"""Chunk management for Auto3D parallel processing.

This module provides the ChunkManager class for handling the division
of input data into chunks for parallel processing.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.sdf_io import SDF2chunks
from Auto3D.foundation.utils.smi_io import iter_smi_records, write_smi_records

if TYPE_CHECKING:
    import logging

    from Auto3D.foundation.config import Auto3DOptions

logger = get_logger(__name__)

# Where this process' cgroup memory limit, if any, is published: v2
# (the unified hierarchy) first, with the v1 layout as a fallback for hosts
# that have not migrated. Module-level `Path` constants, not inline literals,
# so a test can point either at a controlled file instead of the real
# filesystem -- which may or may not have either path, depending on the host.
CGROUP_MEMORY_MAX_PATH = Path("/sys/fs/cgroup/memory.max")
CGROUP_V1_LIMIT_PATH = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")


def _read_cgroup_limit_bytes(path: Path) -> int | None:
    """One cgroup memory-limit file's value in bytes, or None.

    None covers every "nothing to report" case alike: the file does not
    exist (no cgroup, or a different layout), this process cannot read it
    (sandboxed), its content is the literal ``"max"`` (cgroup v2's spelling
    for "no limit"), or its content is not a whole number of bytes at all.
    The caller's fallback -- the host's available memory -- is the same
    answer regardless of which of these it was.
    """
    try:
        text = path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _cgroup_memory_limit_bytes() -> int | None:
    """This process' cgroup memory limit in bytes, or None if unconfined.

    v2's ``memory.max`` is authoritative when it names an actual limit; v1's
    ``memory.limit_in_bytes`` is consulted only when v2 is absent, ``"max"``,
    or unparsable. A v1-only host never has a `memory.max` file at all, so
    "v2 absent" is the ordinary way v1 is reached here, not a rare fallback.
    """
    limit = _read_cgroup_limit_bytes(CGROUP_MEMORY_MAX_PATH)
    if limit is not None:
        return limit
    return _read_cgroup_limit_bytes(CGROUP_V1_LIMIT_PATH)


# Ceiling for the memory-scaled `batchsize_atoms` multiplier. A
# bare `batchsize_atoms * memory_gb` has no upper bound: on an 80 GB GPU the
# documented default (1024 atoms/GB) scales to 81,920 atoms in a single NN
# forward call, which -- combined with `Optimizer.BUCKET_MAX_COUNT = 1024`
# molecules per bucket -- can mean one flattened AIMNet2 call over an entire
# max-size bucket. 1024 * 16 is bounded by a direct measurement, not by
# coincidence with some other default: a 2026-10-09 bucket-policy run merged
# 24 molecules (14 to 53 atoms, 240 conformers total) through both AIMNet2
# and ANI2xt at this batch size with no failure, which confirms a merged
# bucket this size is safe even though the two engines pay for it
# differently -- AIMNet2 skips its padded slots while ANI2xt still computes
# descriptors on them, so ANI2xt is the more expensive of the two at the same
# nominal batch size.
_MAX_SCALED_BATCHSIZE_ATOMS = 1024 * 16


def scaled_batchsize_atoms(batchsize_atoms: int, memory_gb: int) -> int:
    """The absolute atoms-per-batch for a per-gigabyte setting and a memory budget.

    ``Auto3DOptions.batchsize_atoms`` is a per-gigabyte figure and
    ``OptimizationConfig.batchsize_atoms`` is an absolute one; this is the rule
    between them. A module-level function rather than a ``ChunkManager`` method
    because ``smiles2mols`` needs the same rule and has no chunking step to hang
    it off -- it read the field absolutely instead, which left the two entry
    points up to 16x apart on one config.

    The product is clamped to ``_MAX_SCALED_BATCHSIZE_ATOMS`` but
    never to below the caller's own unscaled ``batchsize_atoms``: the cap bounds
    what memory-scaling can produce, it does not second-guess an explicit,
    already-large user setting that made no use of scaling at all (e.g. a fixed
    ``memory=1``).

    Args:
        batchsize_atoms: Atoms per batch per gigabyte, from the config.
        memory_gb: The memory budget in GB, as set or measured by
            ``ChunkManager.calculate_memory_and_chunks``.

    Returns:
        The absolute number of atoms per optimization batch.
    """
    return max(batchsize_atoms, min(batchsize_atoms * memory_gb, _MAX_SCALED_BATCHSIZE_ATOMS))


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
        # Memory-scaled atom batch size, computed in prepare_chunks() by the
        # module-level `scaled_batchsize_atoms` function above -- which shares
        # this attribute's name, so `self.scaled_batchsize_atoms =
        # scaled_batchsize_atoms(...)` there is an assignment from the function,
        # not a recursive one. Kept on the manager (not written back to the
        # shared config) so that calling main() twice with the same
        # Auto3DOptions does not compound the multiplier (OOM risk). Defaults to
        # the unscaled config value.
        self.scaled_batchsize_atoms: int = config.batchsize_atoms

    def calculate_memory_and_chunks(self) -> tuple[int, int, int]:
        """Calculate available memory and chunk configuration.

        Returns:
            Tuple of (memory_gb, chunk_size, num_jobs).
        """
        # num_jobs reflects every GPU the caller listed, independent of which
        # branch below ends up supplying memory_gb: an explicit
        # --memory must not collapse several listed GPUs back down to a
        # single job just because memory_gb no longer needs measuring.
        if self.config.use_gpu and isinstance(self.config.gpu_idx, list):
            num_jobs = len(self.config.gpu_idx)
        else:
            num_jobs = 1

        if self.config.memory is not None:
            memory_gb = int(self.config.memory)
        elif self.config.use_gpu:
            gpu_idx = (
                self.config.gpu_idx[0]
                if isinstance(self.config.gpu_idx, list)
                else self.config.gpu_idx
            )
            free_gb = _gpu_free_memory_gb(gpu_idx)
            if free_gb is None:
                # nvidia-smi unavailable/unparsable (no NVIDIA driver,
                # non-NVIDIA GPU, minimal container). Fall back to a
                # conservative default instead of reaching for torch.cuda,
                # which would trade an unmeasured multiplier for the exact
                # parent-process CUDA context this function exists to avoid.
                # 1 GB keeps batchsize_atoms/chunk_size at their
                # unscaled defaults, matching what an explicit --memory=1
                # would do.
                self._log_info(
                    "nvidia-smi unavailable; could not detect GPU memory. "
                    "Falling back to a 1 GB budget, so `batchsize_atoms` (and "
                    "chunk sizing, where it applies) stays unscaled. Pass "
                    "`memory=<GB>` to set the budget explicitly."
                )
                memory_gb = 1
            else:
                memory_gb = free_gb
        else:
            # `available`, not `total`: `total` ignores every other
            # process already resident on a shared box, which makes chunk
            # sizing -- and the batchsize_atoms scaling derived from it --
            # wildly optimistic. Further clamped to this process' cgroup
            # memory limit, if any: a container or shared box can confine it
            # to far less than psutil reports for the whole machine. Floored
            # at 1 GB either way, matching the GPU branch's own floor
            # (`_gpu_free_memory_gb`'s `max(1, ...)` and the nvidia-smi
            # fallback's explicit `memory_gb = 1`): under 1 GiB, the raw
            # `int(bytes / 1024**3)` truncates to 0, which would collapse
            # chunk sizing -- and the batchsize_atoms scaling derived from
            # it -- to its unscaled floor instead of merely being
            # conservative.
            available = psutil.virtual_memory().available
            limit = _cgroup_memory_limit_bytes()
            if limit is not None and limit < available:
                raw_gb = int(limit / (1024**3))
                memory_gb = max(1, raw_gb)
                floor_note = " (floored to 1 GB)" if raw_gb < 1 else ""
                self._log_info(
                    f"cgroup memory limit ({int(limit / 1024**3)} GB) is lower "
                    f"than the host's available memory "
                    f"({int(available / 1024**3)} GB); using the cgroup "
                    f"limit{floor_note}."
                )
            else:
                raw_gb = int(available / (1024**3))
                memory_gb = max(1, raw_gb)
                floor_note = " (floored to 1 GB)" if raw_gb < 1 else ""
                self._log_info(
                    f"Using the host's available memory: {int(available / 1024**3)} GB{floor_note}."
                )

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
        # multiplier on repeated main() calls. The
        # rule itself is `scaled_batchsize_atoms` above, which smiles2mols --
        # which never chunks, so never reaches this method -- calls directly.
        self.scaled_batchsize_atoms = scaled_batchsize_atoms(self.config.batchsize_atoms, memory_gb)

        # Read input data
        if self.input_format == "smi":
            # iter_smi_records is the single parser for this format: the
            # same one `auto3d validate` and the run pipeline use,
            # so an already-ID-encoded `.smi` cannot be read one way here and
            # a different way everywhere else. on_malformed left at its
            # lenient default ("skip") matches what the previous pandas read
            # did for a short/blank line: skip it rather than raise.
            records = [
                (smi, mol_id) for _line_no, smi, mol_id in iter_smi_records(str(self.input_path))
            ]
        else:  # sdf
            records = SDF2chunks(str(self.input_path))

        data_size = len(records)
        # math.ceil, not `data_size // chunk_size + 1`: the "+1" added
        # an extra chunk even when data_size divides chunk_size evenly, which
        # _create_chunk_files' empty-chunk skip then dropped -- but only after
        # it had already been allocated and counted toward num_chunks, one
        # more job than the work needs.
        num_chunks = max(math.ceil(data_size / chunk_size), num_jobs)

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
        return self._create_chunk_files(records, chunk_idxes, num_chunks)

    def _create_chunk_files(
        self,
        records: list[tuple[str, str]] | list[list[str]],
        chunk_idxes: list[list[int]],
        num_chunks: int,
    ) -> list[tuple[str, str]]:
        """Create individual chunk files for parallel processing.

        Args:
            records: Input data -- (smiles, id) pairs for SMI, or SDF2chunks'
                per-molecule line lists for SDF.
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
                chunk_records = [records[j] for j in chunk_idxes[i]]
                write_smi_records(chunk_path, chunk_records)
                count = len(chunk_records)
            else:  # sdf
                chunk_path = chunk_dir / f"{basename}_{i + 1}.sdf"
                chunks = [records[j] for j in chunk_idxes[i]]
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
