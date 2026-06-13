#!/usr/bin/env python
"""
Generating low-energy conformers from SMILES.
"""
from __future__ import annotations

import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rdkit import Chem

if TYPE_CHECKING:
    from collections.abc import Callable

    from Auto3D.results import WorkflowResult

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import ConfigurationError
from Auto3D.isomers import IsomerEngineFactory
from Auto3D.ranking import ranking
from Auto3D.utils import (
    check_input,
    create_chunk_meta_names,
    reorder_sdf,
)
from Auto3D.utils.file_ops import smiles2smi
from Auto3D.utils.logging_config import configure_logging, get_logger

# Pipeline workers live in workflow_workers to break the auto3D<->workflow import
# cycle. Re-exported here for backward compatibility with code/tests that import
# them from Auto3D.auto3D.
from Auto3D.workflow_workers import (  # noqa: F401
    isomer_wrapper,
    logger_process,
    optim_rank_wrapper,
)

logger = get_logger(__name__)

# Note: TF32 settings are now configured via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. Configuration is applied at pipeline start.


def main(
    args: Auto3DOptions,
    progress_callback: Callable[[dict], None] | None = None,
) -> WorkflowResult:
    """Run the Auto3D conformer generation pipeline.

    Args:
        args: Configuration options as an ``Auto3DOptions`` instance.
        progress_callback: Optional callable invoked with per-step optimizer
            progress events (dicts with ``job``/``step``/``total``/``converged``/
            ``dropped``/``active``) for a live display. Defaults to None, which
            leaves the pipeline behavior unchanged.

    Returns:
        A :class:`Auto3D.results.WorkflowResult` -- a ``str`` subclass holding
        the output SDF path (so it works anywhere the path string did) plus the
        run's ``n_molecules`` / ``n_conformers`` counts.

    Raises:
        SystemExit: If input validation fails or no structures converge.
    """
    # Configure logging based on verbose setting
    configure_logging(verbose=args.verbose)

    from Auto3D.workflow import WorkflowOrchestrator

    # Force the 'spawn' start method for the optimization worker processes.
    #
    # The workers run PyTorch. Forking a process that has already initialized a
    # CUDA context yields a broken context in the child: the worker crashes and
    # the run produces no output (surfacing as "no 3D structure converged").
    # force=True is REQUIRED, not optional: a default-context ProcessPoolExecutor
    # (the isomer-embedding pool, or an earlier pipeline run in the same process)
    # can already have locked the global start method to the platform default
    # ('fork' on Linux). The previous best-effort set_start_method("spawn") would
    # then raise RuntimeError, get swallowed, and the pipeline silently ran on
    # fork -- breaking whenever any CUDA work had touched the parent process
    # (e.g. a prior use_gpu=True call in the same interpreter / test session).
    mp.set_start_method("spawn", force=True)

    from Auto3D.results import WorkflowResult

    orchestrator = WorkflowOrchestrator(args, progress_callback=progress_callback)
    return WorkflowResult(orchestrator.run())

def smiles2mols(smiles: list[str], args: Auto3DOptions) -> list[Chem.Mol]:
    """Find low-energy conformers for a list of SMILES.

    A convenient single-process function for small batches. For larger batches
    (>150 SMILES), use the ``main()`` function for better performance.

    Args:
        smiles: List of SMILES strings to generate conformers for.
        args: Configuration options as an ``Auto3DOptions`` instance.

    Returns:
        List of RDKit Mol objects representing low-energy conformers.

    Raises:
        ConfigurationError: If neither k nor window is specified.
    """
    # Configure PyTorch settings (TF32, cuDNN benchmark)
    from Auto3D.torch_config import TorchConfig, configure_torch
    torch_config = TorchConfig(allow_tf32=args.allow_tf32)
    configure_torch(torch_config)

    with tempfile.TemporaryDirectory() as tmpdirname:
        path0 = str(Path(tmpdirname) / "smiles.smi")
        smiles2smi(smiles, path0)  # save all SMILES into a smi file
        args['path'] = path0
        k = args.k
        window = args.window
        if (not k) and (not window):
            raise ConfigurationError(
                "Either k or window needs to be specified. "
                "Usually, setting '--k=1' satisfies most needs."
            )
        args.input_format = 'smi'
        check_input(args)

        # smi to sdf
        meta = create_chunk_meta_names(path0, tmpdirname)
        isomer_engine = IsomerEngineFactory.create(
            engine_type="rdkit",
            input_path=path0,
            output_path=meta["enumerated_sdf"],
            smiles_enumerated=meta["smiles_enumerated"],
            smiles_reduced=meta["smiles_reduced"],
            smiles_hashed=meta["smiles_hashed"],
            job_dir=tmpdirname,
            max_confs=args.max_confs,
            threshold=args.threshold,
            n_jobs=args.mpi_np,
            enumerate_isomers=args.enumerate_isomer,
        )
        isomer_engine.run()

        # optimize conformers
        if args.use_gpu:
            if isinstance(args.gpu_idx, int):
                idx = args.gpu_idx
            else:
                idx = args.gpu_idx[0]
            device = torch.device(f"cuda:{idx}")
        else:
            device = torch.device("cpu")
        opt_config = args.to_optimization_config()
        opt_engine = optimizing(meta["enumerated_sdf"], meta["optimized_og"],
                                args.optimizing_engine, device, opt_config)
        opt_engine.run()

        # Ranking step
        rank_engine = ranking(meta["optimized_og"], meta["output"],
                              args.threshold, k=k, window=window)
        _ = rank_engine.run()
        conformers = reorder_sdf(meta["output"], path0)

        logger.info("Energy unit: Hartree if implicit.")
    return conformers
