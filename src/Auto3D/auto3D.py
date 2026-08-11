#!/usr/bin/env python
"""
Generating low-energy conformers from SMILES.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem

if TYPE_CHECKING:
    from collections.abc import Callable

    from Auto3D.results import WorkflowResult
    from Auto3D.workflow_workers import ProgressEvent

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import ConfigurationError
from Auto3D.isomers import IsomerEngineFactory
from Auto3D.job_layout import create_chunk_meta_names
from Auto3D.model_factory import create_model, get_device
from Auto3D.models.preflight import preflight_model
from Auto3D.pipeline.input_checks import check_input, check_valid_configuration
from Auto3D.ranking import ranking
from Auto3D.utils.logging_config import configure_logging, get_logger
from Auto3D.utils.reconciliation import find_smiles_not_in_sdf
from Auto3D.utils.sdf_io import reorder_sdf
from Auto3D.utils.smi_io import smiles2smi

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
    progress_callback: Callable[[ProgressEvent], None] | None = None,
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
        run's ``n_molecules`` / ``n_conformers`` counts and its ``failures``
        list (input molecule IDs reconciled away as missing from the output,
        see ``WorkflowOrchestrator._finalize_output``).

    Raises:
        ConfigurationError: If path is None, k/window not specified, or the
            configuration (including the optimizing engine name) is
            otherwise invalid.
        FileFormatError: If the input file format is not supported.
        OptimizationError: If no structure converged.
        ModelLoadError: If the optimizing model could not be obtained or
            loaded.
        DependencyError: If a required optional dependency is missing.

        None of the above is a ``SystemExit``: ``main()`` itself catches
        nothing and lets ``WorkflowOrchestrator.run()``'s exceptions (see
        ``WorkflowOrchestrator._validate_input``/``_finalize_output``)
        propagate as-is. Only the CLI layer (``cli/errors.handle_error``)
        converts them to a process exit code; a direct Python-API caller
        gets the exception object.
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
    output_path = orchestrator.run()
    # getattr, not a direct attribute access: mirrors the defensive read the
    # CLI already does for n_molecules/n_conformers (cli/commands/run.py), so
    # a test/mock that swaps in a bare stand-in for the orchestrator (e.g.
    # test_mp_start_method.py) still gets a valid WorkflowResult instead of an
    # AttributeError.
    return WorkflowResult(output_path, failures=getattr(orchestrator, "failures", None))


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
        ConfigurationError: If neither k nor window is specified, the
            optimizing engine name is not recognized, the configuration is
            otherwise invalid (e.g. an out-of-range ``gpu_idx``), or an
            option this function cannot honor is requested --
            ``enumerate_tautomer=True`` (no tautomer-enumeration step) or a
            non-``'rdkit'`` ``isomer_engine`` (the RDKit engine is
            hardcoded, so ``mode_oe`` has no effect either). Use ``main()``
            for either of those.
        ModelLoadError: If the optimizing model could not be obtained or
            loaded.
        DependencyError: If a required optional dependency is missing.
    """
    # Copy the caller's config up front: smiles2mols must not mutate the
    # object it was given (M15). Every assignment below (path, input_format)
    # lands on this private copy; `args` no longer refers to the caller's
    # object for the rest of this function.
    args = replace(args)

    # smiles2mols has no tautomer-enumeration step and hardcodes the RDKit
    # isomer engine below (mode_oe only ever affects the omega engine, so it
    # has no effect here regardless) -- silently ignoring these three options
    # was M15. Raise instead of letting the caller believe they took effect.
    if args.enumerate_tautomer:
        raise ConfigurationError(
            "smiles2mols does not support enumerate_tautomer=True: it has no "
            "tautomer-enumeration step. Use main() for tautomer enumeration."
        )
    if args.isomer_engine.lower() != "rdkit":
        raise ConfigurationError(
            f"smiles2mols only supports isomer_engine='rdkit', got "
            f"{args.isomer_engine!r}. The RDKit engine is hardcoded here "
            "(mode_oe has no effect either way). Use main() for a "
            "non-RDKit isomer engine."
        )

    # Configure PyTorch settings (TF32, cuDNN benchmark)
    from Auto3D.torch_config import TorchConfig, configure_torch

    torch_config = TorchConfig(allow_tf32=args.allow_tf32)
    configure_torch(torch_config)

    with tempfile.TemporaryDirectory() as tmpdirname:
        path0 = str(Path(tmpdirname) / "smiles.smi")
        smiles2smi(smiles, path0)  # save all SMILES into a smi file
        args.path = path0
        k = args.k
        window = args.window
        if (not k) and (not window):
            raise ConfigurationError(
                "Either k or window needs to be specified. "
                "Usually, setting '--k=1' satisfies most needs."
            )
        args.input_format = "smi"

        # Fail fast on an invalid configuration (notably an out-of-range
        # gpu_idx) the same way main() does via WorkflowOrchestrator --
        # check_input alone does not catch this, so it used to only surface
        # opaquely deep inside optimization.
        config_errors = check_valid_configuration(args)
        if config_errors:
            raise ConfigurationError("Invalid configuration:\n  - " + "\n  - ".join(config_errors))

        check_input(args)

        # Resolve the engine name and verify the model is obtainable HERE
        # (C8/M22), before the `optimizing` step below constructs its own copy
        # for real work. A cold cache with no network, a corrupted cached
        # file, or an unwritable cache directory would otherwise surface only
        # deep inside ranking/optimization as an opaque failure.
        preflight_model(args.optimizing_engine)

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
            use_parallel_embedding=args.use_parallel_embedding,
            parallel_workers=args.parallel_workers,
            parallel_embedding_threshold=args.parallel_embedding_threshold,
        )
        isomer_engine.run()

        # optimize conformers. gpu_idx may be a single int or a list (one
        # entry per GPU, for main()'s multi-process path); smiles2mols is
        # single-process, so only the first index is ever used. Resolved
        # through model_factory.get_device -- the single owner of gpu_idx ->
        # torch.device -- rather than re-building the `cuda:{idx}` string
        # here, which used to bypass get_device's own out-of-range GPUError
        # entirely (defense in depth: check_valid_configuration above already
        # range-checks gpu_idx for this entry point, but get_device is where
        # every other caller, incl. calc_spe/opt_geometry/calc_thermo, gets
        # that check).
        idx = args.gpu_idx if isinstance(args.gpu_idx, int) else args.gpu_idx[0]
        device = get_device(idx, use_gpu=args.use_gpu)
        opt_config = args.to_optimization_config()
        # Built in this process, which is also the one that runs the
        # optimization -- `smiles2mols` is single-process, so there is no spawn
        # boundary here, but see `Auto3D.workflow_workers.optim_rank_wrapper`
        # for why construction must never be hoisted past the frame that works.
        adapter = create_model(args.optimizing_engine, device)
        opt_engine = optimizing(
            meta["enumerated_sdf"],
            meta["optimized_og"],
            adapter=adapter,
            device=device,
            config=opt_config,
        )
        opt_engine.run()

        # Ranking step
        rank_engine = ranking(
            meta["optimized_og"], meta["output"], args.threshold, k=k, window=window
        )
        _ = rank_engine.run()
        conformers = reorder_sdf(meta["output"], path0)

        # Reconcile inputs against outputs (C7): smiles2mols never runs
        # encode_ids/decode_ids -- path0 already holds the same (InChIKey-based)
        # ids that ranking/reorder_sdf wrote to meta["output"] -- so this is
        # already the right pair to compare, no decoding needed. This call's
        # own logging is the report; smiles2mols keeps its `list[Chem.Mol]`
        # return type (unlike main(), nothing here reads a `.failures` carrier
        # today) so a missing input surfaces in the log rather than silently
        # nowhere, closing the "zero production callers" gap for this path too.
        find_smiles_not_in_sdf(path0, meta["output"])

        logger.info("Energy unit: Hartree if implicit.")
    return conformers
