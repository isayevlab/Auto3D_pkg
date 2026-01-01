#!/usr/bin/env python
"""
Generating low-energy conformers from SMILES.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import tempfile
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rdkit import Chem
from send2trash import send2trash

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.config import Auto3DOptions
from Auto3D.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RMSD_THRESHOLD,
)
from Auto3D.exceptions import ConfigurationError
from Auto3D.isomer_engine import oe_isomer, rd_isomer, rd_isomer_sdf, tautomer_engine
from Auto3D.ranking import ranking
from Auto3D.utils import (
    check_input,
    create_chunk_meta_names,
    hash_taut_smi,
    housekeeping,
    reorder_sdf,
)
from Auto3D.utils.file_ops import smiles2smi
from Auto3D.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from logging import LogRecord
    from multiprocessing import Queue

# Note: TF32 settings are now configured via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions. Configuration is applied at pipeline start.


def isomer_wrapper(
    chunk_info: list[tuple[str, str]],
    args: Auto3DOptions,
    queue: Queue[tuple[str, str, str, int] | str],
    logging_queue: Queue[LogRecord | None],
) -> None:
    """Generate isomers for chunks and put results in queue.

    Args:
        chunk_info: List of (path, dir) tuples for each chunk.
        args: Auto3D configuration options.
        queue: Queue for passing enumerated SDF paths to optimizer.
        logging_queue: Queue for centralized logging.
    """
    #prepare logging
    logger = logging.getLogger("auto3d")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

    for i, path_dir in enumerate(chunk_info):
        logger.info(f"\n\nIsomer generation for job{i+1}")
        path, dir = path_dir
        meta = create_chunk_meta_names(path, dir)

        # Tautomer enumeratioin
        if args.enumerate_tautomer:
            output_taut = meta["output_taut"]
            taut_mode = args.tauto_engine
            logger.info("Enumerating tautomers for the input...")
            taut_engine = tautomer_engine(taut_mode, path, output_taut, args.pKaNorm)
            taut_engine.run()
            hash_taut_smi(output_taut, output_taut)
            path = output_taut
            logger.info(f"Tautomers are saved in {output_taut}")

        smiles_enumerated = meta["smiles_enumerated"]
        smiles_reduced = meta["smiles_reduced"]
        smiles_hashed = meta["smiles_hashed"]
        enumerated_sdf = meta["enumerated_sdf"]
        max_confs = args.max_confs
        duplicate_threshold = args.threshold
        mpi_np = args.mpi_np
        enumerate_isomer = args.enumerate_isomer
        isomer_program = args.isomer_engine
        # Isomer enumeration step
        if isomer_program == 'omega':
            mode_oe = args.mode_oe
            oe_isomer(mode_oe, path, smiles_enumerated, smiles_reduced, smiles_hashed,
                    enumerated_sdf, max_confs, duplicate_threshold, enumerate_isomer)
        elif isomer_program == 'rdkit':
            if args.input_format == 'smi':
                engine = rd_isomer(path, smiles_enumerated, smiles_reduced, smiles_hashed, 
                                enumerated_sdf, dir, max_confs, duplicate_threshold, mpi_np, enumerate_isomer)
                engine.run()
            elif args.input_format == 'sdf':
                engine = rd_isomer_sdf(path, enumerated_sdf, max_confs, duplicate_threshold, mpi_np)
                engine.run()
        else: 
            raise ValueError('The isomer enumeration engine must be "omega" or "rdkit", '
                            f'but {args.isomer_engine} was parsed. '
                            'You can set the parameter by appending the following:'
                            '--isomer_engine=rdkit')

        queue.put((enumerated_sdf, path, dir, i+1))
    if isinstance(args.gpu_idx, int) or len(args.gpu_idx) == 1:
        queue.put("Done")
    else:
        for _ in range(len(args.gpu_idx)):
            queue.put("Done")


def optim_rank_wrapper(
    args: Auto3DOptions,
    queue: Queue[tuple[str, str, str, int] | str],
    logging_queue: Queue[LogRecord | None],
    gpu_idx: int,
) -> list[list[Chem.Mol]]:
    #prepare logging
    logger = logging.getLogger("auto3d")
    logger.addHandler(QueueHandler(logging_queue))
    logger.setLevel(logging.INFO)

    conformers = []
    while True:
        sdf_path_dir_job = queue.get()
        if sdf_path_dir_job == "Done":
            break
        enumerated_sdf, path, dir, job = sdf_path_dir_job
        logger.info(f"\n\nOptimizing on job{job}")
        meta = create_chunk_meta_names(path, dir)

        # Optimizing step
        opt_config = args.to_optimization_config()
        optimized_og = meta["optimized_og"]
        optimizing_engine = args.optimizing_engine
        if args.use_gpu:
            device = torch.device(f"cuda:{gpu_idx}")
        else:
            device = torch.device("cpu")
        optimizer = optimizing(enumerated_sdf, optimized_og,
                               optimizing_engine, device, opt_config)
        optimizer.run()

        # Ranking step
        output = meta["output"]
        duplicate_threshold = args.threshold
        k = args.k
        window = args.window
        rank_engine = ranking(optimized_og,
                              output, duplicate_threshold, k=k, window=window)
        conformers.append(rank_engine.run())

        # Housekeeping
        housekeeping_folder = meta["housekeeping_folder"]
        os.mkdir(housekeeping_folder)
        housekeeping(dir, housekeeping_folder, output)
        #Conpress verbose folder
        housekeeping_folder_gz = housekeeping_folder + ".tar.gz"
        with tarfile.open(housekeeping_folder_gz, "w:gz") as tar:
            tar.add(housekeeping_folder, arcname=Path(housekeeping_folder).name)
        shutil.rmtree(housekeeping_folder)
        if not args.verbose:
            try:  # Clusters does not support send2trash
                send2trash(housekeeping_folder_gz)
            except OSError:
                os.remove(housekeeping_folder_gz)
    return conformers

def options(
    path: str | None = None,
    k: int | bool = False,
    window: float | bool = False,
    verbose: bool = False,
    job_name: str = "",
    enumerate_tautomer: bool = False,
    tauto_engine: str = "rdkit",
    pKaNorm: bool = True,
    isomer_engine: str = "rdkit",
    enumerate_isomer: bool = True,
    mode_oe: str = "classic",
    mpi_np: int = 4,
    max_confs: int | None = None,
    use_gpu: bool = True,
    gpu_idx: int | list[int] = 0,
    capacity: int = 42,
    optimizing_engine: str = "AIMNET",
    patience: int = DEFAULT_PATIENCE,
    opt_steps: int = DEFAULT_OPT_STEPS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    threshold: float = DEFAULT_RMSD_THRESHOLD,
    memory: int | None = None,
    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS,
) -> Auto3DOptions:
    """Create configuration options for the Auto3D main function.

    Args:
        path: Input .smi or .sdf file path containing SMILES/molecules.
        k: Output top-k structures per SMILES. Set to int or False.
        window: Output structures within x kcal/mol of lowest energy. Set to float or False.
        verbose: When True, save all metadata while running.
        job_name: Folder name to save all metadata.
        enumerate_tautomer: When True, enumerate tautomers for the input.
        tauto_engine: Program for tautomer enumeration: 'rdkit' or 'oechem'.
        pKaNorm: Normalize ionization to pH ~7.4 (only with tauto_engine='oechem').
        isomer_engine: Program for 3D isomers: 'rdkit' or 'omega'.
        enumerate_isomer: When True, enumerate cis/trans and R/S isomers.
        mode_oe: Omega mode: 'classic', 'macrocycle', 'dense', 'pose', 'rocs', 'fast_rocs'.
        mpi_np: Number of CPU cores for isomer generation.
        max_confs: Maximum conformers per SMILES. None uses dynamic number.
        use_gpu: Use GPU when available.
        gpu_idx: GPU device index or list of indices.
        capacity: Number of SMILES handled per 1GB memory.
        optimizing_engine: Model: 'ANI2x', 'ANI2xt', 'AIMNET', or path to custom NNP.
        patience: Drop conformer if force doesn't decrease for this many steps.
        opt_steps: Maximum optimization steps per structure.
        convergence_threshold: Converge when max force is below this (eV/Å).
        threshold: RMSD threshold for duplicate removal (Å).
        memory: RAM size in GB. None for automatic detection.
        batchsize_atoms: Number of atoms per optimization batch per GB.

    Returns:
        Auto3DOptions configuration object.
    """
    return Auto3DOptions(
        path=path,
        k=k,
        window=window,
        verbose=verbose,
        job_name=job_name,
        enumerate_tautomer=enumerate_tautomer,
        tauto_engine=tauto_engine,
        pKaNorm=pKaNorm,
        isomer_engine=isomer_engine,
        enumerate_isomer=enumerate_isomer,
        mode_oe=mode_oe,
        mpi_np=mpi_np,
        max_confs=max_confs,
        use_gpu=use_gpu,
        gpu_idx=gpu_idx,
        capacity=capacity,
        optimizing_engine=optimizing_engine,
        patience=patience,
        opt_steps=opt_steps,
        convergence_threshold=convergence_threshold,
        threshold=threshold,
        memory=memory,
        batchsize_atoms=batchsize_atoms,
    )


def logger_process(queue: Queue[LogRecord | None], logging_path: str) -> None:
    """A child process for logging all information from other processes."""
    logger = logging.getLogger("auto3d")
    logger.addHandler(logging.FileHandler(logging_path))
    logger.setLevel(logging.INFO)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


def main(args: Auto3DOptions) -> str:
    """Run the Auto3D conformer generation pipeline.

    Args:
        args: Configuration options from the ``options()`` function
              or an ``Auto3DOptions`` instance.

    Returns:
        Path to the output SDF file containing generated conformers.

    Raises:
        SystemExit: If input validation fails or no structures converge.
    """
    # Configure logging based on verbose setting
    configure_logging(verbose=args.verbose)

    from Auto3D.workflow import WorkflowOrchestrator

    # Set multiprocessing start method (spawn is safer for CUDA)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set by another call

    orchestrator = WorkflowOrchestrator(args)
    return orchestrator.run()

def smiles2mols(smiles: list[str], args: Auto3DOptions) -> list[Chem.Mol]:
    """Find low-energy conformers for a list of SMILES.

    A convenient single-process function for small batches. For larger batches
    (>150 SMILES), use the ``main()`` function for better performance.

    Args:
        smiles: List of SMILES strings to generate conformers for.
        args: Configuration options from ``options()`` or ``Auto3DOptions``.

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
        isomer_engine = rd_isomer(path0, meta["smiles_enumerated"],
                                  meta["smiles_reduced"], meta["smiles_hashed"], 
                                  meta["enumerated_sdf"], tmpdirname,
                                  args.max_confs, 0.03,
                                  args.mpi_np, args.enumerate_isomer)
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
