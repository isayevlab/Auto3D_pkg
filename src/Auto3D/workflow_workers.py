"""Pipeline worker functions for the Auto3D conformer generation pipeline.

These functions run in separate processes spawned by ``WorkflowOrchestrator``
(see ``Auto3D.workflow``). They were moved out of ``Auto3D.auto3D`` to break the
auto3D<->workflow import cycle, so the dependency direction is now one-way:
``auto3D`` (API) -> ``workflow`` -> ``workflow_workers``.
"""
from __future__ import annotations

import logging
import os
import shutil
import tarfile
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rdkit import Chem
from send2trash import send2trash

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.isomers import IsomerEngineFactory
from Auto3D.processors import TautomerProcessor
from Auto3D.ranking import ranking
from Auto3D.utils import create_chunk_meta_names, housekeeping

if TYPE_CHECKING:
    from logging import LogRecord
    from multiprocessing import Queue

    from Auto3D.config import Auto3DOptions


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

    tautomer_processor = TautomerProcessor(args)

    # Number of optimizer processes that will consume from the queue.
    # Each optimizer blocks on queue.get() until it receives a "Done" sentinel,
    # so we must emit exactly one sentinel per optimizer in a `finally` block to
    # avoid deadlocking the optimizers when isomer generation fails partway.
    if isinstance(args.gpu_idx, int):
        n_optimizers = 1
    else:
        n_optimizers = len(args.gpu_idx)

    try:
        for i, path_dir in enumerate(chunk_info):
            logger.info(f"\n\nIsomer generation for job{i+1}")
            path, dir = path_dir
            meta = create_chunk_meta_names(path, dir)

            # Tautomer enumeration (if enabled)
            path = tautomer_processor.process(path, meta["output_taut"])

            smiles_enumerated = meta["smiles_enumerated"]
            smiles_reduced = meta["smiles_reduced"]
            smiles_hashed = meta["smiles_hashed"]
            enumerated_sdf = meta["enumerated_sdf"]
            max_confs = args.max_confs
            duplicate_threshold = args.threshold
            mpi_np = args.mpi_np
            enumerate_isomer = args.enumerate_isomer
            isomer_program = args.isomer_engine
            # Isomer enumeration step using factory
            engine = IsomerEngineFactory.create(
                engine_type=isomer_program,
                input_path=path,
                output_path=enumerated_sdf,
                input_format=args.input_format,
                smiles_enumerated=smiles_enumerated,
                smiles_reduced=smiles_reduced,
                smiles_hashed=smiles_hashed,
                job_dir=dir,
                max_confs=max_confs,
                threshold=duplicate_threshold,
                n_jobs=mpi_np,
                enumerate_isomers=enumerate_isomer,
                mode=args.mode_oe if isomer_program == 'omega' else 'classic',
            )
            engine.run()

            queue.put((enumerated_sdf, path, dir, i+1))
    except Exception:
        logger.exception("Isomer generation failed; signaling optimizers to stop.")
        raise
    finally:
        # Always wake every optimizer, even on failure, so none blocks forever.
        for _ in range(n_optimizers):
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
        # Isolate each chunk: a single failing chunk (a molecule the optimizer
        # chokes on, a CUDA OOM, an isomer step that produced nothing, an mkdir
        # collision) must not kill this worker and silently drop every chunk
        # still queued behind it. Log it and move on to the next chunk.
        try:
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

            # optimizing.run() returns early without writing optimized_og when
            # the isomer step yielded an empty/missing SDF for this chunk. Skip
            # ranking rather than letting RDKit raise on a nonexistent path; the
            # chunk simply contributes no conformers.
            if not os.path.exists(optimized_og):
                logger.warning(
                    f"job{job}: no optimized structures were produced; "
                    "skipping ranking for this chunk."
                )
                continue

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
        except Exception:
            logger.exception(
                f"job{job} failed during optimization/ranking; "
                "skipping this chunk and continuing with the rest."
            )
            continue
    return conformers

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
