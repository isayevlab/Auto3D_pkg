"""Pipeline worker functions for the Auto3D conformer generation pipeline.

These functions run in separate processes spawned by ``WorkflowOrchestrator``
(see ``Auto3D.workflow``). They were moved out of ``Auto3D.auto3D`` to break the
auto3D<->workflow import cycle, so the dependency direction is now one-way:
``auto3D`` (API) -> ``workflow`` -> ``workflow_workers``.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
import tarfile
from logging.handlers import QueueHandler
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rdkit import Chem
from send2trash import send2trash

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.config import optimizer_worker_indices
from Auto3D.isomers import IsomerEngineFactory
from Auto3D.job_layout import create_chunk_meta_names, housekeeping
from Auto3D.model_factory import create_model
from Auto3D.processors import TautomerProcessor
from Auto3D.ranking import ranking

if TYPE_CHECKING:
    from logging import LogRecord
    from multiprocessing import Queue

    from Auto3D.config import Auto3DOptions

# Both logger trees a module warning must reach the run log through.
#
# Auto3D.utils.logging_config.get_logger(__name__) produces loggers named
# "Auto3D.*" (matching each module's real __name__, e.g. "Auto3D.ASE.thermo").
# Several call sites instead log through logging.getLogger("auto3d")
# directly -- lowercase -- to work around the fact that "Auto3D.*" is a
# different, case-distinct tree with no ancestor relationship to "auto3d"
# (Auto3D.workflow's self.logger, Auto3D.clash_relief's module logger, the
# stereochemistry-change warning in Auto3D.batch_opt.batchopt). Attaching a
# QueueHandler onto BOTH trees here -- writing to the very same queue -- lets
# get_logger(__name__) warnings reach the run log too, without touching any
# of the call sites above that already rely on "auto3d" working: "auto3d" and
# "Auto3D" are unrelated siblings under the root logger (dotted-name lookup
# is exact-prefix and case-sensitive), so a single log call is only ever
# routed through one of them and nothing is ever delivered twice.
_RUN_LOG_LOGGER_NAMES = ("auto3d", "Auto3D")


def _worker_stdout_to_stderr() -> contextlib.AbstractContextManager[object]:
    """Keep a worker process off the run's stdout.

    These workers are *spawned* (``auto3D.main`` forces the spawn start
    method), so each one is a fresh interpreter with its own ``sys.stdout``
    on the inherited fd 1 -- the parent's ``Auto3D.cli.console.reserve_stdout``
    redirection is a Python-level object swap and does not follow them. The
    first thing an optimizer worker does with a model is build it, which
    imports ``aimnet`` -> ``warp`` and prints a device banner to stdout: one
    per worker, straight into the middle of the ``--json`` document the parent
    is writing on the very same fd.

    Nothing here is meant for stdout in the first place -- every worker
    message goes through the logging queue to the run log and the parent's
    stderr handler -- so anything that does write to stdout is by definition
    third-party, and stderr is where it belongs. Redirected rather than
    discarded, so a library's genuine failure message is still readable.
    """
    return contextlib.redirect_stdout(sys.stderr)


def _attach_run_log_handlers(
    logging_queue: Queue[LogRecord | None],
) -> list[tuple[str, logging.Handler]]:
    """Attach a run-log QueueHandler onto every tree it must reach.

    Idempotent per ``logging_queue``: if a tree already has a ``QueueHandler``
    targeting this exact queue (e.g. a second in-process call to
    ``isomer_wrapper``/``optim_rank_wrapper`` with the same queue, as tests --
    not production -- sometimes do), that tree is left untouched instead of
    growing a second handler that would deliver every subsequent record
    twice. Production is unaffected: each run spawns a fresh worker process
    with a fresh queue, so this only ever attaches once there.

    Returns the ``(logger_name, handler)`` pairs newly added (never the ones
    already present), so a caller (chiefly tests) can remove them again;
    production callers can ignore the return value since these handlers live
    for the worker process's lifetime.
    """
    added: list[tuple[str, logging.Handler]] = []
    for logger_name in _RUN_LOG_LOGGER_NAMES:
        tree_logger = logging.getLogger(logger_name)
        already_attached = any(
            isinstance(h, QueueHandler) and h.queue is logging_queue
            for h in tree_logger.handlers
        )
        if already_attached:
            continue
        handler = QueueHandler(logging_queue)
        tree_logger.addHandler(handler)
        tree_logger.setLevel(logging.INFO)
        added.append((logger_name, handler))
    return added


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
    with _worker_stdout_to_stderr():
        #prepare logging
        logger = logging.getLogger("auto3d")
        _attach_run_log_handlers(logging_queue)

        tautomer_processor = TautomerProcessor(args)

        # Number of optimizer processes that will consume from the queue.
        # Each optimizer blocks on queue.get() until it receives a "Done" sentinel,
        # so we must emit exactly one sentinel per optimizer in a `finally` block to
        # avoid deadlocking the optimizers when isomer generation fails partway.
        # Use the same rule as the spawn site (a CPU run with a list of gpu_idx runs
        # a single optimizer, not one per index) so the counts cannot drift.
        n_optimizers = len(optimizer_worker_indices(args.use_gpu, args.gpu_idx))

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
                    use_parallel_embedding=args.use_parallel_embedding,
                    parallel_workers=args.parallel_workers,
                    parallel_embedding_threshold=args.parallel_embedding_threshold,
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
    progress_queue: Queue[dict] | None = None,
) -> list[list[Chem.Mol]]:
    with _worker_stdout_to_stderr():
        #prepare logging
        logger = logging.getLogger("auto3d")
        _attach_run_log_handlers(logging_queue)

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
                # When a progress queue is supplied (interactive `auto3d run`), tag
                # each event with this chunk's job id and forward it to the main
                # process for the live display. Guarded so a full/closed queue can
                # never break the optimization.
                progress_cb = None
                if progress_queue is not None:
                    def progress_cb(event, _q=progress_queue, _job=job):
                        try:
                            _q.put({**event, "job": _job})
                        except Exception:
                            pass
                # HARD CONSTRAINT: the adapter is built HERE, inside the spawned
                # worker, and must stay here. `optimizing` used to construct it
                # itself; hoisting construction one frame out (to this function)
                # keeps it in the same process, but hoisting it any further -- to
                # `workflow.py`, which drives the pool, where these duplicated
                # `create_model` calls would look like an obvious cleanup -- pushes
                # a device-resident nn.Module, and for AIMNET a live
                # AIMNet2Calculator, across the `spawn` boundary. That is either an
                # unpicklable-object failure or CUDA re-initialization in the
                # parent, and nothing in the signature says so.
                adapter = create_model(optimizing_engine, device)
                optimizer = optimizing(enumerated_sdf, optimized_og,
                                       adapter=adapter, device=device,
                                       config=opt_config,
                                       progress_cb=progress_cb)
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
    """A child process for logging all information from other processes.

    Everything the workers log arrives here, and this is the only place that
    decides where it goes. WARNING and above additionally go to stderr; INFO and
    DEBUG stay in the run log, which is where the step-by-step narrative belongs.

    Why the stderr handler exists: a worker's ``logger`` reaches this process
    through a ``QueueHandler`` and nothing else, so with a file as the only
    handler here, a chunk that failed wrote its traceback to
    ``<job_dir>/Auto3D.log`` and told the user nothing. The loss itself was
    visible -- reconciliation names the missing molecules and the run exits 6 --
    but the *cause* was not, so a systematic bug that failed every chunk
    identically was indistinguishable from a batch of hard molecules, and read
    as "N molecules produced no conformer". The same silence covered the "no
    optimized structures were produced" warning and every other worker warning,
    which is why this is fixed at the collector rather than at one call site.

    Stderr, not stdout: ``--json`` promises stdout carries only the document.
    Interactive runs render a live panel on stderr too, so a warning mid-run can
    tear that panel -- an acceptable trade for a diagnosis, and the reason INFO
    is kept out of it.
    """
    logger = logging.getLogger("auto3d")
    logger.addHandler(logging.FileHandler(logging_path))
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)
