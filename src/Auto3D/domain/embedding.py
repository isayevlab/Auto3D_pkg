"""Parallel conformer embedding using multiprocessing.

Lives at the top level rather than under ``Auto3D.engines.isomers`` because
``isomer_engine`` is its only caller and ``isomers`` is the package that
*wraps* ``isomer_engine``: with this module inside ``isomers``, the two
packages imported each other (``isomers.factory``/the adapters reached into
``isomer_engine``, and ``isomer_engine._run_parallel_embedding`` reached back
into ``isomers.parallel_embed``), a cycle that only stayed latent because
every edge of it was a function-scope import. Moving this module out is what
removes the cycle rather than deferring it.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

#: The context :func:`embed_conformers_parallel`'s pool starts workers from.
#:
#: Explicit, rather than the default context, for two reasons -- neither of them
#: "this code needs spawn". Nothing in this module touches CUDA; it is RDKit
#: work, and fork would serve it fine in isolation.
#:
#: 1. A default-context pool *locks the interpreter's global start method* to the
#:    platform default the first time it is created. This pool runs during the
#:    isomer stage, before the optimization workers -- which do run PyTorch, and
#:    which get a broken CUDA context if forked. That ordering is why ``main()``
#:    had to call ``set_start_method("spawn", force=True)`` rather than the
#:    best-effort form. Taking an explicit context means this pool no longer
#:    touches the global method at all, and nothing downstream has to fight it.
#: 2. Under ``main()`` this pool has always in fact run under spawn, because that
#:    global force preceded it. Leaving it on the default context would have
#:    quietly switched it to fork in a process where torch may already hold
#:    threads and a CUDA context -- a behavior change disguised as a cleanup.
#:
#: ``get_context("spawn")`` returns a context object; unlike ``get_context()``
#: with no argument, it does not read or set the global start method.
EMBEDDING_MP_CONTEXT = multiprocessing.get_context("spawn")
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from Auto3D.domain.clash_relief import relieve_clash
from Auto3D.foundation.constants import (
    CONFORMER_RANDOM_SEED,
    EMBED_TIMEOUT_S,
    PARALLEL_EMBED_MAX_WORKERS,
)
from Auto3D.foundation.process_lifecycle import _exit_when_parent_dies
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.molprops import calculate_conformer_count, has_dummy_atoms

logger = get_logger(__name__)


class SpeciesSkipped(Exception):  # noqa: N818 - a skip signal, not a failure
    """A species the embedding worker refused before embedding anything.

    No ``Error`` suffix, unlike ``Auto3D.foundation.exceptions``'s members: this
    is not a failure the run has to recover from but the worker's way of saying
    "not this one" across a process boundary. The batch continues, the species is
    reported, and nothing upstream treats it as an error -- naming it
    ``SpeciesSkippedError`` would describe the mechanism and misdescribe the
    meaning.

    Carries the reason back to the parent instead of logging it in the worker.
    Under ``spawn`` the pool children have no run-log handler of their own
    (``workflow_workers._attach_run_log_handlers`` wires up the optimizer
    workers, not this pool), so a warning emitted in a worker fell through to
    ``logging.lastResort``: raw on stderr, carrying no level or logger prefix,
    ungoverned by Auto3D's logging configuration -- and absent from the run log,
    which recorded only that the species had vanished. The parent is already
    wired into that configuration, so it logs the single line.

    One argument, always a plain string, and no custom ``__init__``: the
    instance is pickled across the pool boundary, and a signature that differs
    from ``Exception``'s does not survive the round trip.
    """


def _embedding_worker_init() -> None:
    """Prepare one pool worker. Passed as the executor's ``initializer``.

    Two jobs, both of which only make sense inside a worker:

    * ``_exit_when_parent_dies`` -- the same arming every other process Auto3D
      starts gets (the isomer worker, the optimizers, the logger process, the
      Manager servers). Without it this pool was the one that outlived a
      parent-only signal: under ``spawn`` each child holds a dup of the call
      queue's *write* end as well as its read end, so the parent's death never
      produces EOF, and SIGTERM/SIGKILL to the parent runs none of the ``with``
      block's shutdown sentinels either. An idle worker then blocks on
      ``get()`` forever and a busy one keeps burning a core on ETKDG (P-C2).
      A no-op when there is no parent process, and every failure path inside it
      is caught and logged at DEBUG, so it cannot break the pool.
    * ``CoordsAsDouble`` -- RDKit pickles conformer coordinates as float32
      unless asked otherwise, and the worker's result travels back by pickle. At
      SDF write precision that flipped the last digit of one coordinate of one
      species in a 28-species comparison, which made toggling
      ``--parallel-embedding`` change the bytes of the output.

    Set here rather than at module import, deliberately: the pickle properties
    are process-global, this is the process where the pickling happens, and
    every spawned worker runs the initializer. Importing
    ``Auto3D.domain.embedding`` must not reconfigure RDKit for a caller that
    never starts a pool -- the same stance the module takes on
    ``set_start_method`` (see ``EMBEDDING_MP_CONTEXT``).
    """
    _exit_when_parent_dies()
    Chem.SetDefaultPickleProperties(
        Chem.GetDefaultPickleProperties() | Chem.PropertyPickleOptions.CoordsAsDouble
    )


def available_cpu_count() -> int:
    """CPUs this process may run on: the affinity mask where the platform has one, else ``os.cpu_count()``.

    ``os.cpu_count()`` is the machine's count; a 2-CPU cgroup on a 128-core host
    reports 128 and ``resolve_embedding_workers`` would start 32 workers for two
    cores. The affinity mask is what the scheduler will actually give this
    process (taskset, slurm, Kubernetes with cpuset). A cgroup CPU quota
    (``cpu.max``) without a cpuset is still invisible here; pass
    ``parallel_workers`` explicitly in that case.
    """
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            return max(1, len(getter(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def resolve_embedding_workers(
    requested: int | None, n_species: int, *, threads_per_worker: int = 1
) -> int:
    """Worker-process count for parallel conformer embedding.

    ``None`` -- the default -- resolves to
    ``min(cores // threads per worker, species, PARALLEL_EMBED_MAX_WORKERS)``.
    A fixed default of 4 left 124 of 128 cores idle on the 2026-09-21 bench
    (P-C3), and no class default can do better, because the useful number
    depends on the box, on how many species this particular run enumerated,
    and on how many threads each worker will use. So the resolution happens
    here, called at dispatch by whoever is about to start the pool.

    The division is what keeps the box from being oversubscribed: each worker
    hands ``threads_per_worker`` to ``EmbedMultipleConfs`` (the isomer engine
    passes its ``np``/``mpi_np``, default 4), so one worker per core would put
    ``cores x threads`` runnable threads on ``cores`` cores. ``threads_per_worker``
    is floored at 1 before dividing: the engine's direct callers bypass
    ``Auto3DOptions``'s ``mpi_np >= 1`` bound, and a 0 there would be a
    ``ZeroDivisionError``.

    An explicit ``requested`` is obeyed as given -- cap, cores and thread count
    all bypassed, since a caller who names a number has a reason -- and only
    floored at 1, because a pool cannot be started with zero workers.
    ``Auto3DOptions`` already refuses a ``parallel_workers`` below 1; the floor
    is here for the engine's direct callers, which go through no such
    validation.

    The floor can still overshoot the box: one worker handing RDKit
    ``threads_per_worker`` threads is ``threads_per_worker`` runnable threads
    wherever there are fewer cores than that -- 2 cores with ``mpi_np=4``
    measured 2x oversubscribed. Closing it would mean overriding the caller's
    own ``mpi_np``, which is their choice to make, so it is named here instead.

    Args:
        requested: Explicit worker count, or None to scale to the machine.
        n_species: How many species this dispatch has to embed.
        threads_per_worker: RDKit threads each worker will use for embedding.

    Returns:
        A worker count of at least 1.
    """
    if requested is not None:
        return max(1, requested)
    # Called unqualified so a test (and a caller measuring on a different
    # machine shape) can substitute `available_cpu_count` via
    # `monkeypatch.setattr(embedding_module, "available_cpu_count", ...)`.
    usable_cores = max(1, available_cpu_count() // max(1, threads_per_worker))
    return max(1, min(usable_cores, n_species, PARALLEL_EMBED_MAX_WORKERS))


def embed_params(
    *, n_threads: int, prune_rms_thresh: float, timeout_s: int = EMBED_TIMEOUT_S
) -> Any:
    """The ETKDG settings every Auto3D embedding uses, in one place.

    ``EmbedMultipleConfs``'s keyword form cannot express two of these:
    ``onlyHeavyAtomsForRMS`` and ``useSymmetryForPruning`` exist only on the
    parameters object. Left to their defaults, the size of the pool that
    ``pruneRmsThresh`` leaves behind depends on **which RDKit is installed** --
    both default True on 2025.09 but have not always, and ``pyproject.toml``
    floors at ``rdkit>=2022.9.5`` with no upper bound. Stating them makes the
    conformer pool a property of this code, in the same way
    ``CONFORMER_RANDOM_SEED`` does.

    With ``useSymmetryForPruning=True`` RDKit prunes on heavy atoms whatever
    ``onlyHeavyAtomsForRMS`` says (measured: glycerol keeps 9 conformers with
    either RMS flag, 206 with symmetry pruning off and all-atom RMS), so the
    pool holds one hydroxyl / amine orientation per heavy-atom skeleton; the
    optimizer, not the embedding, decides where those hydrogens end up.
    ``onlyHeavyAtomsForRMS`` is still stated so the intent survives a future
    ``useSymmetryForPruning=False``.

    ``ETKDGv3()`` rather than a bare ``EmbedParameters()``: it is exactly the
    parameterization the keyword form applied. Verified field by field --
    ``useExpTorsionAnglePrefs``, ``useBasicKnowledge``, ``useMacrocycleTorsions``
    and ``useMacrocycle14config`` are the four a bare object gets wrong, and
    ``ETversion`` is 2 in both -- so this switch changes no geometry. A bare
    object would silently disable the torsion knowledge ETKDG is named for.
    """
    # Typed `Any` rather than `EmbedParameters`, and deliberately: RDKit's
    # stubs declare this class's attributes as `EmbedParameters` rather than as
    # their value types, so every assignment below is a false positive and the
    # honest return type is unknowable from the stubs. Confining that to this
    # one function is better than scattering per-line ignores across four call
    # sites; the returned object is only ever handed straight to
    # `EmbedMultipleConfs`.
    params: Any = rdDistGeom.ETKDGv3()
    params.randomSeed = CONFORMER_RANDOM_SEED
    params.numThreads = n_threads
    params.pruneRmsThresh = prune_rms_thresh
    params.onlyHeavyAtomsForRMS = True
    params.useSymmetryForPruning = True
    if hasattr(params, "timeout"):
        params.timeout = timeout_s
    else:
        logger.warning(
            "This RDKit has no EmbedParameters.timeout; a species that cannot embed may run long."
        )
    return params


def embed_with_retry(
    mol: Chem.Mol,
    *,
    n_conformers: int,
    n_threads: int,
    prune_rms_thresh: float,
    timeout_s: int = EMBED_TIMEOUT_S,
) -> int:
    """Embed up to ``n_conformers`` conformers; retry once with random
    coordinates if none embed, within the time the first attempt left over.
    Returns the number embedded.

    ETKDG's default initial coordinates can fail on strained systems that
    random initial coordinates still solve, which is what the retry recovers
    (N-m2). That is a statement about the *starting geometry*, not about time:
    an attempt that produced nothing because it burned the whole cap says the
    species is slow, and random coordinates would get no more time than the
    first attempt had.

    So the retry runs on the remainder of the budget -- ``timeout_s`` minus
    attempt 1's measured wall clock -- and is skipped outright when under a
    second of it is left. **Both attempts together are bounded by
    ``timeout_s``** (plus up to a second, since RDKit's ``timeout`` is whole
    seconds and the remainder is rounded up). Given a full cap each instead,
    the per-species worst case was ``2 * timeout_s``: 120 s at the shipped
    default, against the ~67 s one impossible stereoisomer measured serially on
    the 2026-09-21 bench set -- the number the cap exists to bound (P-C3), and
    spent twice on exactly the species that triggers the retry.
    """
    params = embed_params(
        n_threads=n_threads, prune_rms_thresh=prune_rms_thresh, timeout_s=timeout_s
    )
    started = time.monotonic()
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(ids) == 0:
        remaining = timeout_s - (time.monotonic() - started)
        if remaining < 1:
            logger.debug(
                "First ETKDG attempt embedded nothing and used its whole %d s budget; "
                "not retrying, since random initial coordinates would have no more "
                "time than the first attempt had.",
                timeout_s,
            )
            return 0
        logger.debug(
            "First ETKDG attempt embedded nothing; retrying with random initial "
            "coordinates and the %d s the first attempt left.",
            math.ceil(remaining),
        )
        # Guarded like the assignment in `embed_params`, and for the same RDKit:
        # `params.timeout = ...` on a Boost.Python object without that attribute
        # raises rather than creating it. On such an RDKit neither attempt was
        # capped in the first place, so there is no budget to hand on.
        if hasattr(params, "timeout"):
            params.timeout = math.ceil(remaining)
        params.useRandomCoords = True
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    return len(ids)


def _embed_single(
    smi: str,
    name: str,
    n_conformers: int | None,
    threshold: float,
    np_threads: int,
) -> list[tuple[Chem.Mol, int, str]]:
    """Embed conformers for a single SMILES. Worker function.

    This function generates multiple 3D conformers for a SMILES string,
    filters out invalid conformers (those with atom clashes), and returns
    the valid conformers with their indices.

    Args:
        smi: SMILES string of the molecule.
        name: Name/ID of the molecule.
        n_conformers: Maximum number of conformers to generate.
            If None, uses a dynamic formula based on molecular properties.
        threshold: RMSD threshold for duplicate removal during embedding.
        np_threads: Number of threads for RDKit conformer generation.

    Returns:
        List of (mol, conf_idx, conf_id) tuples where:
            - mol: RDKit Mol object with conformers
            - conf_idx: Index of the conformer in the molecule
            - conf_id: Unique identifier string (name_idx format)

    Raises:
        SpeciesSkipped: The SMILES cannot be parsed, or carries a dummy atom.
            Raised rather than logged here: see ``SpeciesSkipped`` for why a
            warning from a pool worker is the wrong place for the reason.
    """
    # Validate SMILES first to avoid unpicklable Boost.Python errors
    mol_noh = Chem.MolFromSmiles(smi)
    if mol_noh is None:
        # Same reason the serial path reports (isomer_engine._run_serial_embedding).
        # This branch returned [] in silence, so a molecule dropped for an
        # unparseable SMILES was reported by the parallel path and not by the
        # serial one -- a switch documented as a performance option decided how
        # much the user was told.
        raise SpeciesSkipped(f"failed to parse {smi!r}")
    if has_dummy_atoms(mol_noh):
        # N-M3: an R-group placeholder is not a species. Clash relief happens
        # to reject every conformer of such a molecule today (neither MMFF nor
        # UFF can type atom `*`), so it already disappeared -- but silently and
        # for an unrelated reason. Named and skipped here instead, before any
        # embedding work, so the same rule holds whatever the force fields do.
        raise SpeciesSkipped("it contains a dummy atom (atomic number 0)")
    mol = Chem.AddHs(mol_noh)

    if n_conformers is None:
        # calculate_conformer_count counts rotatable bonds on the heavy-atom
        # graph (Chem.RemoveAllHs internally), so it returns the same budget
        # whether it is handed this H-complete (AddHs) mol or the serial/SDF
        # paths' own representation -- the parallel path agrees with them
        # regardless of which hydrogen state each one happens to pass in.
        n_conformers = calculate_conformer_count(mol)

    embed_with_retry(
        mol, n_conformers=n_conformers, n_threads=np_threads, prune_rms_thresh=threshold
    )

    results = []
    for i in range(mol.GetNumConformers()):
        # Relieve atom clashes (MMFF, with UFF fallback for elements lacking
        # MMFF params) and keep only conformers that end up clash-free.
        if relieve_clash(mol, i):
            conf_id = f"{name}_{i}"
            results.append((mol, i, conf_id))

    return results


def embed_conformers_parallel(
    smiles_names: list[tuple[str, str]],
    n_conformers: int | None = None,
    threshold: float = 0.3,
    np_threads: int = 1,
    n_workers: int = 4,
) -> Iterator[tuple[Chem.Mol, int, str]]:
    """Embed conformers for multiple SMILES in parallel.

    Uses ProcessPoolExecutor for parallel execution across multiple molecules.
    Each molecule is processed independently in a separate worker process.

    Args:
        smiles_names: List of (smiles, name) tuples to process.
        n_conformers: Maximum conformers per molecule. None for dynamic calculation.
        threshold: RMSD threshold for duplicate removal during embedding.
        np_threads: Number of threads per worker for RDKit operations.
        n_workers: Number of parallel worker processes.

    Yields:
        (mol, conf_idx, conf_id) tuples for each valid conformer where:
            - mol: RDKit Mol object with conformers
            - conf_idx: Index of the conformer in the molecule
            - conf_id: Unique identifier string (name_idx format)

    Example:
        >>> smiles_names = [("C", "methane"), ("CC", "ethane")]
        >>> for mol, idx, name in embed_conformers_parallel(smiles_names):
        ...     print(f"{name}: {mol.GetNumAtoms()} atoms")
    """
    if not smiles_names:
        return

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=EMBEDDING_MP_CONTEXT,
        initializer=_embedding_worker_init,
    ) as executor:
        futures = {
            executor.submit(_embed_single, smi, name, n_conformers, threshold, np_threads): (
                smi,
                name,
            )
            for smi, name in smiles_names
        }

        # Iterate in submission order (not as_completed) so the emitted molecule
        # order is deterministic and matches the serial path; all futures are
        # already running concurrently, so this costs no parallelism.
        for future in futures:
            smi, name = futures[future]
            try:
                conformers = future.result()
            except BrokenProcessPool as exc:
                # A worker died (e.g. OOM-killed): the pool is broken and EVERY
                # remaining future will also raise this. Surface it loudly --
                # the broad except below would otherwise swallow it per-future
                # and silently drop the whole tail of the batch as warnings.
                #
                # The other common cause is not a dead worker at all: under the
                # spawn context each child re-imports the caller's `__main__`,
                # and an unguarded script raises there before embedding anything,
                # breaking the pool. The user's only clue was a child-process
                # traceback about freeze_support, so attach the two ways out.
                # A note rather than a new exception type: an OOM kill must keep
                # surfacing as the BrokenProcessPool everything upstream expects.
                exc.add_note(
                    "Parallel embedding spawns worker processes that re-import "
                    "the calling script. Guard the script's entry point with "
                    'if __name__ == "__main__":, or embed serially '
                    "(use_parallel_embedding=False for main(); smiles2mols is "
                    "serial unless called with parallel_embedding=True)."
                )
                raise
            except SpeciesSkipped as skipped:
                # The worker refused this species and said why. One warning, from
                # here -- the process that owns the run log -- and no second line:
                # the `continue` is also what keeps the generic "produced no
                # conformers" branch below from firing for a species whose reason
                # is already on the record.
                logger.warning("Skipping molecule %r: %s", name, skipped)
                continue
            except Exception as e:
                # Per-molecule boundary: a single molecule's failure (including
                # RDKit's Boost.Python.ArgumentError, which is a TypeError and so
                # escaped the previous narrow except) must not abort the whole
                # batch and silently drop every remaining molecule.
                logger.warning(f"Failed to embed {name}: {type(e).__name__}: {e}")
                continue
            if not conformers:
                # The counterpart to the serial path's `n_written == 0` warning,
                # which this path had no equivalent of. A species that embeds
                # nothing -- unparseable SMILES, or every conformer rejected by
                # clash relief -- is absent from the output and never reaches
                # ranking, so not even "No structure converged" appears for it.
                # Warned here, in the parent, because a message from a
                # ProcessPoolExecutor worker depends on that child's logging
                # configuration, while this one does not.
                logger.warning(
                    f"{name!r} produced no conformers; this species is absent from the output."
                )
                continue
            yield from conformers
