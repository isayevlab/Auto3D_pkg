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

import multiprocessing
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
from Auto3D.foundation.constants import CONFORMER_RANDOM_SEED
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.molprops import calculate_conformer_count

logger = get_logger(__name__)


def embed_params(*, n_threads: int, prune_rms_thresh: float) -> Any:
    """The ETKDG settings every Auto3D embedding uses, in one place.

    ``EmbedMultipleConfs``'s keyword form cannot express two of these:
    ``onlyHeavyAtomsForRMS`` and ``useSymmetryForPruning`` exist only on the
    parameters object. Left to their defaults, the size of the pool that
    ``pruneRmsThresh`` leaves behind depends on **which RDKit is installed** --
    both default True on 2025.09 but have not always, and ``pyproject.toml``
    floors at ``rdkit>=2022.9.5`` with no upper bound. Stating them makes the
    conformer pool a property of this code, in the same way
    ``CONFORMER_RANDOM_SEED`` does.

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
    return params


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
    """
    # Validate SMILES first to avoid unpicklable Boost.Python errors
    mol_noh = Chem.MolFromSmiles(smi)
    if mol_noh is None:
        # Same message the serial path emits (isomer_engine._run_serial_embedding).
        # This branch returned [] in silence, so a molecule dropped for an
        # unparseable SMILES was reported by the parallel path and not by the
        # serial one -- a switch documented as a performance option decided how
        # much the user was told. The parent also warns on an empty result, which
        # is the guaranteed signal; this one adds the reason.
        logger.warning(f"Skipping molecule {name!r}: failed to parse {smi!r}")
        return []
    mol = Chem.AddHs(mol_noh)

    if n_conformers is None:
        # Compute the conformer budget on the H-complete (AddHs) mol so the
        # parallel path agrees with the serial/SDF paths on the RICHER with-H
        # count: CalcNumRotatableBonds only counts O-H / N-H torsions when
        # hydrogens are explicit (e.g. glycerol 238 vs 52 conformers).
        n_conformers = calculate_conformer_count(mol)

    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=n_conformers,
        params=embed_params(n_threads=np_threads, prune_rms_thresh=threshold),
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

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=EMBEDDING_MP_CONTEXT) as executor:
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
            except BrokenProcessPool:
                # A worker died (e.g. OOM-killed): the pool is broken and EVERY
                # remaining future will also raise this. Surface it loudly --
                # the broad except below would otherwise swallow it per-future
                # and silently drop the whole tail of the batch as warnings.
                raise
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
