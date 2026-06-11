# src/Auto3D/isomers/parallel_embed.py
"""Parallel conformer embedding using multiprocessing."""
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.constants import CONFORMER_RANDOM_SEED
from Auto3D.utils import relieve_clash
from Auto3D.utils.chemistry import calculate_conformer_count
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


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
        randomSeed=CONFORMER_RANDOM_SEED,
        numThreads=np_threads,
        pruneRmsThresh=threshold,
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

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _embed_single, smi, name, n_conformers, threshold, np_threads
            ): (smi, name)
            for smi, name in smiles_names
        }

        # Iterate in submission order (not as_completed) so the emitted molecule
        # order is deterministic and matches the serial path; all futures are
        # already running concurrently, so this costs no parallelism.
        for future in futures:
            try:
                yield from future.result()
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
                smi, name = futures[future]
                logger.warning(f"Failed to embed {name}: {type(e).__name__}: {e}")
