# src/Auto3D/isomers/parallel_embed.py
"""Parallel conformer embedding using multiprocessing."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from Auto3D.utils import min_pairwise_distance


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
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))

    if n_conformers is None:
        # Dynamic formula based on: https://doi.org/10.1021/acs.jctc.0c01213
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_heavy = len([a for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
        n_conformers = min(
            max(num_heavy, int(2 * 8.481 * (num_rotatable ** 1.642))),
            1000
        )

    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=n_conformers,
        randomSeed=42,
        numThreads=np_threads,
        pruneRmsThresh=threshold,
    )

    results = []
    for i in range(mol.GetNumConformers()):
        positions = mol.GetConformer(i).GetPositions()

        # Check for atom clashes (distance < 0.9 Angstrom)
        if min_pairwise_distance(positions) < 0.9:
            # Try to fix with MMFF optimization
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
            positions = mol.GetConformer(i).GetPositions()

        # Only keep conformers with valid distances
        if min_pairwise_distance(positions) > 0.9:
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

        for future in as_completed(futures):
            try:
                results = future.result()
                for mol, conf_idx, conf_id in results:
                    yield mol, conf_idx, conf_id
            except Exception as e:
                smi, name = futures[future]
                print(f"Failed to embed {name}: {e}")
