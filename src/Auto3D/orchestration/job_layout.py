#!/usr/bin/env python
"""Where a run puts its files, and what happens to them afterwards.

Pipeline policy, not generic file I/O: :func:`create_chunk_meta_names` is the
single place that decides every intermediate file's name inside a chunk's job
directory, and :func:`housekeeping` is what sweeps those same files into the
folder the caller then tars and deletes. The two are a matched pair -- the
sweep is only safe because the names it collects were all minted inside a
directory Auto3D created -- so they live together, above ``Auto3D.foundation.utils``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_chunk_meta_names(path: str, dir: str) -> dict[str, str]:
    """Create output file names based on chunk input path and directory.

    Generates a dictionary of standardized file paths for all intermediate
    and output files used in the Auto3D workflow.

    Args:
        path: Chunk input .smi file path.
        dir: Chunk job folder path.

    Returns:
        Dictionary mapping meta names to file paths with the following keys:
        - output: Final 3D structure output file
        - optimized_og: Original optimized structures
        - output_taut: Tautomer SMILES output
        - smiles_enumerated: Enumerated SMILES file
        - smiles_reduced: Reduced enumerated SMILES file
        - smiles_hashed: Hashed enumerated SMILES file
        - enumerated_sdf: Enumerated SDF file
        - sorted_sdf: Sorted SDF file
        - housekeeping_folder: Verbose output folder
        - path: Original input path
        - dir: Job directory

    Example:
        >>> meta = create_chunk_meta_names("chunk1.smi", "/tmp/job")
        >>> meta["output"]
        '/tmp/job/chunk1_3d.sdf'
    """
    dct: dict[str, str] = {}
    dir_path = Path(dir)
    stem = Path(path).stem

    output = str(dir_path / f"{stem}_3d.sdf")
    optimized_og = str(dir_path / f"{stem}_3d0.sdf")
    output_taut = str(dir_path / "smi_taut.smi")
    smiles_enumerated = str(dir_path / "smiles_enumerated.smi")
    smiles_reduced = str(dir_path / "smiles_enumerated_reduced.smi")
    smiles_hashed = str(dir_path / "smiles_enumerated_hashed.smi")
    enumerated_sdf = str(dir_path / "smiles_enumerated.sdf")
    sorted_sdf = str(dir_path / "enumerated_sorted.sdf")
    housekeeping_folder = str(dir_path / "verbose")

    dct["output"] = output
    dct["optimized_og"] = optimized_og
    dct["output_taut"] = output_taut
    dct["smiles_enumerated"] = smiles_enumerated
    dct["smiles_reduced"] = smiles_reduced
    dct["smiles_hashed"] = smiles_hashed
    dct["enumerated_sdf"] = enumerated_sdf
    dct["sorted_sdf"] = sorted_sdf
    dct["housekeeping_folder"] = housekeeping_folder
    dct["path"] = path
    dct["dir"] = dir
    return dct


def housekeeping(job_name: str, folder: str, optimized_structures: str) -> None:
    """Move this job directory's metadata files into a folder.

    Moves every entry of ``job_name`` except the optimized structures file
    into ``folder``. **Nothing outside ``job_name`` is ever touched**, which
    is a correctness requirement and not a style preference: the caller
    (``workflow_workers.optim_rank_wrapper``) tars ``folder``, ``rmtree``s it,
    and -- under the default ``verbose=False`` -- sends the tarball to trash
    or, when that is unavailable (the cluster path), plainly ``os.remove``s
    it. Whatever ends up in ``folder`` is therefore *deleted*.

    This function used to additionally sweep ``oeomega_*`` and ``flipper_*``
    out of the **process working directory**, which for an ordinary
    ``cd ~/project && auto3d run mols.smi --k 1`` is the user's own directory:
    a file named e.g. ``~/project/oeomega_settings.txt`` was moved into the
    run's ``verbose`` folder and then destroyed with it, unrecoverably on the
    ``os.remove`` path. That loop ran on *every* run, not only OpenEye ones.
    The OpenEye logfiles it existed to collect now land inside the chunk
    directory instead -- ``isomer_engine.oe_isomer`` runs the OpenEye section
    with its working directory set to the directory it owns -- so the loop
    below collects them like any other metadata file.

    Each move is guarded individually: a single file that cannot be moved
    (permissions, a vanished file) must not abandon the rest of the sweep and
    leave a half-populated ``verbose`` folder plus a spurious traceback
    behind. Everything here is diagnostic -- the ranked output is excluded and
    has already been written by the time this runs.

    Args:
        job_name: Path to the job directory containing files to move.
        folder: Destination folder for metadata files.
        optimized_structures: Path to the final output file (not moved).

    Returns:
        None. Moves files to the destination folder.

    Example:
        >>> housekeeping("/tmp/job1", "/tmp/job1/verbose", "/tmp/job1/output.sdf")
    """
    files = list(Path(job_name).glob("*"))
    for file in files:
        if str(file) == optimized_structures:
            continue
        try:
            shutil.move(str(file), folder)
        except OSError:
            logger.warning("Could not move %s into %s; leaving it where it is.", file, folder)
