from __future__ import annotations

import os

import pandas as pd
from rdkit import Chem

from Auto3D.auto3D import main
from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import ConfigurationError
from Auto3D.utils import hartree2kcalpermol
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["select_tautomers", "get_stable_tautomers"]


def select_tautomers(sdf: str, k: int | None = None, window: float | None = None) -> str:
    """Select and Write the top-k or E <= window tautomers for each input SMILES
    Only k or window needs to be specified, NOT both.

    sdf: main function output

    Output: the path of the low-energy tautomer 3D conformers

    Warning:
        This function writes ``<dir(sdf)>/<stem>_top_tautomers.sdf`` with no
        overwrite gate: ``Chem.SDWriter`` truncates on open, so a direct call
        such as ``select_tautomers("/data/results.sdf", k=1)`` replaces any
        existing ``/data/results_top_tautomers.sdf``. Every route Auto3D
        itself takes is safe -- ``get_stable_tautomers`` passes ``main()``'s
        output, which lives in a job directory created fresh for that run, and
        ``auto3d tautomers`` additionally gates its ``-o`` with
        ``check_output_overwrite`` -- so this is a hazard for direct API
        callers only, the same residual class as
        ``Auto3D.utils.file_ops.smiles2smi`` and ``decode_ids``. See
        ``docs/superpowers/follow-ups-after-4.0.0-remediation.md``.

    Note:
        Tautomers are ranked by the optimized NNP *electronic* energy (``E_tot``)
        only -- no zero-point energy or thermal/entropy correction. Tautomer
        equilibria (e.g. keto/enol, 2-pyridone/2-hydroxypyridine) are often
        decided by ZPE/entropy differences of 1-2 kcal/mol that can invert the
        electronic-energy ordering, so the "most stable" tautomer here is an
        electronic-energy estimate, not a free-energy one. Use the thermo module
        (Auto3D.ASE.thermo) for a free-energy comparison when that matters.

    Raises:
        ConfigurationError: If both k and window are given, if neither is
            given, or if k < 1. ``auto3d tautomers`` already rejects the
            both-given case in ``execute_tautomers`` before calling this
            function, but ``select_tautomers``/``get_stable_tautomers`` are
            also public Python API entry points that can be called directly,
            bypassing that CLI-level guard."""
    logger.info("Begin to select stable tautomers based on their conformer energies...")
    results = []
    if (k is not None) and (window is not None):
        raise ConfigurationError("Only k OR window needs to be specified")
    if (k is not None) and (k < 1):
        raise ConfigurationError(f"tauto_k must be >= 1, got {k}")

    supplier = Chem.SDMolSupplier(sdf, removeHs=False)
    mols = [m for m in supplier if m is not None]
    for mol in mols:
        if mol.HasProp("E_rel(kcal/mol)"):
            mol.ClearProp("E_rel(kcal/mol)")  # conformer-level energy, not tautomer-level

    titles = [mol.GetProp("_Name") for mol in mols]
    # Tautomers of one input molecule are named "id@tautN", so the base ID is
    # the part before '@' -- this is the real tautomer-grouping separator (see
    # test_select_tautomers_groups_by_id), distinct from the '_' conformer index.
    ids = [title.split("@")[0].strip() for title in titles]
    energies = [float(mol.GetProp("E_tot")) * hartree2kcalpermol for mol in mols]
    df = pd.DataFrame({"id": ids, "energy": energies, "mol": mols})
    for group_name, group in df.groupby("id"):
        group = group.sort_values(by="energy")
        out_mols0 = list(group["mol"])
        ref_energy = float(out_mols0[0].GetProp("E_tot")) * hartree2kcalpermol
        #select top k
        if k is not None:
            if k >= len(out_mols0):
                out_mols = out_mols0
            else:
                out_mols = out_mols0[:k]
            for mol in out_mols:
                mol_energy = float(mol.GetProp("E_tot")) * hartree2kcalpermol
                e_rel = mol_energy - ref_energy
                mol.SetProp("E_tautomer_relative(kcal/mol)", str(e_rel))
                mol.SetProp("_Name", group_name)
        #select E <= window
        elif window is not None:
            out_mols = []
            for mol in out_mols0:
                mol_energy = float(mol.GetProp("E_tot")) * hartree2kcalpermol
                e_rel = mol_energy - ref_energy
                if e_rel <= window:
                    mol.SetProp("E_tautomer_relative(kcal/mol)", str(e_rel))
                    mol.SetProp("_Name", group_name)
                    out_mols.append(mol)
        else:
            raise ConfigurationError("Either k OR window needs to be specified")
        results += out_mols
        

    folder = os.path.dirname(sdf)
    # splitext (not split(".")) so an input like 'mol.v2.sdf' keeps 'mol.v2'
    # instead of collapsing to 'mol' and risking output collisions.
    stem = os.path.splitext(os.path.basename(sdf))[0].strip()
    basename = stem + "_top_tautomers.sdf"
    output_path = os.path.join(folder, basename)
    with Chem.SDWriter(output_path) as w:
        for mol in results:
            w.write(mol)
    logger.info("Done.")
    logger.info("The stable tautomers are stored in: %s", output_path)
    return output_path


def get_stable_tautomers(
    args: dict | Auto3DOptions,
    tauto_k: int | None = None,
    tauto_window: float | None = None
) -> str:
    """Get stable tautomers for input molecules.

    Generates low-energy conformers and selects the most stable tautomers
    based on either top-k or energy window criteria.

    Args:
        args: Configuration options as an ``Auto3DOptions`` instance.
            For backward compatibility, a dict with the same keys is also accepted.
        tauto_k: Keep the top-k tautomers (mutually exclusive with tauto_window).
        tauto_window: Keep tautomers within this energy window (kcal/mol)
            of the lowest energy tautomer (mutually exclusive with tauto_k).

    Returns:
        Path to the output SDF file containing stable tautomers.

    Example:
        >>> from Auto3D import Auto3DOptions
        >>> from Auto3D.tautomer import get_stable_tautomers
        >>> args = Auto3DOptions(path="input.smi", k=1, enumerate_tautomer=True)
        >>> output = get_stable_tautomers(args, tauto_k=3)
    """
    out = main(args)
    out_tautomer = select_tautomers(out, tauto_k, tauto_window)
    return out_tautomer
