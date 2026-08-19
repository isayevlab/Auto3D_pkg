from __future__ import annotations

import os

import pandas as pd
from rdkit import Chem

from Auto3D.entry.auto3D import main
from Auto3D.foundation.config import Auto3DOptions
from Auto3D.foundation.exceptions import ConfigurationError
from Auto3D.foundation.utils.energy import e_tot_hartree, hartree2kcalpermol
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.output_guard import check_output_overwrite

logger = get_logger(__name__)

__all__ = ["select_tautomers", "get_stable_tautomers"]


def select_tautomers(
    sdf: str,
    k: int | None = None,
    window: float | None = None,
    *,
    overwrite: bool = False,
) -> str:
    """Select and Write the top-k or E <= window tautomers for each input SMILES
    Only k or window needs to be specified, NOT both.

    sdf: main function output

    Output: the path of the low-energy tautomer 3D conformers

    Args:
        sdf: Path to the SDF this function reads, normally ``main()``'s output.
        k: Keep the top-k tautomers per input molecule.
        window: Keep tautomers within this kcal/mol window of the most stable
            one. Mutually exclusive with `k`.
        overwrite: Allow replacing an existing
            ``<dir(sdf)>/<stem>_top_tautomers.sdf``. Keyword-only, and
            defaults to False because that name is one Auto3D *derives* rather
            than one the caller chose: ``Chem.SDWriter`` truncates on open, so
            ``select_tautomers("/data/results.sdf", k=1)`` used to replace an
            existing ``/data/results_top_tautomers.sdf`` with this call's
            selection, silently. Auto3D's own routes are unaffected --
            ``get_stable_tautomers`` passes ``main()``'s output, which lives in
            a job directory created fresh for that run, and ``auto3d
            tautomers`` additionally gates its ``-o`` with
            ``check_output_overwrite`` -- so the default only ever fires for a
            direct API caller pointing at an occupied path.

    Note:
        Tautomers are ranked by the optimized NNP *electronic* energy (``E_tot``)
        only -- no zero-point energy or thermal/entropy correction. Tautomer
        equilibria (e.g. keto/enol, 2-pyridone/2-hydroxypyridine) are often
        decided by ZPE/entropy differences of 1-2 kcal/mol that can invert the
        electronic-energy ordering, so the "most stable" tautomer here is an
        electronic-energy estimate, not a free-energy one. Use the thermo module
        (Auto3D.entry.ASE.thermo) for a free-energy comparison when that matters.

    Raises:
        ConfigurationError: If both k and window are given, if neither is
            given, if k < 1, or if the derived output path exists and
            `overwrite` is False. ``auto3d tautomers`` already rejects the
            both-given case in ``execute_tautomers`` before calling this
            function, but ``select_tautomers``/``get_stable_tautomers`` are
            also public Python API entry points that can be called directly,
            bypassing that CLI-level guard."""
    logger.info("Begin to select stable tautomers based on their conformer energies...")
    results = []
    if (k is not None) and (window is not None):
        raise ConfigurationError("Only k OR window needs to be specified")
    # Checked here, not inside the grouping loop below: a loop-scoped check
    # cannot fire for an input that yields no groups, so a zero-record SDF
    # used to write an empty output and return its path for a call that
    # specified no selection at all.
    if (k is None) and (window is None):
        raise ConfigurationError("Either k OR window needs to be specified")
    if (k is not None) and (k < 1):
        raise ConfigurationError(f"tauto_k must be >= 1, got {k}")

    # Resolved and gated up front, before the input is read and grouped:
    # refusing after all the work is done costs the user the run for nothing,
    # and the writer at the bottom truncates the moment it opens.
    folder = os.path.dirname(sdf)
    # splitext (not split(".")) so an input like 'mol.v2.sdf' keeps 'mol.v2'
    # instead of collapsing to 'mol' and risking output collisions.
    stem = os.path.splitext(os.path.basename(sdf))[0].strip()
    output_path = os.path.join(folder, stem + "_top_tautomers.sdf")
    check_output_overwrite(output_path, overwrite)

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
    energies = [e_tot_hartree(mol) * hartree2kcalpermol for mol in mols]
    df = pd.DataFrame({"id": ids, "energy": energies, "mol": mols})
    for group_name, group in df.groupby("id"):
        group = group.sort_values(by="energy")
        out_mols0 = list(group["mol"])
        ref_energy = e_tot_hartree(out_mols0[0]) * hartree2kcalpermol
        # select top k
        if k is not None:
            if k >= len(out_mols0):
                out_mols = out_mols0
            else:
                out_mols = out_mols0[:k]
            for mol in out_mols:
                mol_energy = e_tot_hartree(mol) * hartree2kcalpermol
                e_rel = mol_energy - ref_energy
                mol.SetProp("E_tautomer_relative(kcal/mol)", str(e_rel))
                mol.SetProp("_Name", group_name)
        # select E <= window -- window is not None here, guaranteed by the
        # argument check above
        else:
            out_mols = []
            for mol in out_mols0:
                mol_energy = e_tot_hartree(mol) * hartree2kcalpermol
                e_rel = mol_energy - ref_energy
                if e_rel <= window:
                    mol.SetProp("E_tautomer_relative(kcal/mol)", str(e_rel))
                    mol.SetProp("_Name", group_name)
                    out_mols.append(mol)
        results += out_mols

    with Chem.SDWriter(output_path) as w:
        for mol in results:
            w.write(mol)
    logger.info("Done.")
    logger.info("The stable tautomers are stored in: %s", output_path)
    return output_path


def get_stable_tautomers(
    args: Auto3DOptions, tauto_k: int | None = None, tauto_window: float | None = None
) -> str:
    """Get stable tautomers for input molecules.

    Generates low-energy conformers and selects the most stable tautomers
    based on either top-k or energy window criteria.

    Args:
        args: Configuration options as an ``Auto3DOptions`` instance. A plain
            dict is *not* accepted: this forwards to :func:`Auto3D.entry.auto3D.main`,
            which reads attributes off the object.
        tauto_k: Keep the top-k tautomers (mutually exclusive with tauto_window).
        tauto_window: Keep tautomers within this energy window (kcal/mol)
            of the lowest energy tautomer (mutually exclusive with tauto_k).

    Returns:
        Path to the output SDF file containing stable tautomers.

    Example:
        >>> from Auto3D import Auto3DOptions
        >>> from Auto3D.entry.tautomer import get_stable_tautomers
        >>> args = Auto3DOptions(path="input.smi", k=1, enumerate_tautomer=True)
        >>> output = get_stable_tautomers(args, tauto_k=3)
    """
    out = main(args)
    out_tautomer = select_tautomers(out, tauto_k, tauto_window)
    return out_tautomer
