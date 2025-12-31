from __future__ import annotations

import os
from typing import Union

import pandas as pd
from rdkit import Chem

from Auto3D.auto3D import main
from Auto3D.config import Auto3DOptions
from Auto3D.utils import hartree2kcalpermol
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


def select_tautomers(sdf: str, k: int | None = None, window: float | None = None) -> str:
    """Select and Write the top-k or E <= window tautomers for each input SMILES
    Only k or window needs to be specified, NOT both.

    sdf: main function output
    
    Output: the path of the low-energy tautomer 3D conformers"""
    logger.info("Begin to select stable tautomers based on their conformer energies...")
    results = []
    if (k is not None) and (window is not None):
        raise ValueError("Only k OR window needs to be specified")        
    
    mols = Chem.SDMolSupplier(sdf, removeHs=False)
    for mol in mols:
        mol.ClearProp("E_rel(kcal/mol)")  #this is relative energies of conformers

    titles = [mol.GetProp("_Name") for mol in mols]
    ids = [title.split("@")[0].strip() for title in titles]
    energies = [float(mol.GetProp("E_tot")) * hartree2kcalpermol for mol in mols]
    df = pd.DataFrame({"id": ids, "energy": energies, "mol": mols})
    groups = df.groupby(by=["id"])
    for group_name in groups.indices:
        group = groups.get_group(group_name)
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
            raise ValueError("Either k OR window needs to be specified")
        results += out_mols
        

    folder = os.path.dirname(sdf)
    basename = os.path.basename(sdf).split(".")[0].strip() + "_top_tautomers.sdf"
    output_path = os.path.join(folder, basename)
    with Chem.SDWriter(output_path) as w:
        for mol in results:
            w.write(mol)
    logger.info("Done.")
    logger.info("The stable tautomers are stored in: %s", output_path)
    return output_path


def get_stable_tautomers(
    args: Union[dict, Auto3DOptions],
    tauto_k: int | None = None,
    tauto_window: float | None = None
) -> str:
    """Get stable tautomers for input molecules.

    Generates low-energy conformers and selects the most stable tautomers
    based on either top-k or energy window criteria.

    Args:
        args: Configuration options from the ``options()`` function
            or an ``Auto3DOptions`` instance. For backward compatibility,
            a dict with the same keys is also accepted.
        tauto_k: Keep the top-k tautomers (mutually exclusive with tauto_window).
        tauto_window: Keep tautomers within this energy window (kcal/mol)
            of the lowest energy tautomer (mutually exclusive with tauto_k).

    Returns:
        Path to the output SDF file containing stable tautomers.

    Example:
        >>> from Auto3D.auto3D import options
        >>> from Auto3D.tautomer import get_stable_tautomers
        >>> args = options("input.smi", k=1, enumerate_tautomer=True)
        >>> output = get_stable_tautomers(args, tauto_k=3)
    """
    out = main(args)
    out_tautomer = select_tautomers(out, tauto_k, tauto_window)
    return out_tautomer
