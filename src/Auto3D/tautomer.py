import os
from rdkit import Chem
import pandas as pd
from typing import Optional
from Auto3D.auto3D import main
from Auto3D.utils import hartree2kcalpermol


def select_tautomers(sdf: str, k: Optional[int]=None, window:Optional[float]=None) -> str:
    """Select and Write the top-k or E <= window tautomers for each input SMILES
    Only k or window needs to be specified, NOT both.

    sdf: main function output
    
    Output: the path of the low-energy tautomer 3D conformers"""
    print(f"\nBegin to select stable tautomers based on their conformer energies...", flush=True)
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
    print("Done.", flush=True)
    print("The stable tautomers are stored in: %s" % output_path, flush=True)
    return output_path


def get_stable_tautomers(args: dict,  tauto_k: Optional[int]=None, tauto_window:Optional[float]=None) -> str:
    """
    args: the `options` function output, it's used for generating low-energy conformers
    tauto_k: keep the top-k tautomers
    tauto_window: keep the tautomers whose energies are within `window` kcal/mol

    Output:
    an SDF file storing the stable tautomers from the input SMILES file
    """
    out = main(args)
    out_tautomer = select_tautomers(out, tauto_k, tauto_window)
    return out_tautomer
