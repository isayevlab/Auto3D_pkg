import os
import pytest
# from openbabel import pybel
from rdkit import Chem
import Auto3D
from Auto3D.ASE.thermo import calc_thermo
from tests import skip_ani2xt_test


folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_calc_thermo_aimnet():
    #load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/DA.sdf")
    out = calc_thermo(path, "AIMNET", opt_tol=0.003)
    reference_Gs = {"diene": -230.17, "dieneophile": -359.67, "product": -589.84}

    #compare Auto3D output with the above
    # mols = list(pybel.readfile("sdf", out))
    # for mol in mols:
    #     thermo_out = float(mol.data["G_hartree"])
    #     idx = mol.title.strip()
    #     ref = reference_Gs[idx]
    #     diff = abs(thermo_out - ref)
    #     assert(diff <= 0.02)
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    for mol in mols:
        thermo_out = float(mol.GetProp("G_hartree"))
        idx = mol.GetProp('_Name').strip()
        ref = reference_Gs[idx]
        diff = abs(thermo_out - ref)
        assert(diff <= 0.02)
    try:
        os.remove(out)
    except:
        pass

