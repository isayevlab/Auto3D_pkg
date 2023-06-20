import os
import pytest
from rdkit import Chem
import Auto3D
from Auto3D.SPE import calc_spe
# from tests import skip_ani2xt_test
skip_ani2xt_test = False

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.mark.skipif(skip_ani2xt_test, reason="ANI2xt model is not  installed.")
def test_calc_spe_ani2xt():
    #load B97-3c results file
    path = os.path.join(folder, "tests/files/b973c.sdf")
    out = calc_spe(path, "ANI2xt")
    spe = {"817-2-473": -386.111, "510-2-443":-1253.812}

    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert(diff <= 0.01)


def test_calc_spe_ani2x():
    #load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/wb97x_dz.sdf")
    spe = {"817-2-473": -386.178, "510-2-443":-1254.007}
    out = calc_spe(path, "ANI2x")

    #compare Auto3D output with the above
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    for mol in mols:
        spe_out = float(mol.GetProp("E_hartree"))
        idx = mol.GetProp("ID").strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert(diff <= 0.01)

if __name__ == "__main__":
    test_calc_spe_ani2xt()
    # test_calc_spe_ani2x()