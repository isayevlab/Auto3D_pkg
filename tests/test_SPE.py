import os
from openbabel import pybel
import Auto3D
from Auto3D.SPE import calc_spe


folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_calc_spe_ani2xt():
    #load B97-3c results file
    path = os.path.join(folder, "tests/files/b973c.sdf")
    out = calc_spe(path, "ANI2xt")
    spe = {"817-2-473": -386.111, "510-2-443":-1253.812}

    mols = list(pybel.readfile("sdf", out))
    for mol in mols:
        spe_out = float(mol.data["E_hatree"])
        idx = mol.data["ID"].strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert(diff <= 0.01)


def test_calc_spe_ani2x():
    #load wB97X/6-31G* output file
    path = os.path.join(folder, "tests/files/wb97x_dz.sdf")
    spe = {"817-2-473": -386.178, "510-2-443":-1254.007}
    out = calc_spe(path, "ANI2x")

    #compare Auto3D output with the above
    mols = list(pybel.readfile("sdf", out))
    for mol in mols:
        spe_out = float(mol.data["E_hatree"])
        idx = mol.data["ID"].strip()
        spe_ref = spe[idx]
        diff = abs(spe_out - spe_ref)
        assert(diff <= 0.02)

