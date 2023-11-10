import os
import pickle
import numpy as np
import torch
from rdkit import Chem
import Auto3D
from Auto3D.ASE.thermo import model_name2model_calculator, vib_hessian
from Auto3D.ASE.thermo import calc_thermo


folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_calc_thermo_aimnet():
    #load wB97m-D4/Def2-TZVPP output file
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    reference_G = -314.49236715
    reference_H = -314.45168666

    #compare Auto3D output with the above
    out = calc_thermo(path, "AIMNET", opt_tol=0.003)
    mol = next(Chem.SDMolSupplier(out, removeHs=False))

    G_out = float(mol.GetProp("G_hartree"))
    H_out = float(mol.GetProp("H_hartree"))
    assert(abs(reference_G - G_out) <= 0.02)
    assert(abs(reference_H - H_out) <= 0.02)
    try:
        os.remove(out)
    except:
        pass

def test_vib_hessian():
    path = os.path.join(folder, "tests/files/cyclooctane.sdf")
    mol = next(Chem.SDMolSupplier(path, removeHs=False))

    _, calculator = model_name2model_calculator('AIMNET')
    model_path = os.path.join(folder, "src/Auto3D/models/aimnet2_wb97m-d3_0.jpt")
    device = torch.device('cpu')
    model = torch.jit.load(model_path, map_location=device)

    hessian_vib = vib_hessian(mol, calculator, model)
    hessian_freq = hessian_vib.get_frequencies()

    ase_freq = pickle.load(open(os.path.join(folder, "tests/files/cyclooctane_ase_freq.pkl"), "rb"))
    mean_diff = np.mean(np.abs(hessian_freq[6:] - ase_freq[6:]))
    assert(mean_diff <= 10)  # 10 cm-1 error is acceptable

if __name__ == "__main__":
    test_calc_thermo_aimnet()
    test_vib_hessian()
