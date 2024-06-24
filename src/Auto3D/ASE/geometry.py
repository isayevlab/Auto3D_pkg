#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET, userNNP or ANI2x
"""
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import torch
from rdkit import Chem
from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.utils import hartree2ev


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def opt_geometry(path: str, model_name:str, gpu_idx=0, opt_tol=0.003, opt_steps=5000):
    """
    Geometry optimization interface with FIRE optimizer.

    :param path: Input sdf file
    :type path: str
    :param model_name: ANI2x, ANI2xt, userNNP or AIMNET
    :type model_name: str
    :param gpu_idx: GPU cuda index, defaults to 0
    :type gpu_idx: int, optional
    :param opt_tol: Convergence_threshold for geometry optimization (eV/A), defaults to 0.003
    :type opt_tol: float, optional
    :param opt_steps: Maximum geometry optimization steps, defaults to 5000
    :type opt_steps: int, optional
    """
    ev2hatree = 1/hartree2ev
    #create output path that is in the same directory as the input file
    dir = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0] + f"_{model_name}_opt.sdf"
    outpath = os.path.join(dir, basename)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    opt_config = {"opt_steps": opt_steps, "opttol": opt_tol,
                "patience": opt_steps, "batchsize_atoms": 1024}
    opt_engine = optimizing(path, outpath, model_name, device, opt_config)
    opt_engine.run()

    #change the energy unit from ev to hartree
    mols = list(Chem.SDMolSupplier(outpath, removeHs=False))
    with Chem.SDWriter(outpath) as f:
        for mol in mols:
            e = float(mol.GetProp('E_tot')) * ev2hatree
            mol.SetProp('E_tot', str(e))
            f.write(mol)
    return outpath


if __name__ == '__main__':
    path = '/home/jack/Auto3D_pkg/tests/files/DA.sdf'
    out = opt_geometry(path, 'ANI2x', gpu_idx=0, opt_tol=0.003, opt_steps=5000)
    print(out)
    out = opt_geometry(path, 'userNNP', gpu_idx=0, opt_tol=0.003, opt_steps=5000)
    print(out)
    out = opt_geometry(path, 'AIMNET', gpu_idx=0, opt_tol=0.003, opt_steps=5000)
    print(out)
    out = opt_geometry(path, 'ANI2xt', gpu_idx=0, opt_tol=0.003, opt_steps=5000)
    print(out)
