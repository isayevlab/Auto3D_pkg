#!/usr/bin/env python
"""Calculating single point energy using ANI2xt, ANI2x or AIMNET"""
import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import torch
import warnings
from ase import Atoms
import ase.calculators.calculator
try:
    import torchani
    from .batch_opt.ANI2xt_no_rep import ANI2xt
except:
    pass
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm
from Auto3D.batch_opt.batchopt import mols2lists, EnForce_ANI
from Auto3D.batch_opt.batchopt import padding_coords, padding_species
from Auto3D.utils import hartree2ev


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1/hartree2ev


def calc_spe(path:str, model_name:str, gpu_idx=0):
    """Calculating single point energy
    path: Input sdf file
    model_name: ANI2x, ANI2xt or AIMNET
    gpu_idx: GPU cuda index"""
    #Create a output path that is the in the same directory as the input
    dir = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0] + f"_{model_name}_E.sdf"
    outpath = os.path.join(dir, basename)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    if model_name == "ANI2xt":
        model = EnForce_ANI(ANI2xt(device), model_name)
    elif model_name == "AIMNET":
        dict_path = os.path.join(root, "models/aimnet2nqed_pc14iall_b97m_sae.jpt")
        aimnet = torch.jit.load(dict_path, map_location=device)
        model = EnForce_ANI(aimnet, model_name)
    elif model_name == "ANI2x":
        calculator = torchani.models.ANI2x(periodic_table_index=True).to(device)
        model = EnForce_ANI(calculator, model_name)
    else:
        raise ValueError("model has to be 'ANI2x', 'ANI2xt' or 'AIMNET'")

    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    coord, numbers, charges = mols2lists(mols, model_name)
    if model_name == "AIMNET":
        coord_padded = padding_coords(coord, 0)
        numbers_padded = padding_species(numbers, 0)
    else:
        coord_padded = padding_coords(coord, 0)
        numbers_padded = padding_species(numbers, -1)
    
    # if model_name != "ANI2x":
    coord_padded = torch.tensor(coord_padded, device=device, requires_grad=True)
    numbers_padded = torch.tensor(numbers_padded, device=device)
    charges = torch.tensor(charges, device=device)
    es, fs = model.forward_batched(coord_padded, numbers_padded, charges)
    es = es.to('cpu').detach().numpy()

    with Chem.SDWriter(outpath) as f:
        for i, mol in enumerate(mols):
            mol.SetProp('E_hartree', str(es[i] * ev2hatree))
            f.write(mol)
    return outpath
