#!/usr/bin/env python
"""Calculating single point energy using ANI2xt, ANI2x, 'userNNP' or AIMNET"""
from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem

from Auto3D.batch_opt.batchopt import EnForce_ANI, mols2lists
from Auto3D.batch_opt.padding import pad_from_mols
from Auto3D.model_factory import create_model, get_device
from Auto3D.utils import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1/hartree2ev


def calc_spe(path: str, model_name: str, gpu_idx: int = 0) -> str:
    """
    Calculates single point energy.

    :param path: Input sdf file
    :type path: str
    :param model_name: AIMNET, ANI2x, userNNP, or ANI2xt
    :type model_name: str
    :param gpu_idx: GPU cuda index, defaults to 0
    :type gpu_idx: int, optional
    :return: Path to output SDF file with energies
    :rtype: str
    """
    # Create output path in the same directory as the input
    dir_path = Path(path).parent
    stem = Path(path).stem
    if Path(model_name).exists():
        basename = f"{stem}_userNNP_E.sdf"
    else:
        basename = f"{stem}_{model_name}_E.sdf"
    outpath = dir_path / basename

    # Use get_device from model_factory
    device = get_device(gpu_idx)

    # Use ModelFactory to create model adapter
    model_adapter = create_model(model_name, device)

    # Create EnForce_ANI wrapper for batched forward support (new API without name)
    model = EnForce_ANI(model_adapter)

    mols = list(Chem.SDMolSupplier(path, removeHs=False))

    # Use new vectorized padding that returns tensors directly
    coord_padded, numbers_padded, charges = pad_from_mols(
        mols, model_name, device,
        coord_pad=model_adapter.coord_pad, species_pad=model_adapter.species_pad
    )

    es, fs = model.forward_batched(coord_padded, numbers_padded, charges)
    es = es.to('cpu').detach().numpy()

    with Chem.SDWriter(str(outpath)) as f:
        for i, mol in enumerate(mols):
            mol.SetProp('E_hartree', str(es[i] * ev2hatree))
            f.write(mol)
    return str(outpath)

