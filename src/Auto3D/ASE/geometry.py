#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET, userNNP or ANI2x
"""
from __future__ import annotations

import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import torch
from rdkit import Chem

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.utils import hartree2ev

# TF32 settings are configured centrally via Auto3D.torch_config.configure_torch()
# and the allow_tf32 option in Auto3DOptions.


def opt_geometry(
    path: str,
    model_name: str,
    gpu_idx: int = 0,
    opt_tol: float = 0.01,
    opt_steps: int = 2000,
    patience: int | None = None,
    batchsize_atoms: int = 1024,
) -> str:
    """Geometry optimization interface with FIRE optimizer.

    Optimizes molecular geometries from an SDF file using neural network
    potentials (ANI2x, ANI2xt, AIMNET, or custom models).

    Args:
        path: Input SDF file path.
        model_name: Model for optimization. Options:
            - 'ANI2x': ANI2x neural network potential
            - 'ANI2xt': ANI2xt neural network potential
            - 'AIMNET': AIMNet2 model (default in Auto3D)
            - Path to custom NNP model file (.pt)
        gpu_idx: CUDA device index. Defaults to 0.
        opt_tol: Convergence threshold for max force (eV/Å). Defaults to 0.01.
        opt_steps: Maximum optimization steps per structure. Defaults to 2000.
        patience: Drop conformer if force doesn't decrease for this many
            consecutive steps. Defaults to None (uses opt_steps value).
        batchsize_atoms: Number of atoms per optimization batch. Larger values
            use more GPU memory but may be faster. Defaults to 1024.
            Recommendation: ~1024 per GB of GPU memory.

    Returns:
        Path to output SDF file with optimized geometries.

    Example:
        >>> from Auto3D.ASE.geometry import opt_geometry
        >>> output = opt_geometry(
        ...     "molecules.sdf",
        ...     "AIMNET",
        ...     gpu_idx=0,
        ...     patience=250,
        ...     batchsize_atoms=2048,
        ... )
    """
    ev2hatree = 1 / hartree2ev
    # Create output path in the same directory as the input file
    dir = os.path.dirname(path)
    if os.path.exists(path):
        basename = os.path.basename(path).split(".")[0] + "_userNNP_opt.sdf"
    else:
        basename = os.path.basename(path).split(".")[0] + f"_{model_name}_opt.sdf"
    outpath = os.path.join(dir, basename)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    opt_config = {
        "opt_steps": opt_steps,
        "opttol": opt_tol,
        "patience": patience if patience is not None else opt_steps,
        "batchsize_atoms": batchsize_atoms,
    }
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


