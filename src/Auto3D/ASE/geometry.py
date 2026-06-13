#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET, userNNP or ANI2x
"""
from __future__ import annotations

import os

from rdkit import Chem

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.config import OptimizationConfig
from Auto3D.model_factory import get_device
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import hartree2ev


def opt_geometry(
    path: str,
    model_name: str,
    gpu_idx: int = 0,
    opt_tol: float = 0.01,
    opt_steps: int = 2000,
    patience: int | None = None,
    batchsize_atoms: int = 1024,
    use_gpu: bool = True,
    allow_tf32: bool = False,
    out_path: str | None = None,
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
        use_gpu: Use the GPU when available. Defaults to True.
        allow_tf32: Enable TF32 matmul precision on Ampere+ GPUs. Defaults to False.
        out_path: Output SDF path. Defaults to ``<input_stem>_<model>_opt.sdf``
            next to the input file.

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
    # Apply the shared torch configuration so allow_tf32 is honored here too
    # (this path previously ignored it).
    configure_torch(TorchConfig(allow_tf32=allow_tf32))

    # Create output path in the same directory as the input file (unless
    # overridden). splitext (not split(".")) so an input like 'batch.v2.sdf'
    # keeps 'batch.v2' instead of collapsing to 'batch' and risking collisions.
    if out_path is not None:
        outpath = out_path
    else:
        dir = os.path.dirname(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(model_name):  # custom NNP passed as a file path
            basename = stem + "_userNNP_opt.sdf"
        else:
            basename = stem + f"_{model_name}_opt.sdf"
        outpath = os.path.join(dir, basename)

    device = get_device(gpu_idx, use_gpu=use_gpu)

    opt_config = OptimizationConfig(
        opt_steps=opt_steps,
        convergence_threshold=opt_tol,
        patience=patience if patience is not None else opt_steps,
        batchsize_atoms=batchsize_atoms,
    )
    opt_engine = optimizing(path, outpath, model_name, device, opt_config)
    opt_engine.run()

    #change the energy unit from ev to hartree
    mols = list(Chem.SDMolSupplier(outpath, removeHs=False))
    with Chem.SDWriter(outpath) as f:
        for mol in mols:
            # Skip records that failed to re-parse or lack E_tot rather than
            # crashing here, which would discard the entire (already completed)
            # optimization run on a single bad record.
            if mol is None or not mol.HasProp('E_tot'):
                continue
            e = float(mol.GetProp('E_tot')) * ev2hatree
            mol.SetProp('E_tot', str(e))
            f.write(mol)
    return outpath


