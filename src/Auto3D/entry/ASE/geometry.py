#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET, userNNP or ANI2x
"""

from __future__ import annotations

import os

from rdkit import Chem

from Auto3D.engines.batch_opt.batchopt import optimizing
from Auto3D.engines.model_factory import create_model, get_device
from Auto3D.engines.models.policy import (
    check_engine_supports_molecules,
    check_gpu_requested,
)
from Auto3D.engines.models.preflight import resolve_engine_name
from Auto3D.foundation.config import OptimizationConfig
from Auto3D.foundation.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
)
from Auto3D.foundation.torch_config import TorchConfig, configure_torch
from Auto3D.foundation.utils.atomic_io import atomic_write_path
from Auto3D.foundation.utils.energy import E_TOT_HARTREE_PROP, E_TOT_PROP
from Auto3D.foundation.utils.output_guard import check_output_not_input, check_output_overwrite

__all__ = ["opt_geometry"]


def _annotate_and_rewrite(outpath: str) -> None:
    """Add the unit-labeled ``E_tot(Hartree)`` sibling in-place, atomically.

    This function used to CONVERT ``E_tot`` from eV to Hartree, because
    ``optimizing.run()`` wrote eV. It no longer does: ``optimizing.run()``
    writes ``E_tot`` in Hartree like every other Auto3D writer (see
    ``Auto3D.foundation.utils.energy``), so converting again here would divide by 27.211
    a second time. Two jobs remain, and neither is a no-op: this pass DROPS
    records that failed to re-parse or carry no ``E_tot`` (so ``opt_geometry``
    output contains only usable energies), and it guarantees the unit-labeled
    ``E_tot(Hartree)`` sibling regardless of what the optimizer wrote -- the
    same guarantee ``ConformerRanker`` makes for the ranked output. Setting
    the label is idempotent: ``optimizing.run()`` already writes it, and
    re-asserting the identical string here keeps the guarantee attached to
    ``opt_geometry`` itself rather than to whichever writer ran upstream.

    ``optimizing.run()`` has already written its only copy of the optimized
    geometries to ``outpath``. Opening ``Chem.SDWriter(outpath)`` directly
    would truncate that file, so a failure partway through the rewrite would
    destroy a completed optimization run (C14). Stage into a sibling temp file
    and ``os.replace`` it into position instead -- which is what
    :func:`Auto3D.foundation.utils.atomic_io.atomic_write_path` does, for this and the
    other two in-place rewrites in Auto3D. ``os.replace`` is atomic on POSIX
    and on Windows, so ``outpath`` is only ever the old complete file or the
    new complete file, never a partial one.

    Staging does NOT by itself remove the Windows hazard from 74474ed, and an
    earlier version of this docstring wrongly claimed it did. ``reorder_sdf``
    was *already* staging through a temp file when it hit that bug: the failure
    was an open ``SDMolSupplier`` on the ``os.replace`` DESTINATION, which
    Windows refuses to overwrite (``PermissionError``/``WinError 5``) while a
    handle is held. This function reads ``outpath`` and then replaces it, so it
    has the same exposure -- see the explicit release below. Releasing the
    handle stays the caller's duty; ``atomic_write_path`` cannot do it.
    """
    supp = Chem.SDMolSupplier(outpath, removeHs=False)
    mols = list(supp)
    # Release the handle on `outpath` BEFORE os.replace targets it, exactly as
    # utils/sdf_io.py does for reorder_sdf. Writing this as the anonymous
    # `list(Chem.SDMolSupplier(...))` would also work today -- the temporary's
    # refcount drops at the end of the statement -- but only under CPython's
    # refcounting, and it leaves the requirement invisible to the next person
    # who refactors this into a named variable.
    del supp
    with atomic_write_path(outpath, suffix=".sdf") as tmp_path, Chem.SDWriter(tmp_path) as f:
        for mol in mols:
            # Skip records that failed to re-parse or lack E_tot rather
            # than crashing, which would discard the entire (already
            # completed) optimization run on a single bad record.
            if mol is None or not mol.HasProp(E_TOT_PROP):
                continue
            # Same number, stated in a name that carries its unit. No
            # arithmetic: E_tot is already Hartree when it gets here.
            mol.SetProp(E_TOT_HARTREE_PROP, mol.GetProp(E_TOT_PROP))
            f.write(mol)


def opt_geometry(
    path: str,
    model_name: str,
    gpu_idx: int = 0,
    opt_tol: float = DEFAULT_CONVERGENCE_THRESHOLD,
    opt_steps: int = DEFAULT_OPT_STEPS,
    patience: int | None = None,
    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS,
    use_gpu: bool = True,
    allow_tf32: bool = False,
    out_path: str | None = None,
    overwrite: bool = True,
) -> str:
    """Geometry optimization interface with FIRE optimizer.

    Optimizes molecular geometries from an SDF file using neural network
    potentials (ANI2x, ANI2xt, AIMNET, or custom models).

    Args:
        path: Input SDF file path.
        model_name: Model for optimization. Options:
            - 'ANI2x': ANI2x neural network potential
            - 'ANI2xt': ANI2xt neural network potential
            - 'AIMNET': AIMNet2 model (default in Auto3D; alias for 'aimnet2')
            - Any aimnet registry name, e.g. 'aimnet2-2025', 'aimnet2-nse', 'aimnet2-pd'
            - Path to custom NNP model file (.pt)
        gpu_idx: CUDA device index. Defaults to 0.
        opt_tol: Convergence threshold for max force (eV/Å). Defaults to 0.01.
        opt_steps: Maximum optimization steps per structure. Defaults to 2000.
        patience: Drop conformer if force doesn't decrease for this many
            consecutive steps. Defaults to None (uses opt_steps value).
        batchsize_atoms: Number of atoms per optimization batch, used **as
            given**. Larger values use more GPU memory but may be faster.
            Defaults to 1024.

            Note the difference from ``main()``/``Auto3DOptions``, where the same
            parameter name is a per-gigabyte *multiplier*: ``ChunkManager``
            multiplies it by the available memory and clamps the product at
            16,384, so ``batchsize_atoms=1024`` means 1024 there on a 1 GB card
            and 16,384 from 16 GB upward, while here it always means 1024. Two
            meanings for one name, which is why each is spelled out rather than
            cross-referenced.
        use_gpu: Use the GPU when available. Defaults to True.
        allow_tf32: Enable TF32 matmul precision on Ampere+ GPUs. Defaults to False.
        out_path: Output SDF path. Defaults to ``<input_stem>_<model>_opt.sdf``
            next to the input file.
        overwrite: Allow writing over an existing output file. Defaults to
            True, which is the historical behavior every Python-API caller
            was written against. ``auto3d optimize`` passes False unless
            ``--force`` is given, so the CLI refuses to clobber.

    Returns:
        Path to output SDF file with optimized geometries.

    Example:
        >>> from Auto3D.entry.ASE.geometry import opt_geometry
        >>> output = opt_geometry(
        ...     "molecules.sdf",
        ...     "AIMNET",
        ...     gpu_idx=0,
        ...     patience=250,
        ...     batchsize_atoms=2048,
        ... )
    """
    # Fail fast on an unrecognized engine name -- the same guard the CLI's
    # `optimize` command already runs before calling this function
    # (cli/commands/properties.py), now also enforced for direct Python-API
    # callers. Pure offline registry lookup: no network, no model load.
    resolve_engine_name(model_name)

    # opt_geometry never goes through check_input/check_valid_configuration,
    # so without this it would reach model_factory.get_device below and
    # silently fall back to CPU instead of failing the same way `auto3d
    # optimize` already does at its CLI wrapper (cli/commands/properties.py)
    # -- and the same way `auto3d run`/smiles2mols do via check_input /
    # check_valid_configuration. check_gpu_requested is the single source of
    # truth for this policy; called here, before get_device/optimizing below,
    # so no compute (and no model construction) happens first.
    check_gpu_requested(use_gpu)

    # Refuse `-o` pointing at the input: opt_geometry would otherwise stage a
    # rewrite of the very file it is reading and destroy the user's input
    # (C14). Shared guard, so calc_spe/opt_geometry/calc_thermo cannot drift
    # apart on this policy. Needs only the two paths, so it runs before
    # get_device/optimizing.
    check_output_not_input(path, out_path)

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

    # Refuse to truncate a file that already exists. `optimizing.run()` opens
    # `Chem.SDWriter(outpath)`, which truncates on open, so without this
    # `-o precious.sdf` destroyed precious.sdf. Not at the start: that writer
    # is opened at batch_opt/batchopt.py:323, AFTER every bucket has been
    # optimized, so precious.sdf survived the whole optimization and was
    # replaced by the final write -- a completed run, no error, no warning.
    # Checked on the RESOLVED path, so the derived default name is covered
    # too, and before get_device/optimizing so nothing is loaded first.
    check_output_overwrite(outpath, overwrite)

    device = get_device(gpu_idx, use_gpu=use_gpu)

    # ANI2x/ANI2xt can only represent uncharged, in-set molecules (C11): a
    # charged or out-of-set species handed to either would otherwise be
    # silently optimized as a different, neutral species -- wrong energy,
    # wrong forces, wrong geometry.
    input_mols = [mol for mol in Chem.SDMolSupplier(path, removeHs=False) if mol is not None]
    check_engine_supports_molecules(input_mols, model_name)

    opt_config = OptimizationConfig(
        opt_steps=opt_steps,
        convergence_threshold=opt_tol,
        patience=patience if patience is not None else opt_steps,
        batchsize_atoms=batchsize_atoms,
    )
    # Built here, in the process that runs the optimization. `optimizing` no
    # longer constructs its own adapter (audit M41); see the note at
    # `Auto3D.orchestration.workflow_workers.optim_rank_wrapper` about why construction must
    # not be hoisted past the frame that does the work.
    adapter = create_model(model_name, device)
    opt_engine = optimizing(path, outpath, adapter=adapter, device=device, config=opt_config)
    opt_engine.run()

    # `optimizing.run()` already wrote E_tot in Hartree; this pass only adds
    # the unit-labeled sibling, staged through a temp file so a failed rewrite
    # cannot destroy the completed optimization (C14)
    _annotate_and_rewrite(outpath)
    return outpath
