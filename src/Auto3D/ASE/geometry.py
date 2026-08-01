#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET, userNNP or ANI2x
"""
from __future__ import annotations

import os
import stat
import tempfile

from rdkit import Chem

from Auto3D.batch_opt.batchopt import optimizing
from Auto3D.config import OptimizationConfig
from Auto3D.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
)
from Auto3D.model_factory import get_device
from Auto3D.models.preflight import resolve_engine_name
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import hartree2ev
from Auto3D.utils.validation import (
    check_engine_supports_molecules,
    check_gpu_requested,
    check_output_not_input,
)

__all__ = ["opt_geometry"]


def _stage_beside(target: str) -> str:
    """Create an empty temp file in the same directory as ``target``.

    Same directory, because ``os.replace`` raises ``OSError`` across
    filesystems. The temp file inherits ``target``'s permission bits so the
    replaced file keeps the mode it had -- ``tempfile.mkstemp`` creates 0600,
    which would otherwise silently tighten the user's output file. Setting the
    mode before anything is written also preserves a read-only (0444) target's
    protection, which ``rename(2)`` would otherwise bypass.

    The parent directory is resolved with ``realpath``, not ``abspath``:
    ``abspath`` collapses ``..`` lexically, so a target like
    ``/scratch/link/../out.sdf`` (where ``link`` points at another mount)
    would stage the temp file in ``/scratch`` while the replace destination
    really lives elsewhere -- and ``os.replace`` would fail with
    ``EXDEV: Invalid cross-device link`` after a completed run. Only the
    PARENT is resolved: ``os.replace`` acts on the final path component
    itself, so following a symlinked ``target`` would pick the wrong
    directory.
    """
    directory = os.path.realpath(os.path.dirname(os.path.abspath(target)))
    fd, tmp_path = tempfile.mkstemp(suffix=".sdf", dir=directory)
    os.close(fd)
    try:
        os.chmod(tmp_path, stat.S_IMODE(os.stat(target).st_mode))
    except OSError:
        # Best effort: a mode we cannot read is not a reason to abandon a
        # completed optimization. Defensive rather than exercised -- the sole
        # caller, `_annotate_and_rewrite`, only reaches here after
        # `Chem.SDMolSupplier(target)` has already read the file, so `target`
        # exists and is stat-able on every path that gets here today. An
        # earlier version of this comment claimed the branch fires "in normal
        # runs whenever `target` does not exist yet", which is not true of any
        # current caller.
        pass
    return tmp_path


def _annotate_and_rewrite(outpath: str) -> None:
    """Convert E_tot from eV to hartree in-place, atomically.

    ``optimizing.run()`` has already written its only copy of the optimized
    geometries to ``outpath``. Opening ``Chem.SDWriter(outpath)`` directly
    would truncate that file, so a failure partway through the rewrite would
    destroy a completed optimization run (C14). Stage into a sibling temp file
    and ``os.replace`` it into position instead: ``os.replace`` is atomic on
    POSIX and on Windows, so ``outpath`` is only ever the old complete file or
    the new complete file, never a partial one.

    Staging does NOT by itself remove the Windows hazard from 74474ed, and an
    earlier version of this docstring wrongly claimed it did. ``reorder_sdf``
    was *already* staging through a temp file when it hit that bug: the failure
    was an open ``SDMolSupplier`` on the ``os.replace`` DESTINATION, which
    Windows refuses to overwrite (``PermissionError``/``WinError 5``) while a
    handle is held. This function reads ``outpath`` and then replaces it, so it
    has the same exposure -- see the explicit release below.
    """
    # `ev2hatree` is a LOCAL in opt_geometry, so a module-level helper cannot
    # see it -- recompute from the module-level `hartree2ev` import rather than
    # adding a parameter for a constant.
    ev2hatree = 1 / hartree2ev
    supp = Chem.SDMolSupplier(outpath, removeHs=False)
    mols = list(supp)
    # Release the handle on `outpath` BEFORE os.replace targets it, exactly as
    # utils/file_ops.py:743 does for reorder_sdf. Writing this as the anonymous
    # `list(Chem.SDMolSupplier(...))` would also work today -- the temporary's
    # refcount drops at the end of the statement -- but only under CPython's
    # refcounting, and it leaves the requirement invisible to the next person
    # who refactors this into a named variable.
    del supp
    tmp_path = _stage_beside(outpath)
    try:
        with Chem.SDWriter(tmp_path) as f:
            for mol in mols:
                # Skip records that failed to re-parse or lack E_tot rather
                # than crashing, which would discard the entire (already
                # completed) optimization run on a single bad record.
                if mol is None or not mol.HasProp("E_tot"):
                    continue
                e = float(mol.GetProp("E_tot")) * ev2hatree
                mol.SetProp("E_tot", str(e))
                f.write(mol)
        os.replace(tmp_path, outpath)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write must not
        # leave a stray .sdf beside the user's output.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
    opt_engine = optimizing(path, outpath, model_name, device, opt_config)
    opt_engine.run()

    # change the energy unit from eV to hartree, staged through a temp file so
    # a failed rewrite cannot destroy the completed optimization (C14)
    _annotate_and_rewrite(outpath)
    return outpath


