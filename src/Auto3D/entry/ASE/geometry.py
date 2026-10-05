#!/usr/bin/env python
"""
Geometry optimization with ANI2xt, AIMNET, userNNP or ANI2x
"""

from __future__ import annotations

from rdkit import Chem

from Auto3D.engines.batch_opt.batchopt import optimizing
from Auto3D.engines.model_factory import create_model
from Auto3D.entry._run_setup import prepare_single_file_run
from Auto3D.foundation.config import OptimizationConfig
from Auto3D.foundation.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
)
from Auto3D.foundation.exceptions import OptimizationError
from Auto3D.foundation.utils.atomic_io import atomic_write_path
from Auto3D.foundation.utils.energy import E_TOT_HARTREE_PROP, E_TOT_PROP

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

    Raises:
        InputValidationError: if no record of ``path`` is usable (unparseable,
            conformerless, implicit hydrogens, or dummy atoms); see the
            warnings logged for each record.
        OptimizationError: the optimizer reported that it wrote nothing even
            though ``path`` held usable records, so no optimized structure was
            produced.

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
    # Every guard, the output name and the record read, in one call shared with
    # calc_spe and calc_thermo -- see Auto3D.entry._run_setup for the step list,
    # the order, and why each step sits where it does. All of it happens before
    # create_model/optimizing load anything: nothing is downloaded or loaded for
    # a run that is about to be refused.
    setup = prepare_single_file_run(
        path,
        model_name,
        gpu_idx=gpu_idx,
        use_gpu=use_gpu,
        allow_tf32=allow_tf32,
        out_path=out_path,
        overwrite=overwrite,
        tag="opt",
    )
    # If every record of `path` was skipped as defective there is nothing to
    # optimize, and that is a defect in the INPUT -- an InputValidationError
    # (exit 2), not the OptimizationError (exit 7) this used to raise, which
    # names the optimizer for a file the optimizer never saw. Checked here
    # rather than relying on `optimizing.run()`'s own "input file is empty"/"no
    # valid molecules" early returns, which would load the model for nothing.
    setup.require_records(path)
    outpath = setup.out_path
    device = setup.device

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
    # `optimizing` (Auto3D.engines.batch_opt.batchopt) reads `path` itself
    # through `iter_conformer_records`, which is the same `classify_records`
    # policy the prologue above applied (N-C1), so the two agree on what counts
    # as a record without this function having to re-derive or re-write anything
    # for it -- which is why `setup.records` is only counted here, never passed
    # on.
    opt_engine = optimizing(path, outpath, adapter=adapter, device=device, config=opt_config)
    wrote_output = opt_engine.run()

    # optimizing.run() returns False (and leaves outpath untouched) when
    # `path` is missing, empty, contains no parseable record, or every record
    # was skipped by the filter as defective (`record_skip_reason`). Checked
    # on the RETURN VALUE, not `os.path.exists(outpath)`: with overwrite=True
    # (the default here), a stale outpath from an earlier call is left in
    # place by a skipped run, so an existence check alone would let
    # `_annotate_and_rewrite` below silently re-annotate and return THAT file
    # as if it were produced by this call. In practice this should not fire:
    # `setup.records` being non-empty (`require_records` above) already implies
    # `optimizing.run()` will find at least one record through its own,
    # identical filter -- this is defense against that invariant ever
    # drifting, not the primary guard, which is why it stays an
    # OptimizationError while the input check above is an InputValidationError.
    if not wrote_output:
        raise OptimizationError(
            f"No optimized structures were produced from {path!r}: the input "
            "file is missing, empty, contains no parseable record, or every "
            "record in it was skipped by the filter as defective -- no "
            "conformer, implicit hydrogens, or a dummy atom (atomic number 0)."
        )

    # `optimizing.run()` already wrote E_tot in Hartree; this pass only adds
    # the unit-labeled sibling, staged through a temp file so a failed rewrite
    # cannot destroy the completed optimization (C14)
    _annotate_and_rewrite(outpath)
    return outpath
