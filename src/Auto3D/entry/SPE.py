#!/usr/bin/env python
"""Calculating single point energy using ANI2xt, ANI2x, AIMNET or a userNNP model file"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from Auto3D.engines.batch_opt.model_wrapper import EnForce_ANI
from Auto3D.engines.batch_opt.padding import pad_from_mols
from Auto3D.engines.model_factory import create_model
from Auto3D.entry._run_setup import prepare_single_file_run
from Auto3D.foundation.utils.energy import set_e_hartree_from_ev

__all__ = ["calc_spe"]


def calc_spe(
    path: str,
    model_name: str,
    gpu_idx: int = 0,
    use_gpu: bool = True,
    allow_tf32: bool = False,
    out_path: str | None = None,
    overwrite: bool = True,
) -> str:
    """Calculates single point energy.

    Args:
        path: Input sdf file.
        model_name: ``AIMNET``, ``ANI2x``, ``ANI2xt``, an aimnet registry name
            (``aimnet2``, ``aimnet2-2025``, ...), or a path to a userNNP model
            file. The literal string ``userNNP`` is not an engine name.
        gpu_idx: GPU cuda index. Defaults to 0.
        use_gpu: Use the GPU when available. Defaults to True.
        allow_tf32: Enable TF32 matmul precision on Ampere+ GPUs. Defaults to False.
        out_path: Output SDF path. Defaults to ``<input_stem>_<model>_E.sdf`` next
            to the input file.
        overwrite: Allow writing over an existing output file. Defaults to
            True, which is the historical behavior every Python-API caller
            was written against. ``auto3d energy`` passes False unless
            ``--force`` is given, so the CLI refuses to clobber.

    Returns:
        Path to output SDF file with energies.

    Raises:
        InputValidationError: if no record of ``path`` is usable (unparseable,
            conformerless, implicit hydrogens, or dummy atoms); see the
            warnings logged for each record.
    """
    # Every guard, the output name and the record read, in one call shared with
    # opt_geometry and calc_thermo -- see Auto3D.entry._run_setup for the step
    # list, the order, and why each step sits where it does. All of it happens
    # before create_model below: nothing is downloaded or loaded for a run that
    # is about to be refused.
    setup = prepare_single_file_run(
        path,
        model_name,
        gpu_idx=gpu_idx,
        use_gpu=use_gpu,
        allow_tf32=allow_tf32,
        out_path=out_path,
        overwrite=overwrite,
        tag="E",
    )
    # calc_spe computes over the whole set at once (one padded batch), so a file
    # with no usable record is an input error, not an empty output file: this
    # used to write a 0-byte SDF and return its path with exit code 0.
    setup.require_records(path)
    mols = setup.records
    outpath = Path(setup.out_path)
    device = setup.device

    # Use ModelFactory to create model adapter
    model_adapter = create_model(model_name, device)

    # Create EnForce_ANI wrapper for batched forward support (new API without name)
    model = EnForce_ANI(model_adapter)

    # Use new vectorized padding that returns tensors directly. The explicit
    # atom mask is forwarded to the model: an adapter that flattens a padded
    # batch (AIMNet2) must be told which slots are real rather than inferring
    # it from `species == species_pad`, which deletes a legitimate atomic
    # number 0 (an R-group `*` atom) along with the padding (audit C13).
    # One argument, one source: the adapter supplies the species convention AND
    # both pad values. This call used to hand over `model_name` alongside the
    # adapter's two pads, so the remap and the sentinel came from different
    # places and could contradict each other (audit C3/C4).
    coord_padded, numbers_padded, charges, atom_mask = pad_from_mols(mols, model_adapter, device)

    # Energies only. This used to call `forward_batched` and discard the forces
    # it returned, so every single-point energy ran a full backward pass for a
    # tensor nobody read (audit M39). `energy_batched` splits into exactly the
    # same sub-batches -- memory behavior is unchanged -- and goes through
    # `ModelAdapter.energy`, which is energy-only.
    #
    # NOT bucketed by molecule size, deliberately. `pad_from_mols` pads to the
    # global max atom count, so one large record widens the padded tensor for
    # every other one, and `optimizing._make_buckets` already solves exactly that
    # for the optimizer. Reusing it here was considered and declined: the only
    # engines that pay for padded slots at all are the ANI ones (AIMNet2 flattens
    # to real atoms before the model sees them), `forward_batched` already caps
    # memory by ATOM count so a wide batch costs throughput rather than an OOM,
    # and this repository has no way to measure the difference -- CI is CPU-only
    # and no NNP may be loaded here. An unmeasured optimization that also has to
    # re-thread the writer loop's index alignment is not worth the change.
    es = model.energy_batched(coord_padded, numbers_padded, charges, atom_mask=atom_mask)
    es = es.to("cpu").detach().numpy()

    with Chem.SDWriter(str(outpath)) as f:
        for i, mol in enumerate(mols):
            set_e_hartree_from_ev(mol, float(es[i]))
            f.write(mol)
    return str(outpath)
