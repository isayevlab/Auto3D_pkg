#!/usr/bin/env python
"""Calculating single point energy using ANI2xt, ANI2x, 'userNNP' or AIMNET"""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from Auto3D.batch_opt.batchopt import EnForce_ANI
from Auto3D.batch_opt.padding import pad_from_mols
from Auto3D.model_factory import create_model, get_device
from Auto3D.models.preflight import resolve_engine_name
from Auto3D.torch_config import TorchConfig, configure_torch
from Auto3D.utils import hartree2ev
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.validation import check_engine_supports_molecules, check_gpu_requested

logger = get_logger(__name__)

__all__ = ["calc_spe"]

ev2hatree = 1/hartree2ev


def calc_spe(
    path: str,
    model_name: str,
    gpu_idx: int = 0,
    use_gpu: bool = True,
    allow_tf32: bool = False,
    out_path: str | None = None,
) -> str:
    """Calculates single point energy.

    Args:
        path: Input sdf file.
        model_name: AIMNET, ANI2x, userNNP, or ANI2xt.
        gpu_idx: GPU cuda index. Defaults to 0.
        use_gpu: Use the GPU when available. Defaults to True.
        allow_tf32: Enable TF32 matmul precision on Ampere+ GPUs. Defaults to False.
        out_path: Output SDF path. Defaults to ``<input_stem>_<model>_E.sdf`` next
            to the input file.

    Returns:
        Path to output SDF file with energies.
    """
    # Fail fast on an unrecognized engine name -- the same guard the CLI's
    # `energy` command already runs before calling this function
    # (cli/commands/properties.py), now also enforced for direct Python-API
    # callers. Pure offline registry lookup: no network, no model load.
    resolve_engine_name(model_name)

    # calc_spe never goes through check_input/check_valid_configuration, so
    # without this it would reach model_factory.get_device below and silently
    # fall back to CPU instead of failing the same way `auto3d energy`
    # already does at its CLI wrapper (cli/commands/properties.py) -- and the
    # same way `auto3d run`/smiles2mols do via check_input /
    # check_valid_configuration. check_gpu_requested is the single source of
    # truth for this policy; called here, before get_device/create_model
    # below, so no compute (and no model construction) happens first.
    check_gpu_requested(use_gpu)

    # Apply the shared torch configuration so allow_tf32 is honored here too
    # (this path previously ignored it).
    configure_torch(TorchConfig(allow_tf32=allow_tf32))

    # Create output path in the same directory as the input (unless overridden)
    if out_path is not None:
        outpath = Path(out_path)
    else:
        dir_path = Path(path).parent
        stem = Path(path).stem
        if Path(model_name).exists():
            basename = f"{stem}_userNNP_E.sdf"
        else:
            basename = f"{stem}_{model_name}_E.sdf"
        outpath = dir_path / basename

    # Filter once up front: drop None records (unparseable) and conformerless
    # molecules so pad_from_mols never dereferences a bad record, and so the
    # writer loop below stays index-aligned with the energies tensor. Parsing
    # `mols` needs only `path`, not a device or model, so it -- and the C11
    # guard right below, which needs only `mols`/`model_name` -- both happen
    # before get_device/create_model, matching check_gpu_requested's
    # already-first placement: every guard that can fail fast, does, before
    # any device/model construction.
    mols = []
    for i, mol in enumerate(Chem.SDMolSupplier(path, removeHs=False)):
        if mol is None:
            logger.warning(f"Skipping molecule at index {i}: failed to parse")
            continue
        if mol.GetNumConformers() == 0:
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else '<unnamed>'
            logger.warning(f"Skipping record without a conformer: {name!r}.")
            continue
        mols.append(mol)

    # If every record was dropped (all None / conformerless), pad_from_mols([])
    # would raise a cryptic "max() arg is an empty sequence". Write an empty
    # output SDF and return its path so callers get a clear signal instead.
    if not mols:
        logger.warning(
            f"No valid molecules with conformers in {path}; nothing to compute."
        )
        with Chem.SDWriter(str(outpath)):
            pass
        return str(outpath)

    # ANI2x/ANI2xt can only represent uncharged, in-set molecules (C11): a
    # charged or out-of-set species handed to either would otherwise be
    # silently evaluated as a different, neutral species by the forward pass
    # below -- tens of kcal/mol wrong, with wrong forces. Checked before
    # get_device/create_model (below) so a bad molecule fails fast, without
    # constructing a (possibly GPU-resident) model first.
    check_engine_supports_molecules(mols, model_name)

    # Use get_device from model_factory (honors use_gpu)
    device = get_device(gpu_idx, use_gpu=use_gpu)

    # Use ModelFactory to create model adapter
    model_adapter = create_model(model_name, device)

    # Create EnForce_ANI wrapper for batched forward support (new API without name)
    model = EnForce_ANI(model_adapter)

    # Use new vectorized padding that returns tensors directly. calc_spe only
    # needs energies (not forces), so the atom mask pad_from_mols also returns
    # is unused here.
    coord_padded, numbers_padded, charges, _atom_mask = pad_from_mols(
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

