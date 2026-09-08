"""Vectorized padding operations for molecular batches.

This module provides efficient padding functions for preparing molecular data
for batch processing with neural network potentials. The functions replace
the inefficient loop-based padding_coords and padding_species functions with
vectorized PyTorch operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Annotation only, and pointing DOWN the stack: batch_opt depends on
    # models/, never the reverse and never on model_factory.
    from Auto3D.engines.models.contract import ModelAdapter


def pad_from_mols(
    mols: list,  # List of RDKit Mol objects
    adapter: ModelAdapter,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad molecular data directly from RDKit Mol objects.

    Builds the padded coordinate, species, charge, and atom-mask tensors in a
    single pass, avoiding intermediate per-molecule list creation.

    Args:
        mols: List of RDKit Mol objects with conformers.
        adapter: The model this batch is being built for, satisfying
            :class:`Auto3D.engines.models.contract.ModelAdapter`. It supplies all three
            model-dependent pieces -- the species convention
            (``adapter.to_species``) and both fill values (``adapter.coord_pad``,
            ``adapter.species_pad``).

            This used to be a model-name *string* plus the two pad values as
            separate arguments, so both call sites (``SPE.py``,
            ``batch_opt/batchopt.py``) handed over a name AND an adapter's pads:
            the remap came from one source and the sentinel from another, and
            nothing structurally stopped them from contradicting each other.
            That is the C3/C4 failure class, and taking the adapter is what makes
            it impossible rather than merely absent.
        device: Target device for tensors (CPU or CUDA).

    Returns:
        Tuple of (coords_tensor, species_tensor, charges_tensor, atom_mask)
        where:
        - coords_tensor: Shape (batch, max_atoms, 3), dtype float32. Leaf, with
          ``requires_grad=False`` -- grad state is the CALLER'S to set, not
          this function's (see the comment at the return statement below).
        - species_tensor: Shape (batch, max_atoms), dtype long
        - charges_tensor: Shape (batch,), dtype float32 (see note below)
        - atom_mask: Shape (batch, max_atoms), dtype bool, True for real atoms
          and False for padded slots. Callers must use this mask to identify
          padding rather than comparing species against ``species_pad``: a
          custom NNP's ``species_pad`` value can collide with a real species
          index (e.g. Auto3D's own ANI2xt convention maps hydrogen to index 0,
          the same value some adapters use as ``species_pad``), which would
          silently zero and exclude real atoms from the force-convergence
          check (audit C13).
    """
    from rdkit.Chem import rdmolops

    batch_size = len(mols)
    max_atoms = max(mol.GetNumAtoms() for mol in mols)

    # Pre-allocate CPU tensors with padding values. Built host-side and moved
    # to `device` ONCE each, after the loop below, rather than per-molecule
    # inside it (issue #24): constructing `torch.tensor(..., device=device)`
    # for every molecule's coords AND species -- two small tensors -- was two
    # blocking host-to-device copies per molecule, each its own CUDA
    # synchronization, none of which batches with the others because the
    # destination slice differs every iteration.
    coords_tensor = torch.full((batch_size, max_atoms, 3), adapter.coord_pad, dtype=torch.float32)
    species_tensor = torch.full((batch_size, max_atoms), adapter.species_pad, dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    charges = []

    # Fill in actual values -- CPU tensors throughout this loop, no device
    # traffic yet.
    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(conf.GetPositions(), dtype=torch.float32)

        spec = adapter.to_species([a.GetAtomicNum() for a in mol.GetAtoms()])
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long)
        atom_mask[i, :n] = True

        charges.append(rdmolops.GetFormalCharge(mol))

    # Float (not long) to match the ASE Calculator path (ASE/thermo.py) and the
    # dtype the AIMNet2 adapter casts charges to internally; formal charges are
    # integers exactly representable in float32, and ANI models ignore charge.
    charges_tensor = torch.tensor(charges, dtype=torch.float32)

    # The one H2D copy per tensor promised above. A no-op (same tensor
    # returned) when `device` is CPU, which is every fast-tier test in this
    # repository -- the four transfers only exist when `device` is CUDA.
    coords_tensor = coords_tensor.to(device)
    species_tensor = species_tensor.to(device)
    atom_mask = atom_mask.to(device)
    charges_tensor = charges_tensor.to(device)

    # Grad state is deliberately NOT set here (issue #18). Both production
    # callers own it themselves and neither needed this: `ensemble_opt`
    # (batchopt.py) immediately `.detach()`es this tensor before building its
    # optimization state, and the step loop (`optimization_engine._step_active_subset`)
    # calls its own `coord.requires_grad_(True)` on the per-step subset every
    # iteration -- so the flag set here was overwritten before first use on
    # that path. `SPE.calc_spe` feeds this coords tensor straight into
    # `energy_batched` without detaching first, so setting it True here used
    # to matter there -- except `AIMNet2Adapter`'s calculator sets
    # `requires_grad_(True)` on its own copy of coord internally
    # (`aimnet.calculators.derivatives`), and the three ANI/custom adapters'
    # `energy()` build a graph if and only if the coords they are HANDED
    # already require grad. Setting it True unconditionally therefore built an
    # autograd graph for every SPE sub-batch through the full model -- for
    # ANI2x's 8-model ensemble, activations saved for a backward that
    # `energy_batched` (M39) deliberately never calls -- roughly doubling peak
    # memory for no benefit.
    return coords_tensor, species_tensor, charges_tensor, atom_mask
