# src/Auto3D/batch_opt/padding.py
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
        - coords_tensor: Shape (batch, max_atoms, 3), dtype float32, requires_grad=True
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

    # Pre-allocate tensors with padding values
    coords_tensor = torch.full(
        (batch_size, max_atoms, 3), adapter.coord_pad, dtype=torch.float32, device=device
    )
    species_tensor = torch.full(
        (batch_size, max_atoms), adapter.species_pad, dtype=torch.long, device=device
    )
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=device)
    charges = []

    # Fill in actual values - create tensors directly on target device
    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(conf.GetPositions(), dtype=torch.float32, device=device)

        spec = adapter.to_species([a.GetAtomicNum() for a in mol.GetAtoms()])
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long, device=device)
        atom_mask[i, :n] = True

        charges.append(rdmolops.GetFormalCharge(mol))

    # Float (not long) to match the ASE Calculator path (ASE/thermo.py) and the
    # dtype the AIMNet2 adapter casts charges to internally; formal charges are
    # integers exactly representable in float32, and ANI models ignore charge.
    charges_tensor = torch.tensor(charges, dtype=torch.float32, device=device)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor, atom_mask
