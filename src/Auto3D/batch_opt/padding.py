# src/Auto3D/batch_opt/padding.py
"""Vectorized padding operations for molecular batches.

This module provides efficient padding functions for preparing molecular data
for batch processing with neural network potentials. The functions replace
the inefficient loop-based padding_coords and padding_species functions with
vectorized PyTorch operations.
"""
from __future__ import annotations

import torch


def pad_from_mols(
    mols: list,  # List of RDKit Mol objects
    model_name: str,
    device: torch.device,
    coord_pad: float = 0.0,
    species_pad: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad molecular data directly from RDKit Mol objects.

    Builds the padded coordinate, species, charge, and atom-mask tensors in a
    single pass, avoiding intermediate per-molecule list creation.

    Args:
        mols: List of RDKit Mol objects with conformers.
        model_name: Name of the model ('AIMNET', 'ANI2x', 'ANI2xt', or custom).
            Affects how atomic numbers are mapped to species indices.
        device: Target device for tensors (CPU or CUDA).
        coord_pad: Padding value for coordinates. Default 0.0.
        species_pad: Padding value for species. Default -1.

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
        (batch_size, max_atoms, 3),
        coord_pad,
        dtype=torch.float32,
        device=device
    )
    species_tensor = torch.full(
        (batch_size, max_atoms),
        species_pad,
        dtype=torch.long,
        device=device
    )
    atom_mask = torch.zeros(
        (batch_size, max_atoms), dtype=torch.bool, device=device
    )
    charges = []

    # Fill in actual values - create tensors directly on target device
    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(
            conf.GetPositions(), dtype=torch.float32, device=device
        )

        from Auto3D.batch_opt.species import to_model_species

        spec = to_model_species([a.GetAtomicNum() for a in mol.GetAtoms()], model_name)
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long, device=device)
        atom_mask[i, :n] = True

        charges.append(rdmolops.GetFormalCharge(mol))

    # Float (not long) to match the ASE Calculator path (ASE/thermo.py) and the
    # dtype the AIMNet2 adapter casts charges to internally; formal charges are
    # integers exactly representable in float32, and ANI models ignore charge.
    charges_tensor = torch.tensor(charges, dtype=torch.float32, device=device)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor, atom_mask
