# src/Auto3D/batch_opt/padding.py
"""Vectorized padding operations for molecular batches.

This module provides efficient padding functions for preparing molecular data
for batch processing with neural network potentials. The functions replace
the inefficient loop-based padding_coords and padding_species functions with
vectorized PyTorch operations.
"""
from __future__ import annotations

from typing import Sequence

import torch


def pad_molecular_batch(
    coords: Sequence[Sequence[tuple[float, float, float]]],
    species: Sequence[Sequence[int]],
    charges: Sequence[int],
    device: torch.device,
    coord_pad: float = 0.0,
    species_pad: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized padding of molecular data.

    Efficiently pads coordinate and species lists to create uniform tensors
    for batch processing. Pre-allocates tensors with padding values and fills
    in actual molecular data.

    Args:
        coords: List of coordinate lists, each inner list has (x, y, z) tuples
            representing atomic positions in Angstroms.
        species: List of atomic number lists (or mapped indices for ANI2xt).
        charges: List of molecular charges (integers).
        device: Target device for tensors (CPU or CUDA).
        coord_pad: Padding value for coordinates. Default 0.0.
        species_pad: Padding value for species. Default -1 (convention for
            masked atoms in TorchANI models).

    Returns:
        Tuple of (coords_tensor, species_tensor, charges_tensor) where:
        - coords_tensor: Shape (batch, max_atoms, 3), dtype float32, requires_grad=True
        - species_tensor: Shape (batch, max_atoms), dtype long
        - charges_tensor: Shape (batch,), dtype long
    """
    batch_size = len(coords)
    max_atoms = max(len(s) for s in species)

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
    charges_tensor = torch.tensor(charges, dtype=torch.long, device=device)

    # Fill in actual values
    for i, (coord, spec) in enumerate(zip(coords, species)):
        n = len(spec)
        coords_tensor[i, :n] = torch.tensor(coord, dtype=torch.float32)
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor


def pad_from_mols(
    mols: list,  # List of RDKit Mol objects
    model_name: str,
    device: torch.device,
    coord_pad: float = 0.0,
    species_pad: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad molecular data directly from RDKit Mol objects.

    More efficient than the separate mols2lists + padding_coords + padding_species
    approach, as it avoids intermediate list creation and directly builds tensors.

    Args:
        mols: List of RDKit Mol objects with conformers.
        model_name: Name of the model ('AIMNET', 'ANI2x', 'ANI2xt', or custom).
            Affects how atomic numbers are mapped to species indices.
        device: Target device for tensors (CPU or CUDA).
        coord_pad: Padding value for coordinates. Default 0.0.
        species_pad: Padding value for species. Default -1.

    Returns:
        Tuple of (coords_tensor, species_tensor, charges_tensor) where:
        - coords_tensor: Shape (batch, max_atoms, 3), dtype float32, requires_grad=True
        - species_tensor: Shape (batch, max_atoms), dtype long
        - charges_tensor: Shape (batch,), dtype long
    """
    from rdkit.Chem import rdmolops

    # ANI2xt uses mapped indices instead of atomic numbers
    ani2xt_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

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
    charges = []

    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(
            conf.GetPositions(), dtype=torch.float32
        )

        if model_name == "ANI2xt":
            spec = [ani2xt_index[a.GetAtomicNum()] for a in mol.GetAtoms()]
        else:
            spec = [a.GetAtomicNum() for a in mol.GetAtoms()]
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long)

        charges.append(rdmolops.GetFormalCharge(mol))

    charges_tensor = torch.tensor(charges, dtype=torch.long, device=device)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor
