# src/Auto3D/batch_opt/padding.py
"""Vectorized padding operations for molecular batches.

This module provides efficient padding functions for preparing molecular data
for batch processing with neural network potentials. The functions replace
the inefficient loop-based padding_coords and padding_species functions with
vectorized PyTorch operations.
"""
from __future__ import annotations

from collections.abc import Sequence

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
    # Float (not long) for parity with pad_from_mols and the ASE Calculator, and
    # to match the dtype the AIMNet2 adapter casts charges to (ANI ignores them).
    charges_tensor = torch.tensor(charges, dtype=torch.float32, device=device)

    # Fill in actual values - create tensors directly on target device
    for i, (coord, spec) in enumerate(zip(coords, species, strict=True)):
        n = len(spec)
        coords_tensor[i, :n] = torch.tensor(coord, dtype=torch.float32, device=device)
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long, device=device)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor


def pad_from_mols(
    mols: list,  # List of RDKit Mol objects
    model_name: str,
    device: torch.device,
    coord_pad: float = 0.0,
    species_pad: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad molecular data directly from RDKit Mol objects.

    Builds the padded coordinate, species, and charge tensors in a single pass,
    avoiding intermediate per-molecule list creation.

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

    from Auto3D.utils.chemistry import getidx

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

    # Fill in actual values - create tensors directly on target device
    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(
            conf.GetPositions(), dtype=torch.float32, device=device
        )

        if model_name == "ANI2xt":
            # getidx raises a friendly ValueError naming the element + model
            # for atoms ANI2xt does not support (anything outside H,C,N,O,F,S,Cl).
            spec = [getidx(a.GetAtomicNum(), model="ANI2xt") for a in mol.GetAtoms()]
        else:
            spec = [a.GetAtomicNum() for a in mol.GetAtoms()]
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long, device=device)

        charges.append(rdmolops.GetFormalCharge(mol))

    # Float (not long) to match the ASE Calculator path (ASE/thermo.py) and the
    # dtype the AIMNet2 adapter casts charges to internally; formal charges are
    # integers exactly representable in float32, and ANI models ignore charge.
    charges_tensor = torch.tensor(charges, dtype=torch.float32, device=device)

    return coords_tensor.requires_grad_(True), species_tensor, charges_tensor
