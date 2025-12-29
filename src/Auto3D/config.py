"""
Configuration classes for Auto3D.

This module provides typed configuration using dataclasses and Protocols
for better type safety and IDE support.
"""

from dataclasses import dataclass, field
from typing import Protocol, TypedDict, Iterator, runtime_checkable
import torch


@dataclass
class Auto3DOptions:
    """Configuration options for Auto3D conformer generation.

    This dataclass replaces the legacy `options()` function, providing
    proper type hints and validation.

    Example:
        >>> from Auto3D import Auto3DOptions, main
        >>> config = Auto3DOptions(path="input.smi", k=5)
        >>> result = main(config)
    """

    # Input/Output
    path: str | None = None
    """Path to input .smi or .sdf file containing SMILES/molecules."""

    k: int | bool = False
    """Output top-k structures for each SMILES. Set to int or False."""

    window: float | bool = False
    """Output structures within x kcal/mol of lowest energy. Set to float or False."""

    verbose: bool = False
    """When True, save all metadata while running."""

    job_name: str = ""
    """Folder name to save all metadata."""

    # Tautomer settings
    enumerate_tautomer: bool = False
    """When True, enumerate tautomers for the input."""

    tauto_engine: str = "rdkit"
    """Program to enumerate tautomers: 'rdkit' or 'oechem'."""

    pKaNorm: bool = True
    """Normalize ionization state to pH ~7.4 (only with tauto_engine='oechem')."""

    # Isomer settings
    isomer_engine: str = "rdkit"
    """Program for generating 3D isomers: 'rdkit' or 'omega'."""

    enumerate_isomer: bool = True
    """When True, enumerate cis/trans and R/S isomers."""

    mode_oe: str = "classic"
    """Omega mode: 'classic', 'macrocycle', 'dense', 'pose', 'rocs', or 'fast_rocs'."""

    mpi_np: int = 4
    """Number of CPU cores for isomer generation."""

    max_confs: int | None = None
    """Maximum conformers per SMILES. None uses dynamic number (num_heavy_atoms - 1)."""

    # GPU settings
    use_gpu: bool = True
    """Use GPU when available."""

    gpu_idx: int | list[int] = 0
    """GPU device index or list of indices."""

    capacity: int = 42
    """Number of SMILES handled per 1GB memory."""

    # Optimization settings
    optimizing_engine: str = "AIMNET"
    """Model for optimization: 'ANI2x', 'ANI2xt', 'AIMNET', or path to custom NNP."""

    patience: int = 1000
    """Drop conformer from optimization if force doesn't decrease for this many steps."""

    opt_steps: int = 5000
    """Maximum optimization steps per structure."""

    convergence_threshold: float = 0.003
    """Optimization converges when max force is below this (eV/Å)."""

    threshold: float = 0.3
    """RMSD threshold for duplicate removal (Å)."""

    memory: int | None = None
    """RAM size assigned to Auto3D in GB. None for automatic detection."""

    batchsize_atoms: int = 1024
    """Number of atoms per optimization batch per GB."""

    def __post_init__(self):
        """Normalize string values to lowercase."""
        self.tauto_engine = self.tauto_engine.lower()
        self.isomer_engine = self.isomer_engine.lower()
        self.mode_oe = self.mode_oe.lower()

    def __getitem__(self, key: str):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow dict-like assignment for backward compatibility."""
        setattr(self, key, value)

    def get(self, key: str, default=None):
        """Dict-like get method for backward compatibility."""
        return getattr(self, key, default)

    def keys(self):
        """Return field names for backward compatibility."""
        return [f.name for f in self.__dataclass_fields__.values()]

    def items(self):
        """Return field name-value pairs for backward compatibility."""
        return [(k, getattr(self, k)) for k in self.keys()]


class ChunkMeta(TypedDict):
    """Metadata for a processing chunk."""

    output: str
    """Path to final output SDF."""

    optimized_og: str
    """Path to optimized SDF before ranking."""

    output_taut: str
    """Path to tautomer output SMI."""

    smiles_enumerated: str
    """Path to enumerated SMILES file."""

    smiles_reduced: str
    """Path to reduced SMILES file."""

    smiles_hashed: str
    """Path to hashed SMILES file."""

    enumerated_sdf: str
    """Path to enumerated SDF file."""

    sorted_sdf: str
    """Path to sorted SDF file."""

    housekeeping_folder: str
    """Path to housekeeping folder."""

    path: str
    """Input path for this chunk."""

    dir: str
    """Working directory for this chunk."""


@runtime_checkable
class NNPModel(Protocol):
    """Protocol for Neural Network Potential models.

    Custom NNP models must implement this interface to be compatible
    with Auto3D's optimization engine.

    Attributes:
        coord_pad: Padding value for coordinates (typically 0).
        species_pad: Padding value for species (typically -1).

    Example:
        >>> class MyNNP(torch.nn.Module):
        ...     coord_pad = 0
        ...     species_pad = -1
        ...
        ...     def forward(self, species, coords, charges):
        ...         # Return energies tensor of shape (batch_size,)
        ...         return energies
    """

    coord_pad: int
    """Padding value for coordinates in batched tensors."""

    species_pad: int
    """Padding value for atomic species in batched tensors."""

    def forward(
        self,
        species: torch.Tensor,
        coords: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate energies for a batch of molecules.

        Args:
            species: Atomic numbers, shape (batch_size, max_atoms).
            coords: Atomic coordinates, shape (batch_size, max_atoms, 3).
            charges: Molecular charges, shape (batch_size,).

        Returns:
            Energies tensor of shape (batch_size,) in eV.
        """
        ...


# Type alias for backward compatibility
OptionsDict = Auto3DOptions
