# src/Auto3D/models/adapter.py
"""Model adapters providing consistent interface for all NNP models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn

from Auto3D.constants import HARTREE_TO_EV


class ModelAdapter(Protocol):
    """Protocol defining the standard interface for NNP model adapters.

    All model adapters must implement this interface to ensure consistent
    behavior across different neural network potential backends.
    """

    coord_pad: float
    species_pad: int
    device: torch.device

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).

        Returns:
            Tuple of (energies, forces) where energies has shape (batch,)
            and forces has shape (batch, n_atoms, 3). Units: eV.
        """
        ...


class BaseModelAdapter(ABC, nn.Module):
    """Base class for model adapters.

    Provides common functionality for all NNP model adapters including:
    - Model storage and device management
    - Padding value configuration
    - Gradient disabling for inference mode
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        coord_pad: float = 0.0,
        species_pad: int = 0,
    ) -> None:
        """Initialize the adapter.

        Args:
            model: The underlying neural network model.
            device: Target device for computations.
            coord_pad: Padding value for coordinates.
            species_pad: Padding value for atomic species.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.coord_pad = coord_pad
        self.species_pad = species_pad

        # Disable gradients for model parameters (inference mode)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).

        Returns:
            Tuple of (energies, forces) where energies has shape (batch,)
            and forces has shape (batch, n_atoms, 3). Units: eV.
        """
        ...


class AIMNetAdapter(BaseModelAdapter):
    """Adapter for AIMNet2 models.

    AIMNet2 models use atomic numbers directly and support charged molecules.
    The default model is the wb97m ensemble force model.
    """

    def __init__(self, device: torch.device, model_path: Path | None = None) -> None:
        """Initialize AIMNet adapter.

        Args:
            device: Target device for computations.
            model_path: Optional path to model file. Defaults to built-in model.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "aimnet2_wb97m_ens_f.jpt"

        model = torch.jit.load(str(model_path), map_location=device)
        super().__init__(model, device, coord_pad=0.0, species_pad=0)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using AIMNet2.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        result = self.model(dict(coord=coords, numbers=species, charge=charges))
        energy = result['energy'].to(torch.double)
        forces = result['forces']
        return energy, forces


class ANI2xtAdapter(BaseModelAdapter):
    """Adapter for ANI2xt model.

    ANI2xt is a retrained version of ANI with improved performance.
    Uses indexed species (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6).
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize ANI2xt adapter.

        Args:
            device: Target device for computations.
        """
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
        model = ANI2xt(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using ANI2xt.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Indexed atomic species (batch, n_atoms).
            charges: Molecular charges (batch,) - not used by ANI2xt.

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        coords = coords.requires_grad_(True)
        energy = self.model(species, coords)
        grad = torch.autograd.grad([energy.sum()], [coords])[0]
        forces = -grad
        return energy, forces


class ANI2xAdapter(BaseModelAdapter):
    """Adapter for ANI2x model from TorchANI.

    ANI2x uses periodic table indexing for species.
    Requires torchani to be installed.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize ANI2x adapter.

        Args:
            device: Target device for computations.
        """
        import torchani
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using ANI2x.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,) - not used by ANI2x.

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        # Convert to float32 for ANI2x (it uses float32 internally)
        input_dtype = coords.dtype
        coords_f32 = coords.float().requires_grad_(True)

        energy = self.model((species, coords_f32)).energies * HARTREE_TO_EV
        grad = torch.autograd.grad([energy.sum()], [coords_f32])[0]
        forces = -grad

        # Convert back to input dtype for consistency
        return energy.to(input_dtype), forces.to(input_dtype)


class CustomModelAdapter(BaseModelAdapter):
    """Adapter for user-provided custom NNP models.

    Custom models should be TorchScript models that implement:
    - forward(species, coords, charges) -> energies
    - Optional: coord_pad and species_pad attributes

    If coord_pad and species_pad are not defined, defaults are used.
    """

    def __init__(self, model_path: str, device: torch.device) -> None:
        """Initialize custom model adapter.

        Args:
            model_path: Path to the TorchScript model file.
            device: Target device for computations.
        """
        model = torch.jit.load(model_path, map_location=device)
        coord_pad = getattr(model, 'coord_pad', 0.0)
        species_pad = getattr(model, 'species_pad', -1)
        super().__init__(model, device, coord_pad, species_pad)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using custom model.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers or indexed species (batch, n_atoms).
            charges: Molecular charges (batch,).

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        # Convert to float32 for compatibility with most NNP models (e.g., ANI2x)
        input_dtype = coords.dtype
        coords_f32 = coords.float().requires_grad_(True)
        charges_f32 = charges.float()

        energy = self.model(species, coords_f32, charges_f32)

        grad = torch.autograd.grad([energy.sum()], [coords_f32])[0]
        forces = -grad

        # Convert output back to input dtype for consistency
        return energy.to(input_dtype), forces.to(input_dtype)
