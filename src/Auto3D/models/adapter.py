# src/Auto3D/models/adapter.py
"""Model adapters providing consistent interface for all NNP models."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from Auto3D.constants import HARTREE_TO_EV
from Auto3D.exceptions import NumericalError


def _try_compile(model: nn.Module, mode: str = "default") -> nn.Module:
    """Attempt to compile a model with torch.compile.

    Args:
        model: The model to compile.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
            Defaults to "default" and the model is compiled with dynamic=True.
            The optimization batch shrinks every step as conformers converge, and
            bucketing produces variable sub-batch sizes, so shapes are not static.
            "reduce-overhead" (CUDA graphs) requires static shapes and would
            trigger constant recompilation/guard failures here (review finding
            #24); dynamic default mode avoids that.

    Returns:
        Compiled model, or original model if compilation fails.
    """
    if not hasattr(torch, 'compile'):
        return model
    try:
        return torch.compile(model, mode=mode, fullgraph=False, dynamic=True)
    except Exception as e:
        warnings.warn(f"torch.compile failed, using eager mode: {e}")
        return model


def _validate_outputs(energy: torch.Tensor, forces: torch.Tensor) -> None:
    """Validate model outputs for numerical stability.

    Checks for NaN and Inf values in energy and force tensors, raising
    an exception if numerical instability is detected.

    Args:
        energy: Energy tensor from model forward pass.
        forces: Force tensor from model forward pass.

    Raises:
        NumericalError: If NaN or Inf values are detected.
    """
    if torch.isnan(energy).any():
        nan_count = torch.isnan(energy).sum().item()
        raise NumericalError(
            f"NaN detected in {nan_count} energy value(s). "
            "This may indicate problematic molecular geometries."
        )
    if torch.isinf(energy).any():
        inf_count = torch.isinf(energy).sum().item()
        raise NumericalError(
            f"Inf detected in {inf_count} energy value(s). "
            "This may indicate atomic clashes or numerical overflow."
        )
    if torch.isnan(forces).any():
        nan_count = torch.isnan(forces).sum().item()
        raise NumericalError(
            f"NaN detected in {nan_count} force component(s). "
            "This may indicate problematic molecular geometries."
        )
    if torch.isinf(forces).any():
        inf_count = torch.isinf(forces).sum().item()
        raise NumericalError(
            f"Inf detected in {inf_count} force component(s). "
            "This may indicate atomic clashes or numerical overflow."
        )


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol defining the standard interface for NNP model adapters.

    All model adapters must implement this interface to ensure consistent
    behavior across different neural network potential backends.

    This protocol is runtime_checkable, allowing isinstance() checks.
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
    - Gradient disabling for model parameters (weights are frozen)
    - Optional torch.compile() for performance optimization

    Note on torch.inference_mode():
        This class CANNOT use torch.inference_mode() or torch.no_grad() in forward
        methods because force calculations require computing gradients of energy
        with respect to atomic coordinates via torch.autograd.grad(). Model parameters
        have requires_grad=False (frozen weights), but coordinates must have
        requires_grad=True for force computation. All autograd.grad() calls use
        create_graph=False to avoid building second-order gradient graphs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        coord_pad: float = 0.0,
        species_pad: int = 0,
        compile_model: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            model: The underlying neural network model.
            device: Target device for computations.
            coord_pad: Padding value for coordinates.
            species_pad: Padding value for atomic species.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        super().__init__()
        self.device = device
        self.coord_pad = coord_pad
        self.species_pad = species_pad
        self._compiled = False

        # Disable gradients for model parameters (inference mode)
        for p in model.parameters():
            p.requires_grad_(False)

        # Optionally compile the model
        if compile_model:
            model = _try_compile(model)
            self._compiled = True

        self.model = model

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


class AIMNet2Adapter(BaseModelAdapter):
    """Adapter for AIMNet2 models served by the `aimnet` package.

    Models are resolved by registry name/alias (e.g. 'aimnet2',
    'aimnet2-2025', 'aimnet2-nse') and auto-downloaded + sha256-validated
    into ~/.cache/aimnet on first use. Supports charged molecules and the
    full AIMNet2 element set.

    The optimizer feeds a padded (B, N, 3) batch. AIMNet2 does not tolerate
    padding atoms (species 0 at the origin yields NaN), so this adapter
    flattens real atoms and uses the calculator's ragged `mol_idx` batching,
    then scatters forces back into the padded (B, N, 3) layout (padded slots
    receive zero force).
    """

    def __init__(
        self,
        model_name: str = "aimnet2",
        device: torch.device | None = None,
        compile_model: bool = False,
        use_ensemble: bool = False,
    ) -> None:
        """Initialize the AIMNet2 adapter.

        Args:
            model_name: aimnet registry name/alias.
            device: Target device.
            compile_model: Forwarded to AIMNet2Calculator (torch.compile).
            use_ensemble: Reserved; a single registry member is used in 4.0.
        """
        from aimnet.calculators import AIMNet2Calculator

        if device is None:
            device = torch.device("cpu")
        self.model_name = model_name
        self._use_ensemble = use_ensemble
        calc = AIMNet2Calculator(model_name, device=device, compile_model=compile_model)
        super().__init__(calc.model, device, coord_pad=0.0, species_pad=0, compile_model=False)
        self._calc = calc

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies (eV) and forces (eV/A) for a padded batch.

        Args:
            coords: (batch, n_atoms, 3); padded slots at coord_pad.
            species: atomic numbers (batch, n_atoms); padded slots == species_pad (0).
            charges: molecular charges (batch,).

        Returns:
            (energy[batch], forces[batch, n_atoms, 3]) in eV and eV/A. Padded
            atom slots have zero force.
        """
        b, n = species.shape[0], species.shape[1]
        mask = species != self.species_pad                     # (B, N) real-atom mask
        coord_flat = coords[mask]                              # (M, 3)
        numbers_flat = species[mask]                           # (M,)
        mol_idx = torch.arange(b, device=species.device).unsqueeze(1).expand(b, n)[mask]  # (M,)

        result = self._calc(
            {
                "coord": coord_flat,
                "numbers": numbers_flat,
                "charge": charges.to(coord_flat.dtype),
                "mol_idx": mol_idx,
            },
            forces=True,
        )
        energy = result["energy"].reshape(-1).to(torch.double)  # (B,)
        forces_flat = result["forces"].reshape(-1, 3)           # (M, 3)

        forces = torch.zeros(b, n, 3, dtype=forces_flat.dtype, device=forces_flat.device)
        forces[mask] = forces_flat
        _validate_outputs(energy, forces)
        return energy, forces


class ANI2xtAdapter(BaseModelAdapter):
    """Adapter for ANI2xt model.

    ANI2xt is a retrained version of ANI with improved performance.
    Uses indexed species (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6).

    This model benefits significantly from torch.compile() optimization.
    """

    def __init__(self, device: torch.device, compile_model: bool = False) -> None:
        """Initialize ANI2xt adapter.

        Args:
            device: Target device for computations.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
        model = ANI2xt(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1, compile_model=compile_model)

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
        # create_graph=False (default) avoids building second-order gradient graph
        grad = torch.autograd.grad([energy.sum()], [coords], create_graph=False)[0]
        forces = -grad
        _validate_outputs(energy, forces)
        return energy, forces


class ANI2xAdapter(BaseModelAdapter):
    """Adapter for ANI2x model from TorchANI.

    ANI2x uses periodic table indexing for species.
    Requires torchani to be installed.

    This model benefits significantly from torch.compile() optimization.
    """

    def __init__(self, device: torch.device, compile_model: bool = False) -> None:
        """Initialize ANI2x adapter.

        Args:
            device: Target device for computations.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        import torchani
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1, compile_model=compile_model)

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
        # create_graph=False (default) avoids building second-order gradient graph
        grad = torch.autograd.grad([energy.sum()], [coords_f32], create_graph=False)[0]
        forces = -grad

        _validate_outputs(energy, forces)
        # Convert back to input dtype for consistency
        return energy.to(input_dtype), forces.to(input_dtype)


class CustomModelAdapter(BaseModelAdapter):
    """Adapter for user-provided custom NNP models.

    Custom models should be TorchScript models that implement:
    - forward(species, coords, charges) -> energies
    - Optional: coord_pad and species_pad attributes

    If coord_pad and species_pad are not defined, defaults are used.

    Note: Custom models are typically TorchScript, which has limited
    torch.compile() benefits.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        compile_model: bool = False,
    ) -> None:
        """Initialize custom model adapter.

        Args:
            model_path: Path to the TorchScript model file.
            device: Target device for computations.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        model = torch.jit.load(model_path, map_location=device)
        coord_pad = getattr(model, 'coord_pad', 0.0)
        species_pad = getattr(model, 'species_pad', -1)
        # TorchScript models don't benefit from torch.compile
        super().__init__(model, device, coord_pad, species_pad, compile_model=False)

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

        # create_graph=False (default) avoids building second-order gradient graph
        grad = torch.autograd.grad([energy.sum()], [coords_f32], create_graph=False)[0]
        forces = -grad

        _validate_outputs(energy, forces)
        # Convert output back to input dtype for consistency
        return energy.to(input_dtype), forces.to(input_dtype)
