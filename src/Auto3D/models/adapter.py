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
from Auto3D.models.loading import load_custom_nnp


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
    # This runs on every NN forward, i.e. every FIRE step. Each `.any()`/`.item()`
    # on a CUDA tensor is a host-device sync that serializes the stream, so the
    # happy path (finite outputs) does a SINGLE combined reduction. The detailed,
    # additionally-synchronizing NaN/Inf breakdown is computed only on the rare
    # failure branch, where one extra sync is irrelevant.
    if bool(torch.isfinite(energy).all() & torch.isfinite(forces).all()):
        return

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
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms and False
                for padded slots, as returned by
                :func:`Auto3D.batch_opt.padding.pad_from_mols`. Required from
                any caller that passes a PADDED batch; ``None`` means every
                slot holds a real atom. An adapter must never re-derive this by
                comparing ``species`` against ``species_pad`` (audit C13).

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
        species_pad: int = -1,
        compile_model: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            model: The underlying neural network model.
            device: Target device for computations.
            coord_pad: Fill value for unused coordinate slots.
            species_pad: Fill value for unused species slots. Defaults to -1 to
                agree with ``Auto3D.batch_opt.padding.pad_from_mols`` and with
                the documented custom-NNP convention; the previous default of 0
                collided with ANI2xt's hydrogen index, so the two layers
                disagreed about which slots were padding. Every adapter below
                passes this explicitly, so the default only applies to
                third-party subclasses -- for which -1 is the safe value,
                because it can never be a real atomic number or a 0-based
                species index.
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
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms and False
                for padded slots, threaded through from
                :func:`Auto3D.batch_opt.padding.pad_from_mols`. ``None`` means
                "every slot is a real atom" and is correct only for an
                unpadded batch. Subclasses that need to know which slots are
                padding must use THIS mask, never a comparison against
                ``self.species_pad`` (audit C13).

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
    ) -> None:
        """Initialize the AIMNet2 adapter.

        Args:
            model_name: aimnet registry name/alias.
            device: Target device.
            compile_model: Forwarded to AIMNet2Calculator (torch.compile).
        """
        from aimnet.calculators import AIMNet2Calculator

        if device is None:
            device = torch.device("cpu")
        self.model_name = model_name
        calc = AIMNet2Calculator(model_name, device=device, compile_model=compile_model)
        super().__init__(calc.model, device, coord_pad=0.0, species_pad=0, compile_model=False)
        self._calc = calc

    @property
    def calculator(self):
        """Return the underlying AIMNet2Calculator.

        Exposed for callers (e.g. ``Auto3D.ASE.thermo._load_hessian_model``)
        that need the calculator itself rather than this adapter's
        ``(coords, species, charges) -> (energy, forces)`` forward()
        interface -- e.g. to reach the calculator's native analytic Hessian,
        which includes the external D3 dispersion and Coulomb terms that
        differentiating the bare ``.model`` would drop.
        """
        return self._calc

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies (eV) and forces (eV/A) for a padded batch.

        Args:
            coords: (batch, n_atoms, 3); padded slots at coord_pad.
            species: atomic numbers (batch, n_atoms); padded slots at
                species_pad. The VALUE is never inspected here -- see below.
            charges: molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms, as
                returned by :func:`Auto3D.batch_opt.padding.pad_from_mols`.
                Required for a padded batch. ``None`` means every slot is a
                real atom, which is what the unpadded single-molecule callers
                (``ASE/thermo.py``'s ASE Calculator, ``auto3d models test``)
                want.

        Returns:
            (energy[batch], forces[batch, n_atoms, 3]) in eV and eV/A. Padded
            atom slots have zero force.

        The real-atom mask is the caller's explicit ``atom_mask``, NEVER
        ``species != self.species_pad``. This adapter's ``species_pad`` is 0
        and it consumes raw atomic numbers, so the sentinel comparison deleted
        atomic number 0 -- an R-group/dummy ``*`` atom, which
        ``utils.validation._requires_aimnet`` routes to precisely this engine.
        For ``*CCO`` the padder reported 9 real atoms and this adapter scored
        8: the energy belonged to a different species, and the dummy atom got
        exactly zero force and stayed frozen for the whole optimization. That
        is the collision class ``padding.pad_from_mols`` documents (audit C13).
        """
        b, n = species.shape[0], species.shape[1]
        if atom_mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=species.device)
        else:
            mask = atom_mask.to(device=species.device, dtype=torch.bool)
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
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using ANI2xt.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Indexed atomic species (batch, n_atoms).
            charges: Molecular charges (batch,) - not used by ANI2xt.
            atom_mask: Accepted for interface uniformity and deliberately
                unused: ANI consumes ``species_pad = -1`` as its own dummy-atom
                index, so the model itself skips padded slots. -1 is not a
                sentinel this adapter compares against, and it can never
                collide with a real 0-based species index.

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
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using ANI2x.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,) - not used by ANI2x.
            atom_mask: Accepted for interface uniformity and deliberately
                unused: torchani consumes ``species_pad = -1`` as its own
                dummy-atom index, which can never collide with a real atomic
                number.

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

    Custom models implement the contract defined in
    ``Auto3D.models.contract`` (:class:`~Auto3D.models.contract.CustomNNP`):
    - ``forward(species, coords, charges) -> energies`` -- species FIRST, and
      energies only. This adapter derives forces from the returned energy by
      autograd, so the model must not return them.
    - ``coord_pad`` and ``species_pad`` attributes. Both are REQUIRED; a missing
      one is rejected at load rather than silently defaulted.

    Note the argument order is the reverse of this adapter's own
    ``forward(coords, species, charges)``, which is Auto3D's internal
    :class:`ModelAdapter` interface and returns ``(energies, forces)``.
    ``load_custom_nnp`` rejects a model that confuses the two.

    The model file may be EITHER a TorchScript archive
    (``torch.jit.script(m).save(path)``) OR an eager nn.Module saved with
    ``torch.save(m, path)``; the adapter auto-detects. Eager loading is required
    because modern AIMNet2-based models are no longer torch.jit.script-able.

    Note: if your model pads batches, use a non-zero ``species_pad`` -- some
    backends (e.g. AIMNet2) produce NaN on species-0 padded atoms.

    Note: Custom models have limited torch.compile() benefits.

    Note: inputs are cast to float32 before the forward pass. If your NNP
    requires float64 precision (e.g. for very small energy differences),
    wrap it to upcast internally, as Auto3D will feed it float32 coordinates.
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
        # Accept either a TorchScript archive or an eager nn.Module checkpoint
        # (shared load contract -- see Auto3D.models.loading.load_custom_nnp).
        # load_custom_nnp validates the contract, so coord_pad/species_pad are
        # guaranteed present here. Reading them directly (rather than through
        # getattr defaults that disagreed with BaseModelAdapter's) is what keeps
        # one padding value in play instead of two.
        model = load_custom_nnp(model_path, device)
        # TorchScript models don't benefit from torch.compile
        super().__init__(
            model, device, model.coord_pad, model.species_pad, compile_model=False
        )

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using custom model.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers or indexed species (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Accepted for interface uniformity and NOT forwarded:
                the published custom-NNP contract
                (:class:`Auto3D.models.contract.CustomNNP`) is
                ``forward(species, coords, charges)``, so a user model
                identifies its own padding from the ``species_pad`` value it
                declared. Choose a ``species_pad`` that cannot collide with a
                real species index (-1 is always safe); see the class
                docstring.

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        # Intentional downcast to float32 for compatibility with most NNP
        # models (e.g., ANI2x). This silently loses precision for fp64 models;
        # such models should upcast internally (see class docstring).
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
