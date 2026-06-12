# src/Auto3D/batch_opt/model_wrapper.py
"""Model wrapper providing batched inference for NNP models.

This module contains the EnForce_ANI class which wraps model adapters
and provides batched forward functionality for calculating energies and forces.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from Auto3D.exceptions import OptimizationError
from Auto3D.utils import hartree2ev

if TYPE_CHECKING:
    from Auto3D.model_factory import BaseModelAdapter


class EnForce_ANI(nn.Module):
    """Wrapper for model adapters with batched forward support.

    Takes in a model adapter and provides batched forward functionality
    for calculating energies and forces.

    Args:
        model_adapter: A model adapter implementing the forward(coords, species, charges) interface,
                       or a raw model (for backward compatibility).
        name_or_batchsize: Either a string name (deprecated old API) or an int batchsize_atoms.
        batchsize_atoms: Maximum number of atoms that can be handled in one batch.

    Examples:
        >>> # New API with model adapter
        >>> from Auto3D.model_factory import create_model
        >>> adapter = create_model("AIMNET", device)
        >>> model = EnForce_ANI(adapter)
        >>> energy, forces = model.forward(coords, species, charges)

        >>> # Batched computation for large systems
        >>> energy, forces = model.forward_batched(coords, species, charges)
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        name_or_batchsize: str | int | None = None,
        batchsize_atoms: int = 1024 * 16,
    ) -> None:
        """Initialize EnForce_ANI wrapper.

        Args:
            model_adapter: A model adapter implementing the forward interface.
            name_or_batchsize: For backward compatibility - either model name (deprecated)
                               or batchsize_atoms as int.
            batchsize_atoms: Maximum number of atoms per batch (default: 16384).
        """
        super().__init__()
        # Handle backward compatibility
        if isinstance(name_or_batchsize, str):
            # Old API: EnForce_ANI(model, name, batchsize_atoms)
            warnings.warn(
                "Passing 'name' to EnForce_ANI is deprecated and will be removed in Auto3D v2.0. "
                "Use model adapters from Auto3D.model_factory instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.add_module("ani", model_adapter)
            self.model = model_adapter
            self.name = name_or_batchsize
            self.batchsize_atoms = batchsize_atoms
            self._use_legacy_forward = True
        elif isinstance(name_or_batchsize, int):
            # New API with explicit batchsize: EnForce_ANI(model_adapter, batchsize_atoms)
            self.model = model_adapter
            self.batchsize_atoms = name_or_batchsize
            self.name = None
            self._use_legacy_forward = False
        else:
            # New API: EnForce_ANI(model_adapter) or EnForce_ANI(model_adapter, None, batchsize)
            self.model = model_adapter
            self.batchsize_atoms = batchsize_atoms
            self.name = None
            self._use_legacy_forward = False

    def forward(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the energies and forces for input molecules.

        Delegates to the model adapter's forward method, or uses legacy
        logic for backward compatibility with raw models.

        Note on torch.inference_mode():
            This method CANNOT use torch.inference_mode() because force
            calculation requires computing gradients of energy with respect
            to atomic coordinates via torch.autograd.grad(). Model parameters
            have requires_grad=False (frozen weights), but coordinates must
            have requires_grad=True for force computation.

        Args:
            coord: Coordinates for all input structures. Shape (B, N, 3), where
                  B is the number of structures, N is the number of atoms in
                  each structure, 3 represents xyz dimensions.
            numbers: The periodic numbers for all atoms. Shape (B, N).
            charges: Molecular charges. Shape (B,).

        Returns:
            Tuple of (energies, forces) where energies has shape (B,) and
            forces has shape (B, N, 3).
        """
        if self._use_legacy_forward:
            return self._legacy_forward(coord, numbers, charges)
        return self.model.forward(coord, numbers, charges)

    def _legacy_forward(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward implementation for backward compatibility.

        .. deprecated:: 1.0
            This method is deprecated and will be removed in Auto3D v2.0.
            Use model adapters from :mod:`Auto3D.model_factory` instead.

        Handles raw models that were passed with the old API.

        Note: Cannot use torch.inference_mode() because force calculation
        requires computing gradients via torch.autograd.grad(). Model parameters
        are already frozen (requires_grad=False), but coordinates must have
        requires_grad=True for force computation.

        Args:
            coord: Coordinates for all input structures.
            numbers: The periodic numbers for all atoms.
            charges: Molecular charges.

        Returns:
            Tuple of (energies, forces).
        """
        warnings.warn(
            "_legacy_forward is deprecated and will be removed in Auto3D v2.0. "
            "Use model adapters from Auto3D.model_factory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.name == "AIMNET":
            d = self.ani(dict(coord=coord, numbers=numbers, charge=charges))
            e = d["energy"].to(torch.double)
            f = d["forces"]
        elif self.name == "ANI2xt":
            e = self.ani(numbers, coord)
            # create_graph=False avoids building second-order gradient graph
            g = torch.autograd.grad([e.sum()], [coord], create_graph=False)[0]
            f = -g
        elif self.name == "ANI2x":
            e = self.ani((numbers, coord)).energies
            e = e * hartree2ev  # ANI2x output energy unit is Hartree; convert to eV
            # create_graph=False avoids building second-order gradient graph
            g = torch.autograd.grad([e.sum()], [coord], create_graph=False)[0]
            f = -g
        else:
            # user NNP that was loaded from a file
            e = self.ani(numbers, coord, charges)
            # create_graph=False avoids building second-order gradient graph
            g = torch.autograd.grad([e.sum()], [coord], create_graph=False)[0]
            f = -g
        return e, f

    def forward_batched(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the energies and forces for input molecules in batches.

        Splits large batches into smaller chunks based on batchsize_atoms
        to manage memory usage effectively.

        Args:
            coord: Coordinates for all input structures. Shape (B, N, 3), where
                  B is the number of structures, N is the number of atoms in
                  each structure, 3 represents xyz dimensions.
            numbers: The periodic numbers for all atoms. Shape (B, N).
            charges: Molecular charges. Shape (B,).

        Returns:
            Tuple of (energies, forces) concatenated across batches.
            Energies has shape (B,), forces has shape (B, N, 3).
        """
        B, N = coord.shape[:2]
        e_list: list[torch.Tensor] = []
        f_list: list[torch.Tensor] = []
        idx = torch.arange(B, device=coord.device)

        # Ensure at least 1 molecule per batch to avoid empty batches
        batch_size = max(1, self.batchsize_atoms // N)

        def _run(batch_idx: torch.Tensor, bsize: int) -> None:
            # Process a slice of molecules; on CUDA OOM, free the cache and retry
            # the failing slice with a halved batch. A single molecule that still
            # OOMs cannot be split further (an NNP needs the whole molecule), so
            # raise a clear, actionable error instead of crashing opaquely.
            for sub in batch_idx.split(bsize):
                try:
                    _e, _f = self(coord[sub], numbers[sub], charges[sub])
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if sub.numel() == 1:
                        raise OptimizationError(
                            f"A single molecule with {N} atoms exhausted GPU memory even "
                            f"at batch size 1. Reduce batchsize_atoms or use a smaller model."
                        )
                    _run(sub, max(1, sub.numel() // 2))
                    continue
                e_list.append(_e)
                f_list.append(_f)

        _run(idx, batch_size)

        return torch.cat(e_list, dim=0), torch.cat(f_list, dim=0)
