# src/Auto3D/batch_opt/model_wrapper.py
"""Model wrapper providing batched inference for NNP models.

This module contains the EnForce_ANI class which wraps model adapters
and provides batched forward functionality for calculating energies and forces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from Auto3D.exceptions import OptimizationError

if TYPE_CHECKING:
    from Auto3D.model_factory import BaseModelAdapter


class EnForce_ANI(nn.Module):
    """Wrapper for model adapters with batched forward support.

    Takes in a model adapter and provides batched forward functionality
    for calculating energies and forces.

    Args:
        model_adapter: A model adapter implementing the
            forward(coords, species, charges) interface.
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
        batchsize_atoms: int = 1024 * 16,
    ) -> None:
        """Initialize EnForce_ANI wrapper.

        Args:
            model_adapter: A model adapter implementing the forward interface.
            batchsize_atoms: Maximum number of atoms per batch (default: 16384).

        The second parameter used to be ``name_or_batchsize: str | int | None``,
        type-switched between a model name (the pre-adapter API) and a batch size.
        Passing a string warned that it would be "removed in Auto3D v2.0"; the
        package reached 3.0.0 with it still in place, and no caller in ``src/``
        ever passed one. Removed, so the parameter has one meaning.
        """
        super().__init__()
        # A caller migrating off the removed API would pass a model name here and,
        # with the union gone, silently set the batch size to a string -- surfacing
        # much later inside batching as an unrelated comparison error. Rejected on
        # the spot, naming what the parameter is now for.
        if not isinstance(batchsize_atoms, int) or isinstance(batchsize_atoms, bool):
            raise TypeError(
                "EnForce_ANI's second parameter is batchsize_atoms (an int), got "
                f"{batchsize_atoms!r}. The model-name form was removed in 3.0.0; "
                "build an adapter with Auto3D.model_factory.create_model instead."
            )
        self.model = model_adapter
        self.batchsize_atoms = batchsize_atoms

    def forward(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
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
            atom_mask: Boolean (B, N), True for real atoms and False for padded
                slots, as returned by
                :func:`Auto3D.batch_opt.padding.pad_from_mols`. Forwarded to
                the adapter so it never has to re-derive padding from a
                species sentinel (audit C13). ``None`` means the batch is
                unpadded.

        Returns:
            Tuple of (energies, forces) where energies has shape (B,) and
            forces has shape (B, N, 3).
        """
        return self.model.forward(coord, numbers, charges, atom_mask=atom_mask)


    def forward_batched(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
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
            atom_mask: Boolean (B, N), True for real atoms. Sliced with the
                same molecule indices as ``coord``/``numbers``/``charges`` so
                each sub-batch's adapter call receives the mask for exactly
                its own molecules. ``None`` means the batch is unpadded.

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
                    _e, _f = self(
                        coord[sub], numbers[sub], charges[sub],
                        atom_mask=None if atom_mask is None else atom_mask[sub],
                    )
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
