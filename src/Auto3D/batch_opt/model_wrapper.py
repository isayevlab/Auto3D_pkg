# src/Auto3D/batch_opt/model_wrapper.py
"""Model wrapper providing batched inference for NNP models.

This module contains the EnForce_ANI class which wraps model adapters
and provides batched forward functionality for calculating energies and forces.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from Auto3D.exceptions import OptimizationError

# The CONTRACT, not the construction layer. This module used to name
# `Auto3D.model_factory.BaseModelAdapter` -- the numerical layer reaching up into
# the factory for a type that is actually defined below it, and reaching for the
# implementation base class rather than the interface. A runtime (not
# TYPE_CHECKING) import because the gate below consults it; contract.py costs
# nothing beyond torch, which this module already imports.
from Auto3D.models.contract import ModelAdapter, missing_adapter_members


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
        model_adapter: ModelAdapter,
        batchsize_atoms: int = 1024 * 16,
    ) -> None:
        """Initialize EnForce_ANI wrapper.

        Args:
            model_adapter: An object satisfying
                :class:`Auto3D.models.contract.ModelAdapter`. Checked here --
                this is the one seam in Auto3D where that Protocol is
                load-bearing.
            batchsize_atoms: Maximum number of atoms per batch (default: 16384).

        Raises:
            TypeError: ``model_adapter`` is missing contract members, or
                ``batchsize_atoms`` is not an int.

        The second parameter used to be ``name_or_batchsize: str | int | None``,
        type-switched between a model name (the pre-adapter API) and a batch size.
        Passing a string warned that it would be "removed in Auto3D v2.0"; the
        package reached 3.0.0 with it still in place, and no caller in ``src/``
        ever passed one. Removed, so the parameter has one meaning.
        """
        super().__init__()
        # What this catches is a CATEGORY ERROR -- a raw nn.Module, a third-party
        # calculator, a leftover engine-name string -- named here instead of
        # surfacing as an AttributeError several frames deep inside
        # forward_batched. It is NOT a contract check: a presence test cannot see
        # arity, so an object with a wrong-signature forward still passes, and a
        # MagicMock passes trivially (the unit tests rely on that). Do not
        # oversell it in this message; the next reader will believe it.
        #
        # The missing names are computed from the Protocol rather than trusting
        # the bare isinstance boolean, so widening ModelAdapter widens this
        # message in the same edit. Never use issubclass against ModelAdapter --
        # it raises TypeError for any Protocol with data members.
        missing = missing_adapter_members(model_adapter)
        if missing:
            raise TypeError(
                f"EnForce_ANI needs a model adapter satisfying "
                f"Auto3D.models.contract.ModelAdapter; "
                f"{type(model_adapter).__name__} is missing "
                f"{', '.join(missing)}. Build one with "
                f"Auto3D.model_factory.create_model."
            )
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
            # Process slices of molecules; on CUDA OOM, free the cache and retry
            # the failing slice with a halved batch. A single molecule that still
            # OOMs cannot be split further (an NNP needs the whole molecule), so
            # raise a clear, actionable error instead of crashing opaquely.
            #
            # `bsize` is a local variable that SHRINKS and STAYS SHRUNK for the
            # rest of this call once an OOM is observed (audit M37): the old
            # code recursed on just the failing slice at a halved size but kept
            # splitting every OTHER remaining slice at the original `bsize`,
            # repeating the same OOM-and-recurse cycle for each one. A `while`
            # loop over `remaining` (re-queuing the failed slice at the front
            # instead of recursing) makes the smaller size the new default for
            # everything that has not run yet.
            remaining = batch_idx
            while remaining.numel() > 0:
                sub, remaining = remaining[:bsize], remaining[bsize:]
                oom = False
                try:
                    _e, _f = self(
                        coord[sub], numbers[sub], charges[sub],
                        atom_mask=None if atom_mask is None else atom_mask[sub],
                    )
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if oom:
                    # empty_cache() and the retry run AFTER the except block,
                    # not inside it: while an `except` clause is executing, the
                    # OOM'd exception is still `sys.exc_info()`'s "currently
                    # handled exception", and its traceback keeps every local
                    # of the failed forward -- including its activations --
                    # reachable. empty_cache() can only release already-free
                    # blocks, so calling it there could not reclaim the memory
                    # the retry needed (audit M37). CPython clears the
                    # currently-handled exception as soon as the `except`
                    # clause is left, which happens above (no `continue`/
                    # `raise` inside it), so by the time control reaches here
                    # nothing from the failed forward is still referenced.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if sub.numel() == 1:
                        raise OptimizationError(
                            f"A single molecule with {N} atoms exhausted GPU memory even "
                            f"at batch size 1. Reduce batchsize_atoms or use a smaller model."
                        )
                    bsize = max(1, sub.numel() // 2)
                    remaining = torch.cat([sub, remaining])
                    continue
                e_list.append(_e)
                f_list.append(_f)

        _run(idx, batch_size)

        return torch.cat(e_list, dim=0), torch.cat(f_list, dim=0)
