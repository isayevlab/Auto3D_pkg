# src/Auto3D/batch_opt/model_wrapper.py
"""Model wrapper providing batched inference for NNP models.

This module contains the EnForce_ANI class which wraps model adapters
and provides batched forward functionality for calculating energies and forces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The CONTRACT, not the construction layer. This module used to name
# `Auto3D.engines.model_factory.BaseModelAdapter` -- the numerical layer reaching up into
# the factory for a type that is actually defined below it, and reaching for the
# implementation base class rather than the interface. A runtime (not
# TYPE_CHECKING) import because the gate below consults it; contract.py costs
# nothing beyond torch, which this module already imports.
from Auto3D.engines.models.adapter import validate_energies
from Auto3D.engines.models.contract import ModelAdapter, missing_adapter_members
from Auto3D.foundation.exceptions import OptimizationError


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
        >>> from Auto3D.engines.model_factory import create_model
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
                :class:`Auto3D.engines.models.contract.ModelAdapter`. Checked here --
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
                f"Auto3D.engines.models.contract.ModelAdapter; "
                f"{type(model_adapter).__name__} is missing "
                f"{', '.join(missing)}. Build one with "
                f"Auto3D.engines.model_factory.create_model."
            )
        # A caller migrating off the removed API would pass a model name here and,
        # with the union gone, silently set the batch size to a string -- surfacing
        # much later inside batching as an unrelated comparison error. Rejected on
        # the spot, naming what the parameter is now for.
        if not isinstance(batchsize_atoms, int) or isinstance(batchsize_atoms, bool):
            raise TypeError(
                "EnForce_ANI's second parameter is batchsize_atoms (an int), got "
                f"{batchsize_atoms!r}. The model-name form was removed in 3.0.0; "
                "build an adapter with Auto3D.engines.model_factory.create_model instead."
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
                :func:`Auto3D.engines.batch_opt.padding.pad_from_mols`. Forwarded to
                the adapter so it never has to re-derive padding from a
                species sentinel (audit C13). ``None`` means the batch is
                unpadded.

        Returns:
            Tuple of (energies, forces) where energies has shape (B,) and
            forces has shape (B, N, 3).
        """
        return self.model.forward(coord, numbers, charges, atom_mask=atom_mask)

    def _run_in_sub_batches(self, coord: torch.Tensor, compute) -> list:
        """Split the batch by molecule count and call ``compute`` on each slice.

        Shared by :meth:`forward_batched` and :meth:`energy_batched` so the
        OOM-recovery policy exists once. Each caller concatenates the results
        itself, because they return different numbers of tensors.

        Args:
            coord: The full padded batch, only for its ``(B, N)`` shape and
                device. Slicing is the caller's job, inside ``compute``.
            compute: ``(sub_indices) -> result``, called once per sub-batch.

        Returns:
            One entry per successful sub-batch, in molecule order.

        Raises:
            OptimizationError: A single molecule exhausted GPU memory.
        """
        B, N = coord.shape[:2]
        results: list = []
        # Ensure at least 1 molecule per batch to avoid empty batches
        remaining = torch.arange(B, device=coord.device)
        bsize = max(1, self.batchsize_atoms // N)

        # Process slices of molecules; on CUDA OOM, free the cache and retry
        # the failing slice with a halved batch. A single molecule that still
        # OOMs cannot be split further (an NNP needs the whole molecule), so
        # raise a clear, actionable error instead of crashing opaquely.
        #
        # `bsize` SHRINKS and STAYS SHRUNK for the rest of this call once an OOM
        # is observed (audit M37): the old code recursed on just the failing
        # slice at a halved size but kept splitting every OTHER remaining slice
        # at the original `bsize`, repeating the same OOM-and-recurse cycle for
        # each one. The `while` loop over `remaining` (re-queuing the failed
        # slice at the front instead of recursing) makes the smaller size the new
        # default for everything that has not run yet.
        while remaining.numel() > 0:
            sub, remaining = remaining[:bsize], remaining[bsize:]
            oom = False
            try:
                out = compute(sub)
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
            results.append(out)

        return results

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
        results = self._run_in_sub_batches(
            coord,
            lambda sub: self(
                coord[sub],
                numbers[sub],
                charges[sub],
                atom_mask=None if atom_mask is None else atom_mask[sub],
            ),
        )
        return (
            torch.cat([e for e, _f in results], dim=0),
            torch.cat([f for _e, f in results], dim=0),
        )

    def energy_batched(
        self,
        coord: torch.Tensor,
        numbers: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies only, in the same sub-batches, with no backward pass.

        :meth:`forward_batched` derives forces unconditionally, because every
        adapter's ``forward`` does. A single-point energy never reads them, so
        ``calc_spe`` used to pay for a full backward pass per sub-batch and then
        discard the result (audit M39). This routes through
        :meth:`Auto3D.engines.models.contract.ModelAdapter.energy` instead, which is
        energy-only and dtype-preserving.

        How much that saves depends on the engine, and the honest answer is
        engine-specific: ``ANI2xtAdapter``, ``ANI2xAdapter`` and
        ``CustomModelAdapter`` each skip a ``torch.autograd.grad`` call, whereas
        ``AIMNet2Adapter.energy`` is deliberately still ``forward(...)[0]`` --
        its calculator's ``forces=True`` route is the one documented to keep the
        energy connected to ``coord`` for a Hessian caller, and moving the
        default engine onto the ``forces=False`` route would change which
        external-module code path computes its energy. That is a numerical
        equality claim requiring a real model to verify, so it is not made here.

        No ``no_grad`` wrapper, for the same reason ``ModelAdapter.energy`` has
        none: ``AIMNet2Adapter.energy`` computes forces internally via autograd,
        so disabling grad here would break the default engine outright.

        Args:
            coord: Coordinates, shape (B, N, 3).
            numbers: Species in the adapter's own convention, shape (B, N).
            charges: Molecular charges, shape (B,).
            atom_mask: Boolean (B, N), True for real atoms, sliced per sub-batch
                exactly as in :meth:`forward_batched`. ``None`` means unpadded.

        Returns:
            Energies, shape (B,), in eV, **detached**. Unlike
            :meth:`Auto3D.engines.models.contract.ModelAdapter.energy`, which stays
            graph-connected for a Hessian caller, this batched wrapper drops
            the graph as each sub-batch completes so memory tracks
            ``batchsize_atoms``. A caller needing gradients must go to the
            adapter directly, as ``ASE/thermo.py``'s Hessian path does.

        Raises:
            NumericalError: A non-finite energy. ``forward``'s
                ``_validate_outputs`` was this path's only NaN gate, and
                dropping it would turn ``auto3d energy``'s exit-5 diagnosis into
                an output file full of ``nan``.
            OptimizationError: A single molecule exhausted GPU memory.
        """
        # `.detach()` per sub-batch, not once at the end: no `no_grad` wrapper
        # is possible here (AIMNet2's `energy` differentiates internally), so
        # each result arrives graph-connected, and accumulating them attached
        # held every completed sub-batch until the final `cat` -- leaving the
        # OOM-retry below nothing it could free, since `empty_cache()` cannot
        # release still-referenced blocks.
        #
        # How much that held depends on the engine, so no single claim covers
        # all four. `ANI2xtAdapter.energy`, `ANI2xAdapter.energy` and
        # `CustomModelAdapter.energy` are pure forwards with no
        # `autograd.grad`, so their saved activations stayed alive and peak
        # memory tracked the whole input rather than `batchsize_atoms`.
        # `AIMNet2Adapter.energy` routes through `forward`, whose calculator
        # differentiates with `create_graph=False` and frees its buffers before
        # returning -- so for the default engine what accumulated was grad_fn
        # nodes and the energy tensors, a real but smaller effect.
        #
        # A single-point energy has no backward pass, so nothing downstream
        # wants the graph either way.
        results = self._run_in_sub_batches(
            coord,
            lambda sub: self.model.energy(
                coord[sub],
                numbers[sub],
                charges[sub],
                atom_mask=None if atom_mask is None else atom_mask[sub],
            ).detach(),
        )
        energies = torch.cat(results, dim=0)
        validate_energies(energies)
        return energies
