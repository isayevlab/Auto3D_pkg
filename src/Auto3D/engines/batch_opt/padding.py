"""Vectorized padding operations for molecular batches.

This module provides efficient padding functions for preparing molecular data
for batch processing with neural network potentials. The functions replace
the inefficient loop-based padding_coords and padding_species functions with
vectorized PyTorch operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    # Annotation only, and pointing DOWN the stack: batch_opt depends on
    # models/, never the reverse and never on model_factory.
    from Auto3D.engines.models.contract import ModelAdapter


class PaddedBatch(NamedTuple):
    """What :func:`pad_from_mols` returns.

    A ``tuple``, so ``coords, species, charges, atom_mask = pad_from_mols(...)``
    keeps working and every existing caller is unchanged; the names are for the
    callers that slice or forward the batch rather than consume it whole. Four
    loose tensors that must stay aligned on their leading axis is a shape that
    invites one of them to be indexed differently from the rest -- and the one
    most easily left behind is ``atom_mask``, the tensor whose whole purpose is
    to say which slots are real atoms (audit C13). :meth:`sub` indexes all four
    together so a sub-batch cannot be built half-sliced.
    """

    coords: torch.Tensor  # (B, N, 3) float32, leaf, requires_grad False
    species: torch.Tensor  # (B, N) long
    charges: torch.Tensor  # (B,) float32
    atom_mask: torch.Tensor  # (B, N) bool, True for real atoms

    @property
    def n_mols(self) -> int:
        """``B`` -- the number of molecules, the length of the leading axis."""
        return self.coords.shape[0]

    def sub(self, index: slice | torch.Tensor | Sequence[int]) -> PaddedBatch:
        """The same batch restricted to ``index`` over the molecule axis.

        Args:
            index: Anything that indexes a leading axis of length ``n_mols`` and
                KEEPS that axis -- a slice, a bool tensor of that length, or a
                tensor/list of molecule indices. A scalar is refused rather than
                accepted and reinterpreted; see ``Raises``.

        Returns:
            A :class:`PaddedBatch` whose four fields were all indexed with the
            SAME ``index``. ``charges`` is one-dimensional and the other three
            are not, which is exactly why writing the four index expressions out
            by hand at a call site is worth removing.

            A slice returns four VIEWS sharing storage with this batch, while a
            bool mask or an index tensor copies (that is PyTorch's rule for basic
            versus advanced indexing, not a choice made here), so an in-place
            update of a slice-derived sub-batch -- the shape the FIRE loop's
            coordinate step has -- also writes through to the parent, and to any
            other sub-batch overlapping it. Uniformity is deliberately not forced:
            cloning would double the memory of the tensor-indexed form, which
            already copies.

        Raises:
            TypeError: ``index`` is a scalar (an ``int``, a ``bool`` or a 0-d
                tensor) or a tuple. A scalar indexes the molecule axis AWAY
                instead of restricting it, which would hand back a ``PaddedBatch``
                of ``(N, 3)``/``(N,)``/``()``/``(N,)`` tensors whose ``n_mols``
                then reports the atom count -- confidently wrong, and first
                visible somewhere downstream. A tuple is multi-axis indexing,
                which this single-axis method never means.
        """
        if isinstance(index, int | bool) or (torch.is_tensor(index) and index.dim() == 0):
            raise TypeError(
                "PaddedBatch.sub needs a slice, a bool tensor, or an index tensor "
                "over the molecule axis; a scalar would drop it "
                "(use sub(slice(i, i + 1)))"
            )
        if isinstance(index, tuple):
            raise TypeError(
                "PaddedBatch.sub takes one index over the molecule axis, not a "
                "tuple of axes; pass a list or tensor of molecule indices."
            )
        return PaddedBatch(
            self.coords[index],
            self.species[index],
            self.charges[index],
            self.atom_mask[index],
        )


def pad_from_mols(
    mols: list,  # List of RDKit Mol objects
    adapter: ModelAdapter,
    device: torch.device,
) -> PaddedBatch:
    """Pad molecular data directly from RDKit Mol objects.

    Builds the padded coordinate, species, charge, and atom-mask tensors in a
    single pass, avoiding intermediate per-molecule list creation.

    Args:
        mols: List of RDKit Mol objects with conformers.
        adapter: The model this batch is being built for, satisfying
            :class:`Auto3D.engines.models.contract.ModelAdapter`. It supplies all three
            model-dependent pieces -- the species convention
            (``adapter.to_species``) and both fill values (``adapter.coord_pad``,
            ``adapter.species_pad``).

            This used to be a model-name *string* plus the two pad values as
            separate arguments, so both call sites (``SPE.py``,
            ``batch_opt/batchopt.py``) handed over a name AND an adapter's pads:
            the remap came from one source and the sentinel from another, and
            nothing structurally stopped them from contradicting each other.
            That is the C3/C4 failure class, and taking the adapter is what makes
            it impossible rather than merely absent.
        device: Target device for tensors (CPU or CUDA).

    Returns:
        A :class:`PaddedBatch` of (coords, species, charges, atom_mask). It is a
        ``NamedTuple``, so positional unpacking is unchanged; the fields are:
        - coords: Shape (batch, max_atoms, 3), dtype float32. Leaf, with
          ``requires_grad=False`` -- grad state is the CALLER'S to set, not
          this function's (see the comment at the return statement below).
        - species: Shape (batch, max_atoms), dtype long
        - charges: Shape (batch,), dtype float32 (see note below)
        - atom_mask: Shape (batch, max_atoms), dtype bool, True for real atoms
          and False for padded slots. Callers must use this mask to identify
          padding rather than comparing species against ``species_pad``: a
          custom NNP's ``species_pad`` value can collide with a real species
          index (e.g. Auto3D's own ANI2xt convention maps hydrogen to index 0,
          the same value some adapters use as ``species_pad``), which would
          silently zero and exclude real atoms from the force-convergence
          check (audit C13).
    """
    from rdkit.Chem import rdmolops

    batch_size = len(mols)
    max_atoms = max(mol.GetNumAtoms() for mol in mols)

    # Pre-allocate CPU tensors with padding values. Built host-side and moved
    # to `device` ONCE each, after the loop below, rather than per-molecule
    # inside it (issue #24): constructing `torch.tensor(..., device=device)`
    # for every molecule's coords AND species -- two small tensors -- was two
    # blocking host-to-device copies per molecule, each its own CUDA
    # synchronization, none of which batches with the others because the
    # destination slice differs every iteration.
    coords_tensor = torch.full((batch_size, max_atoms, 3), adapter.coord_pad, dtype=torch.float32)
    species_tensor = torch.full((batch_size, max_atoms), adapter.species_pad, dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    charges = []

    # Fill in actual values -- CPU tensors throughout this loop, no device
    # traffic yet.
    for i, mol in enumerate(mols):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        coords_tensor[i, :n] = torch.tensor(conf.GetPositions(), dtype=torch.float32)

        spec = adapter.to_species([a.GetAtomicNum() for a in mol.GetAtoms()])
        species_tensor[i, :n] = torch.tensor(spec, dtype=torch.long)
        atom_mask[i, :n] = True

        charges.append(rdmolops.GetFormalCharge(mol))

    # Float (not long) to match the ASE Calculator path (ASE/thermo.py) and the
    # dtype the AIMNet2 adapter casts charges to internally; formal charges are
    # integers exactly representable in float32, and ANI models ignore charge.
    charges_tensor = torch.tensor(charges, dtype=torch.float32)

    # The one H2D copy per tensor promised above. A no-op (same tensor
    # returned) when `device` is CPU, which is every fast-tier test in this
    # repository -- the four transfers only exist when `device` is CUDA.
    coords_tensor = coords_tensor.to(device)
    species_tensor = species_tensor.to(device)
    atom_mask = atom_mask.to(device)
    charges_tensor = charges_tensor.to(device)

    # Grad state is deliberately NOT set here (issue #18). Both production
    # callers own it themselves and neither needed this: `ensemble_opt`
    # (batchopt.py) immediately `.detach()`es this tensor before building its
    # optimization state, and the step loop (`optimization_engine._step_active_subset`)
    # calls its own `coord.requires_grad_(True)` on the per-step subset every
    # iteration -- so the flag set here was overwritten before first use on
    # that path. `SPE.calc_spe` feeds this coords tensor straight into
    # `energy_batched` without detaching first, so setting it True here used
    # to matter there -- except `AIMNet2Adapter`'s calculator sets
    # `requires_grad_(True)` on its own copy of coord internally
    # (`aimnet.calculators.derivatives`), and the three ANI/custom adapters'
    # `energy()` build a graph if and only if the coords they are HANDED
    # already require grad. Setting it True unconditionally therefore built an
    # autograd graph for every SPE sub-batch through the full model -- for
    # ANI2x's 8-model ensemble, activations saved for a backward that
    # `energy_batched` (M39) deliberately never calls -- roughly doubling peak
    # memory for no benefit.
    return PaddedBatch(coords_tensor, species_tensor, charges_tensor, atom_mask)
