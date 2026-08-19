# tests/test_ani2xt_atom_energies.py
"""Gate the M7 rewrite of ANI2xt's per-element energy loop.

``ANI2xt.forward`` used to loop over its seven per-element networks doing
``if mask.any(): atom_energies[mask] = network(aev[mask])``. Measured with
``tests/helpers_sync_count.py``, that is **22 host-device synchronizations per
forward** with all seven elements present (7 for the guard, 7 masked reads,
7 masked writes, 1 for ``_validate_outputs``) and 16 with a drug-like four
elements -- on every step of every optimization.

It was also uncompilable. ``if mask.any():`` is a data-dependent branch, and a
graph break *inside* a ``for`` loop gives Dynamo nowhere to place a resume
point, so it skipped the entire frame: ``compile_model=True`` produced **zero**
subgraphs for this model, not seven. Deleting the guard alone does not fix that,
because ``nonzero`` and boolean-mask indexing are dynamic-output-shape ops and
break the same way. Only a loop body with no data-dependent op at all compiles,
which is why the per-element indices are computed outside ``forward`` and passed
in.

Every test here runs on CPU, needs no GPU, and -- because ``_atom_energies``,
``element_indices`` and ``self_atomic_energies`` are module-level functions
taking ``networks`` as a parameter rather than methods on a model that owns a
torchani AEV computer -- **needs no torchani**. That testability is the reason
for the extraction, not a side effect of it.

What these tests cannot establish, and nothing here pretends otherwise:

* Whether the *real* ``ANI2xt.forward``, with torchani's ``AEVComputer`` in the
  frame, also reaches one subgraph. The AEV computer may break the graph on its
  own, and any break inside the per-element loop re-triggers the whole-frame
  skip. That needs torchani.
* Any wall-clock number. A sync costs only what it serializes, which depends on
  the GPU and the batch size. See ``benchmarks/bench_optimization_perf.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from Auto3D.engines.models.ani2xt import (
    NUM_ELEMENTS,
    _atom_energies,
    element_indices,
    self_atomic_energies,
)
from tests.helpers_sync_count import BOOL_MASK_LABELS, NONZERO, SyncCounter, count_graphs

AEV_DIM = 8


def _networks(num_elements: int = NUM_ELEMENTS, seed: int = 0) -> nn.ModuleList:
    """Stand-in for ANI2xt's per-element MLPs. Shape-compatible, torchani-free."""
    torch.manual_seed(seed)
    return nn.ModuleList([nn.Linear(AEV_DIM, 1) for _ in range(num_elements)])


def _reference_atom_energies(networks, species_idx, aev):
    """The pre-M7 loop: guard, boolean-mask read, boolean-mask write.

    Reimplemented here rather than imported, so bit-identity is measured against
    an independent statement of the old behaviour.
    """
    batch, atoms = species_idx.shape
    out = torch.zeros(batch, atoms, device=aev.device, dtype=torch.float64)
    for elem_idx, network in enumerate(networks):
        mask = species_idx == elem_idx
        if mask.any():
            out[mask] = network(aev[mask]).squeeze(-1).to(torch.float64)
    return out


def _reference_element_indices(species_idx, num_elements=NUM_ELEMENTS):
    """``nonzero`` per element -- what ``element_indices`` must reproduce exactly."""
    flat = species_idx.reshape(-1)
    return [torch.nonzero(flat == elem, as_tuple=True)[0] for elem in range(num_elements)]


def _reference_self_energies(species_idx, shifts, num_elements=NUM_ELEMENTS):
    """The inline self-energy loop that used to run on every forward."""
    out = torch.zeros(species_idx.shape[0], device=species_idx.device, dtype=torch.float64)
    for elem_idx in range(num_elements):
        counts = (species_idx == elem_idx).sum(dim=1).to(torch.float64)
        out += counts * shifts[elem_idx]
    return out


_SPECIES_CASES = {
    "all seven elements": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0], [6, 5, 4, 3, 2, 1, 0, 1]]),
    "only two of seven": torch.tensor([[0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0]]),
    "one element only": torch.full((2, 8), 3),
    "with padding": torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1], [3, 4, -1, -1, -1, -1, -1, -1]]),
    "all padded": torch.full((2, 8), -1),
    "single molecule single atom": torch.tensor([[5]]),
    "out of range species": torch.tensor([[0, 6, 7, 99, -3, 2, 2, 2]]),
    "empty batch": torch.zeros(0, 8, dtype=torch.long),
}


def _aev_for(species_idx: torch.Tensor, seed: int = 3) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*species_idx.shape, AEV_DIM)


class TestElementIndices:
    """``element_indices`` reproduces seven ``nonzero`` calls with one readback."""

    @pytest.mark.parametrize("label", list(_SPECIES_CASES))
    def test_matches_nonzero_reference(self, label):
        """Same indices, same order, for every species pattern that can occur.

        Padded slots (``-1``) and out-of-range species get their own buckets and
        are dropped, rather than being clamped into a neighbouring element's
        network -- which is what would silently feed a stray atom to the wrong
        MLP.
        """
        species = _SPECIES_CASES[label].long()
        expected = _reference_element_indices(species)
        actual = element_indices(species)

        assert len(actual) == len(expected) == NUM_ELEMENTS
        for elem_idx, (got, want) in enumerate(zip(actual, expected)):
            assert got.dtype is torch.int64
            assert torch.equal(got, want), f"element {elem_idx} of {label}"

    def test_matches_nonzero_reference_over_random_patterns(self):
        """A sweep, because the bucket arithmetic is easy to get subtly wrong.

        Species are drawn from ``[-3, NUM_ELEMENTS + 3)`` so padding, valid
        indices and out-of-range values all occur, in every combination.
        """
        for seed in range(200):
            torch.manual_seed(seed)
            batch = int(torch.randint(1, 6, (1,)))
            atoms = int(torch.randint(1, 25, (1,)))
            species = torch.randint(-3, NUM_ELEMENTS + 3, (batch, atoms))
            for got, want in zip(element_indices(species), _reference_element_indices(species)):
                assert torch.equal(got, want), f"seed {seed}"

    def test_performs_exactly_one_host_readback(self):
        """One ``tolist``, no ``item``, no ``nonzero``. That is the whole point.

        Seven ``nonzero`` calls were seven synchronizations; this is one. The
        readback is counted by wrapping ``Tensor.tolist``/``Tensor.item`` rather
        than by dispatch mode, because a device-to-host copy is invisible in a
        CPU-only process -- there is no second device for it to cross.
        """
        calls = {"tolist": 0, "item": 0}
        original_tolist, original_item = torch.Tensor.tolist, torch.Tensor.item

        def counting_tolist(self, *args, **kwargs):
            calls["tolist"] += 1
            return original_tolist(self, *args, **kwargs)

        def counting_item(self, *args, **kwargs):
            calls["item"] += 1
            return original_item(self, *args, **kwargs)

        species = torch.randint(-1, NUM_ELEMENTS, (16, 40))
        torch.Tensor.tolist, torch.Tensor.item = counting_tolist, counting_item
        try:
            element_indices(species)
        finally:
            torch.Tensor.tolist, torch.Tensor.item = original_tolist, original_item

        assert calls == {"tolist": 1, "item": 0}

    def test_does_no_nonzero_and_no_boolean_mask_indexing(self):
        """The dispatch-level statement of the same claim."""
        counter = SyncCounter()
        with counter:
            element_indices(torch.randint(-1, NUM_ELEMENTS, (16, 40)))
        assert counter.counts[NONZERO] == 0, counter.report()
        assert counter.bool_mask_ops == 0, counter.report()

    def test_reference_really_does_seven_nonzeros(self):
        """Prove the baseline, so the comparison above is not against nothing."""
        counter = SyncCounter()
        with counter:
            _reference_element_indices(torch.randint(-1, NUM_ELEMENTS, (16, 40)))
        assert counter.counts[NONZERO] == NUM_ELEMENTS, counter.report()


class TestAtomEnergies:
    """``_atom_energies`` is bit-identical to the masked loop it replaced."""

    @pytest.mark.parametrize("label", list(_SPECIES_CASES))
    def test_matches_masked_reference(self, label):
        """Exact equality, including the batch where only 2 of 7 elements appear.

        That case is the one that proves the deleted ``if mask.any():`` guard
        protected nothing: for an absent element the index is empty,
        ``network(empty)`` returns an empty tensor and ``index_copy`` with an
        empty index is a no-op.
        """
        species = _SPECIES_CASES[label].long()
        aev = _aev_for(species)
        networks = _networks()
        batch, atoms = species.shape

        expected = _reference_atom_energies(networks, species, aev)
        actual = _atom_energies(
            networks,
            aev.reshape(batch * atoms, AEV_DIM),
            element_indices(species),
            batch * atoms,
        ).reshape(batch, atoms)

        assert actual.dtype is torch.float64
        assert torch.equal(actual, expected), label

    def test_padded_and_out_of_range_rows_stay_zero(self):
        """Rows belonging to no element contribute exactly zero energy.

        Not "approximately zero": they are never written, so the initial zero
        survives. A padded atom that picked up an energy would shift a molecule's
        total by an amount depending on how much padding its bucket happened to
        need.
        """
        species = torch.tensor([[0, -1, 7, 3, -1]])
        aev = _aev_for(species)
        result = _atom_energies(_networks(), aev.reshape(5, AEV_DIM), element_indices(species), 5)
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert (result[[0, 3]] != 0.0).all()

    def test_does_no_boolean_mask_indexing(self):
        """Zero masked reads, zero masked writes, zero ``nonzero`` in the loop."""
        species = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
        aev = _aev_for(species)
        networks = _networks()
        index = element_indices(species)

        counter = SyncCounter()
        with counter:
            _atom_energies(networks, aev.reshape(8, AEV_DIM), index, 8)
        assert counter.total == 0, counter.report()

    def test_masked_reference_does_twenty_one_syncs(self):
        """Pin the baseline this replaced: 7 guards + 7 reads + 7 writes.

        Without this the "22 -> 2 per forward" claim would rest on an unmeasured
        memory of what the old code cost.

        Two atoms of every element, deliberately. With exactly *one* atom of an
        element the assigned value has ``numel() == 1`` and ATen's
        ``canDispatchToMaskedFill`` fast path lowers the write to
        ``masked_fill_``, which does not sync -- so a sparser batch measures
        fewer than 21 and the honest figure is "up to 21, and 21 for any
        realistic molecule".
        """
        species = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 0, 0]])
        counter = SyncCounter()
        with counter:
            _reference_atom_energies(_networks(), species, _aev_for(species))
        assert counter.total == 3 * NUM_ELEMENTS, counter.report()
        assert counter.bool_mask_ops == 2 * NUM_ELEMENTS, counter.report()

    def test_rewrite_does_one_sync_where_the_reference_did_twenty_one(self):
        """End to end for the loop: 21 -> 1, and the 1 is the counts readback.

        The remaining synchronization is ``element_indices``' single host
        readback, which a CPU-only process cannot observe as a device copy --
        hence the ``tolist`` count rather than a dispatch count. Add
        ``_validate_outputs``' one scalar read, which is unchanged and lives in
        the adapter, and the ANI2xt forward goes from 22 to 2.
        """
        species = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 0, 0]])
        aev = _aev_for(species)
        networks = _networks()

        readbacks = 0
        original_tolist = torch.Tensor.tolist

        def counting_tolist(self, *args, **kwargs):
            nonlocal readbacks
            readbacks += 1
            return original_tolist(self, *args, **kwargs)

        counter = SyncCounter()
        torch.Tensor.tolist = counting_tolist
        try:
            with counter:
                index = element_indices(species)
                _atom_energies(networks, aev.reshape(16, AEV_DIM), index, 16)
        finally:
            torch.Tensor.tolist = original_tolist

        assert counter.total == 0, counter.report()
        assert readbacks == 1


class TestCompilation:
    """The graph-count claims, which are the load-bearing half of M7."""

    def test_atom_energies_compiles_to_one_graph(self):
        """One subgraph, and ``fullgraph=True`` succeeds.

        ``fullgraph=True`` is the sharper assertion: it turns any graph break
        into an exception, so it cannot be satisfied by a frame that Dynamo
        quietly skipped.
        """
        species = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0], [1, 1, 2, 2, 3, 3, 4, 4]])
        aev = _aev_for(species)
        networks = _networks()
        index = element_indices(species)
        flat = aev.reshape(16, AEV_DIM)

        assert count_graphs(_atom_energies, networks, flat, index, 16) == 1
        assert count_graphs(_atom_energies, networks, flat, index, 16, fullgraph=True) == 1

    def test_masked_reference_compiles_to_zero_graphs(self):
        """The control, without which "1 graph" means nothing.

        A data-dependent branch inside a ``for`` loop does not split the frame
        into several graphs -- Dynamo abandons the frame, giving **zero**. This
        is why the original report of "breaks into 7 graphs" was wrong in the
        direction that made it worse, and why ``compile_model=True`` could not
        have been speeding up this module.
        """
        species = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0], [1, 1, 2, 2, 3, 3, 4, 4]])
        aev = _aev_for(species)
        networks = _networks()

        assert count_graphs(_reference_atom_energies, networks, species, aev) == 0

    def test_masked_reference_cannot_satisfy_fullgraph(self):
        """And it fails outright under ``fullgraph=True``, naming the reason."""
        species = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
        with pytest.raises(Exception, match="[Dd]ata-dependent"):
            count_graphs(
                _reference_atom_energies, _networks(), species, _aev_for(species), fullgraph=True
            )


class TestSelfAtomicEnergies:
    """Hoisting a pure function of ``species`` out of the hot path."""

    @pytest.mark.parametrize("label", list(_SPECIES_CASES))
    def test_matches_inline_reference(self, label):
        """Bit-identical: the same seven terms summed in the same order."""
        species = _SPECIES_CASES[label].long()
        shifts = torch.tensor(
            [-0.5984, -38.0826, -54.7031, -75.1901, -99.8006, -398.1224, -460.1387],
            dtype=torch.float64,
        )
        assert torch.equal(
            self_atomic_energies(species, shifts),
            _reference_self_energies(species, shifts),
        )

    def test_is_independent_of_coordinates(self):
        """Hoisting is only safe because nothing here depends on geometry.

        Stated as a test rather than a comment: the value is a function of
        ``species`` and ``energy_shifts`` alone, so computing it once per bucket
        instead of once per step cannot change any energy.
        """
        species = torch.tensor([[0, 1, 2, 3, 4, 5, 6, -1]])
        shifts = torch.arange(NUM_ELEMENTS, dtype=torch.float64) * -1.5
        first = self_atomic_energies(species, shifts)
        second = self_atomic_energies(species.clone(), shifts.clone())
        assert torch.equal(first, second)
        assert first.dtype is torch.float64

    def test_padded_slots_contribute_no_shift(self):
        """``-1`` matches no element, so padding adds no self-energy."""
        shifts = torch.full((NUM_ELEMENTS,), -7.0, dtype=torch.float64)
        unpadded = torch.tensor([[0, 1, 2]])
        padded = torch.tensor([[0, 1, 2, -1, -1]])
        assert torch.equal(
            self_atomic_energies(unpadded, shifts), self_atomic_energies(padded, shifts)
        )
