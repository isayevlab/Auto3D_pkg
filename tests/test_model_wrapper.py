# tests/test_model_wrapper.py
"""Unit tests for the model_wrapper module.

Tests for the BatchedEnergyForce class which wraps model adapters
and provides batched forward functionality.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import torch

from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce


class TestEnForceANIForward:
    """Tests for BatchedEnergyForce forward method."""

    def test_enforce_ani_forward_delegates_to_adapter(self):
        """BatchedEnergyForce.forward should delegate to model adapter's forward method."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 5, 3),
        )

        wrapper = BatchedEnergyForce(mock_adapter)
        coord = torch.randn(2, 5, 3)
        numbers = torch.ones(2, 5, dtype=torch.long)
        charges = torch.zeros(2)

        e, f = wrapper.forward(coord, numbers, charges)

        assert e.shape == (2,)
        assert f.shape == (2, 5, 3)
        mock_adapter.forward.assert_called_once()

    def test_enforce_ani_forward_with_batchsize_kwarg(self):
        """BatchedEnergyForce should accept batchsize_atoms as keyword argument."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0]),
            torch.randn(1, 5, 3),
        )

        wrapper = BatchedEnergyForce(mock_adapter, batchsize_atoms=512)

        assert wrapper.batchsize_atoms == 512

    def test_enforce_ani_forward_with_int_second_arg(self):
        """BatchedEnergyForce should accept int as second argument for batchsize."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0]),
            torch.randn(1, 5, 3),
        )

        wrapper = BatchedEnergyForce(mock_adapter, 256)

        assert wrapper.batchsize_atoms == 256


class TestEnForceANIForwardBatched:
    """Tests for BatchedEnergyForce forward_batched method."""

    def test_enforce_ani_forward_batched_splits_large_batches(self):
        """forward_batched should split large batches based on batchsize_atoms."""
        mock_adapter = MagicMock()

        def mock_forward(coords, species, charges, atom_mask=None):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        # Set small batchsize_atoms to force multiple batches
        wrapper = BatchedEnergyForce(mock_adapter, batchsize_atoms=10)

        # 4 molecules * 5 atoms = 20 atoms total > 10 batchsize_atoms
        coord = torch.randn(4, 5, 3)
        numbers = torch.ones(4, 5, dtype=torch.long)
        charges = torch.zeros(4)

        e, f = wrapper.forward_batched(coord, numbers, charges)

        assert e.shape == (4,)
        assert f.shape == (4, 5, 3)
        # Should have been called multiple times due to batching
        assert mock_adapter.forward.call_count >= 2

    def test_enforce_ani_forward_batched_handles_single_batch(self):
        """forward_batched should handle cases where all fits in one batch."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.randn(3, 5, 3),
        )

        # Large batchsize_atoms so all molecules fit in one batch
        wrapper = BatchedEnergyForce(mock_adapter, batchsize_atoms=10000)

        coord = torch.randn(3, 5, 3)
        numbers = torch.ones(3, 5, dtype=torch.long)
        charges = torch.zeros(3)

        e, f = wrapper.forward_batched(coord, numbers, charges)

        assert e.shape == (3,)
        assert f.shape == (3, 5, 3)
        assert mock_adapter.forward.call_count == 1

    def test_enforce_ani_forward_batched_handles_exact_batch_boundary(self):
        """forward_batched should handle exact batch size boundaries."""
        mock_adapter = MagicMock()

        def mock_forward(coords, species, charges, atom_mask=None):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        # batchsize_atoms = 10, atoms_per_mol = 5, so 2 mols per batch exactly
        wrapper = BatchedEnergyForce(mock_adapter, batchsize_atoms=10)

        # 4 molecules should fit into exactly 2 batches
        coord = torch.randn(4, 5, 3)
        numbers = torch.ones(4, 5, dtype=torch.long)
        charges = torch.zeros(4)

        e, f = wrapper.forward_batched(coord, numbers, charges)

        assert e.shape == (4,)
        assert f.shape == (4, 5, 3)
        assert mock_adapter.forward.call_count == 2

    def test_enforce_ani_forward_batched_minimum_one_molecule_per_batch(self):
        """forward_batched should process at least 1 molecule per batch."""
        mock_adapter = MagicMock()

        def mock_forward(coords, species, charges, atom_mask=None):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        # Very small batchsize_atoms (1) with large molecule (10 atoms)
        wrapper = BatchedEnergyForce(mock_adapter, batchsize_atoms=1)

        coord = torch.randn(3, 10, 3)  # 3 molecules with 10 atoms each
        numbers = torch.ones(3, 10, dtype=torch.long)
        charges = torch.zeros(3)

        e, f = wrapper.forward_batched(coord, numbers, charges)

        assert e.shape == (3,)
        assert f.shape == (3, 10, 3)
        # Should process one molecule at a time
        assert mock_adapter.forward.call_count == 3


class TestEnForceANIImportPath:
    """``model_wrapper`` is the one home of ``BatchedEnergyForce``.

    A companion test used to assert the same class was reachable as
    ``from Auto3D.engines.batch_opt.batchopt import BatchedEnergyForce``, which pinned the
    compat barrel in place. ``batchopt`` still imports the class -- it uses it --
    but no first-party module may reach it through there, which
    ``tests/test_import_boundaries.py`` now enforces statically.
    """

    def test_enforce_ani_import_from_model_wrapper(self):
        """BatchedEnergyForce should be importable from model_wrapper."""
        from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce as EnForce_ANI_wrapper

        assert EnForce_ANI_wrapper is BatchedEnergyForce


class TestEnForceANIModule:
    """Tests for BatchedEnergyForce as a PyTorch module."""

    def test_enforce_ani_is_nn_module(self):
        """BatchedEnergyForce should be a torch.nn.Module."""
        mock_adapter = MagicMock()
        wrapper = BatchedEnergyForce(mock_adapter)

        assert isinstance(wrapper, torch.nn.Module)

    def test_enforce_ani_stores_model(self):
        """BatchedEnergyForce should store the model adapter."""
        mock_adapter = MagicMock()
        wrapper = BatchedEnergyForce(mock_adapter)

        assert wrapper.model is mock_adapter

    def test_enforce_ani_default_batchsize(self):
        """BatchedEnergyForce should use default batchsize_atoms of 16384."""
        mock_adapter = MagicMock()
        wrapper = BatchedEnergyForce(mock_adapter)

        assert wrapper.batchsize_atoms == 1024 * 16  # 16384


def _batch_like_test_forward_batched_retries_on_oom():
    """The 2-molecule, 3-atom batch every OOM-retry test in this module uses.

    Lifted out of ``test_forward_batched_retries_on_oom`` (which built it
    inline) so the tests below do not each grow their own copy.
    """
    coord = torch.randn(2, 3, 3)
    numbers = torch.tensor([[1, 6, -1], [1, 6, -1]])
    charges = torch.zeros(2)
    return coord, numbers, charges


def test_forward_batched_retries_on_oom():
    """A transient CUDA OOM on a multi-molecule batch must be retried with a
    smaller batch (not crash the whole run). The adapter here OOMs on any batch
    larger than 1 molecule and succeeds at batch size 1."""
    import torch

    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    class _OOMAdapter(FakeAdapter):
        """Conforms to the contract (inherited), then OOMs on purpose.

        Subclassing the shared double rather than re-declaring an ad-hoc one is
        what keeps ``BatchedEnergyForce``'s contract gate tightenable: a hand-rolled
        stub listing only the members this test happens to exercise goes red for
        a reason that has nothing to do with OOM retry.
        """

        def forward(self, coord, numbers, charges, atom_mask=None):
            if coord.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return coord.pow(2).sum(dim=(1, 2)), torch.zeros_like(coord)

    wrapper = BatchedEnergyForce(_OOMAdapter(), batchsize_atoms=10_000)
    coord, numbers, charges = _batch_like_test_forward_batched_retries_on_oom()

    e, f = wrapper.forward_batched(coord, numbers, charges)
    assert e.shape == (2,) and torch.isfinite(e).all()
    assert f.shape == coord.shape


def test_forward_batched_retries_when_the_final_cat_ooms(monkeypatch):
    """The final ``torch.cat`` that assembles sub-batch results is
    itself a GPU allocation and can OOM even when every sub-batch forward
    succeeded. One retry, after ``empty_cache()``, must cover it too.

    This is the headline scoping property of that retry -- it must re-run only
    the failed concatenation, never the sub-batch forward above it -- and
    nothing asserted it before.
    ``batchsize_atoms=16`` over this 2-molecule, 3-atom batch is one
    sub-batch, so one ``forward`` call is the whole adapter-call budget;
    ``FakeAdapter`` records one entry in ``.calls`` per invocation, so
    counting those (rather than adding a second counting double) pins it
    directly. Three ``torch.cat`` calls: the energies concat fails once and
    succeeds on retry (2), then the forces concat succeeds outright (1) --
    retried separately in ``model_wrapper.py``, so a failure on the first does
    not re-run alongside the second.
    """
    import torch

    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    coord, numbers, charges = _batch_like_test_forward_batched_retries_on_oom()
    real_cat = torch.cat
    calls = {"n": 0}

    def flaky_cat(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("fake")
        return real_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", flaky_cat)
    adapter = FakeAdapter()
    wrapper = BatchedEnergyForce(adapter, batchsize_atoms=16)
    energies, forces = wrapper.forward_batched(coord, numbers, charges)
    assert torch.isfinite(energies).all()

    forward_calls = [c for c in adapter.calls if c["kind"] == "forward"]
    assert len(forward_calls) == 1, "the sub-batch forward must not be re-run by the OOM retry"
    assert calls["n"] == 3


def test_forward_batched_retries_only_the_forces_cat_when_that_is_the_one_that_ooms(monkeypatch):
    """The sibling of the test above: the first injected failure lands on the
    forces concatenation (the second ``torch.cat``) instead of the energies
    one. Each concatenation is retried separately in ``model_wrapper.py``, so
    the energies concatenation, already finished by the time the forces one
    OOMs, must not be re-run alongside it -- only a joint retry around both
    calls would do that, and it would still satisfy a test that only counts
    ``torch.cat`` calls in total. Energies tensors are 1-D (shape ``(B,)``)
    and forces tensors are 3-D (shape ``(B, N, 3)``), which is enough to tell
    the two calls apart without touching production code.
    """
    import torch

    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    coord, numbers, charges = _batch_like_test_forward_batched_retries_on_oom()
    real_cat = torch.cat
    calls = {"energies": 0, "forces": 0}

    def flaky_cat(tensors, *args, **kwargs):
        is_forces = tensors[0].dim() == 3
        key = "forces" if is_forces else "energies"
        calls[key] += 1
        if is_forces and calls["forces"] == 1:
            raise torch.cuda.OutOfMemoryError("fake")
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", flaky_cat)
    adapter = FakeAdapter()
    wrapper = BatchedEnergyForce(adapter, batchsize_atoms=16)
    energies, forces = wrapper.forward_batched(coord, numbers, charges)
    assert torch.isfinite(energies).all()

    assert calls["energies"] == 1, "the forces retry must not re-run the energies concatenation"
    assert calls["forces"] == 2


def test_forward_batched_propagates_if_the_retry_also_ooms(monkeypatch):
    """A single retry covers a transient condition; a second failure
    at the same concatenation must propagate, not loop or swallow the error.

    No test previously drove the final concatenation to fail twice in a row.
    """
    import pytest
    import torch

    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    coord, numbers, charges = _batch_like_test_forward_batched_retries_on_oom()

    def always_ooms(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("fake, never recovers")

    monkeypatch.setattr(torch, "cat", always_ooms)
    wrapper = BatchedEnergyForce(FakeAdapter(), batchsize_atoms=16)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="never recovers"):
        wrapper.forward_batched(coord, numbers, charges)


def test_a_runtime_error_naming_out_of_memory_is_retried_like_the_typed_one():
    """Some paths raise a bare ``RuntimeError`` whose message names the
    condition ("CUDA error: out of memory") instead of the typed
    ``torch.cuda.OutOfMemoryError``. Both must be treated the same way."""
    import torch

    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    coord, numbers, charges = _batch_like_test_forward_batched_retries_on_oom()

    class _RuntimeOOM(FakeAdapter):
        def forward(self, *a, **k):
            if not getattr(self, "_once", False):
                self._once = True
                raise RuntimeError("CUDA error: out of memory (fake)")
            return super().forward(*a, **k)

    wrapper = BatchedEnergyForce(_RuntimeOOM(), batchsize_atoms=16)
    energies, _ = wrapper.forward_batched(coord, numbers, charges)
    assert torch.isfinite(energies).all()


def test_empty_cache_runs_with_exception_context_cleared(monkeypatch):
    """M37: empty_cache()/the retry must run AFTER the except block has been
    left, not while the OOM exception (and everything its traceback keeps
    alive, including the failed forward's activations) is still the
    currently-handled exception -- otherwise empty_cache() can only release
    already-free blocks and cannot reclaim what the retry needs.

    ``torch.cuda.is_available`` is forced True. The call under test is guarded
    by it -- correctly, since ``empty_cache()`` has nothing to do without CUDA
    -- so on a CPU-only machine the spy below is never invoked and this test
    passes for no reason, or fails on a missing key. Without the patch it
    asserts something only on a GPU box: it passed locally on an 8-GPU host and
    failed all three CPU-only CI jobs with ``KeyError``. Nothing here touches a
    real device; the guard is the only thing being satisfied.
    """
    import sys

    from tests.helpers_adapter import FakeAdapter

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = {}

    class _OOMAdapter(FakeAdapter):
        def forward(self, coord, numbers, charges, atom_mask=None):
            if coord.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return coord.pow(2).sum(dim=(1, 2)), torch.zeros_like(coord)

    wrapper = BatchedEnergyForce(_OOMAdapter(), batchsize_atoms=10_000)

    real_empty_cache = torch.cuda.empty_cache

    def spy_empty_cache():
        seen["exc_info_at_empty_cache"] = sys.exc_info()
        return real_empty_cache()

    torch.cuda.empty_cache = spy_empty_cache
    try:
        coord = torch.randn(4, 3, 3)
        numbers = torch.tensor([[1, 6, -1]] * 4)
        charges = torch.zeros(4)
        wrapper.forward_batched(coord, numbers, charges)
    finally:
        torch.cuda.empty_cache = real_empty_cache

    assert seen["exc_info_at_empty_cache"] == (None, None, None), (
        "empty_cache() ran while the OOM exception was still the currently "
        "handled exception -- its traceback (and the failed forward's "
        "activations) were still reachable."
    )


def test_bsize_shrinkage_persists_for_remainder_of_batch():
    """M37: once a slice OOMs and is halved, LATER slices in the same
    forward_batched call must reuse the shrunk size, not repeat the same
    OOM-and-recurse cycle at the original (pre-OOM) size for every remaining
    slice."""
    from tests.helpers_adapter import FakeAdapter

    calls = []

    class _OOMAdapter(FakeAdapter):
        def forward(self, coord, numbers, charges, atom_mask=None):
            calls.append(coord.shape[0])
            if coord.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return coord.pow(2).sum(dim=(1, 2)), torch.zeros_like(coord)

    # 8 molecules, batchsize_atoms=12 // 3 atoms/mol -> initial bsize=4,
    # splitting the 8 molecules into two top-level size-4 slices. Both
    # top-level slices OOM (size 4 > 2) under the *original* size.
    wrapper = BatchedEnergyForce(_OOMAdapter(), batchsize_atoms=12)
    coord = torch.randn(8, 3, 3)
    numbers = torch.tensor([[1, 6, -1]] * 8)
    charges = torch.zeros(8)

    e, f = wrapper.forward_batched(coord, numbers, charges)

    assert e.shape == (8,)
    oom_triggering_calls = [c for c in calls if c > 2]
    assert len(oom_triggering_calls) == 1, (
        "the shrunk batch size must persist for the rest of the call -- "
        f"expected exactly one OOM-triggering (>2) call, got sizes {calls}"
    )


def test_a_model_name_in_the_batchsize_slot_is_rejected():
    """The removed API's shape must fail loudly, not become a bad batch size.

    Until 3.0.0 the second parameter was `name_or_batchsize: str | int | None`,
    type-switched between a model name and a batch size, and passing a string
    warned it would be "removed in Auto3D v2.0". The package reached 3.0.0 with it
    still there and no caller in `src/` ever passing one, so it is gone.

    With the union removed and nothing else added, `BatchedEnergyForce(adapter, "AIMNET")`
    would have assigned a string to `batchsize_atoms` and failed much later inside
    batching, as a comparison error naming neither the parameter nor the removal.
    """
    import pytest

    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce

    with pytest.raises(TypeError, match="batchsize_atoms"):
        BatchedEnergyForce(MagicMock(), "AIMNET")


class TestEnForceANIRejectsNonAdapters:
    """The adapter contract is enforced here, at the one seam that consumes it.

    ``ModelAdapter`` was declared ``@runtime_checkable`` and then never checked
    anywhere in ``src/`` or ``tests/``, while every signature that wanted "an
    adapter" annotated the ABC instead. This class is what makes the Protocol
    load-bearing: a category error (a raw ``nn.Module``, an
    ``AIMNet2Calculator``, an engine-name string) is named here instead of
    surfacing as an ``AttributeError`` several frames deep inside
    ``forward_batched``.

    Note what this does NOT catch: presence is not arity. An object with a
    wrong-signature ``forward`` still passes, and a ``MagicMock`` passes
    trivially (which the tests above rely on). The gate is for category errors.
    """

    def test_a_raw_nn_module_is_rejected_and_the_gap_is_named(self):
        import pytest

        with pytest.raises(TypeError) as excinfo:
            BatchedEnergyForce(torch.nn.Linear(1, 1))
        message = str(excinfo.value)
        assert "ModelAdapter" in message
        # The missing members must be enumerated, not merely alluded to.
        for name in ("to_species", "coord_pad", "species_pad", "energy"):
            assert name in message, f"{name} is missing but unnamed: {message}"

    def test_an_engine_name_string_is_rejected(self):
        """The pre-adapter API took a model name here; a stale caller must not
        get an object whose ``forward`` fails much later."""
        import pytest

        with pytest.raises(TypeError, match="ModelAdapter"):
            BatchedEnergyForce("AIMNET")

    def test_a_conforming_double_is_accepted(self):
        """The gate must not reject a structural (non-subclass) adapter --
        production has always accepted those, which is why annotating the ABC
        instead of the Protocol made the Protocol decorative."""
        from tests.helpers_adapter import FakeAdapter, padded_batch

        adapter = FakeAdapter()
        wrapper = BatchedEnergyForce(adapter)
        coords, species, charges, _mults, atom_mask = padded_batch()
        e, f = wrapper.forward(coords, species, charges, atom_mask=atom_mask)
        assert e.shape == (2,)
        assert f.shape == (2, 3, 3)

    def test_the_missing_member_list_comes_from_the_protocol(self):
        """Derived, not hand-listed: widening ``ModelAdapter`` must widen this
        message in the same edit."""
        import pytest

        from Auto3D.engines.models.contract import ModelAdapter

        class NothingAtAll:
            pass

        with pytest.raises(TypeError) as excinfo:
            BatchedEnergyForce(NothingAtAll())
        message = str(excinfo.value)
        for name in ModelAdapter.__annotations__:
            assert name in message


def test_oom_shrunk_bsize_persists_across_calls():
    """The halved sub-batch size must be remembered for the next step,
    not recomputed from batchsize_atoms on every forward_batched call."""
    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    calls = {"oom": 0}

    class _OOMAdapter(FakeAdapter):
        def forward(self, coord, numbers, charges, atom_mask=None):
            if coord.shape[0] > 16:
                calls["oom"] += 1
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return super().forward(coord, numbers, charges, atom_mask)

    B, N = 64, 5
    nn_ = BatchedEnergyForce(_OOMAdapter(), batchsize_atoms=B * N)  # bsize starts at 64
    coord = torch.zeros(B, N, 3)
    numbers = torch.ones(B, N, dtype=torch.long)
    charges = torch.zeros(B)
    mask = torch.ones(B, N, dtype=torch.bool)
    nn_.forward_batched(coord, numbers, charges, mask)
    first = calls["oom"]
    assert first >= 1
    nn_.forward_batched(coord, numbers, charges, mask)
    assert calls["oom"] == first, "second call must start at the shrunk size"


def test_a_transient_oom_on_a_short_tail_does_not_persist_a_tiny_bsize():
    """The shrunk size comes from `bsize`, not from the tail's length.

    The last slice of a batch is short because the molecules ran out, not
    because memory did, so halving *it* published a size for
    ``_bsize_by_n[N]`` that no other part of the batch had shown to be
    necessary -- and ``_bsize_by_n`` is read again by every later optimizer
    step. Here 6 molecules at ``bsize`` 4 leave a tail of 2 that OOMs
    once; the sub-batch size for the rest of the run must be 2 (4 // 2), not
    1 (2 // 2).
    """
    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    sizes = []

    class _OneTransientOOM(FakeAdapter):
        """OOMs exactly once, on the first two-molecule slice it is handed."""

        def __init__(self):
            super().__init__()
            self.fired = False

        def forward(self, coord, numbers, charges, atom_mask=None):
            sizes.append(coord.shape[0])
            if not self.fired and coord.shape[0] == 2:
                self.fired = True
                raise torch.cuda.OutOfMemoryError("simulated transient OOM")
            return super().forward(coord, numbers, charges, atom_mask)

    B, N = 6, 3
    wrapper = BatchedEnergyForce(_OneTransientOOM(), batchsize_atoms=4 * N)  # bsize = 4
    coord = torch.zeros(B, N, 3)
    numbers = torch.ones(B, N, dtype=torch.long)
    charges = torch.zeros(B)

    e, _ = wrapper.forward_batched(coord, numbers, charges)

    assert e.shape == (B,)
    assert wrapper._bsize_by_n[N] == 2, (
        f"the tail's own length was persisted as the sub-batch size; slices attempted were {sizes}"
    )
    # 4 (ok), 2 (OOM), 2 (the requeued tail, retried at the halved bsize).
    assert sizes == [4, 2, 2], sizes


def test_an_oom_far_below_the_sub_batch_size_does_not_walk_bsize_down_by_halves():
    """The halved size is capped at the size that just failed.

    ``bsize`` starts from ``batchsize_atoms // N`` and routinely exceeds the
    number of molecules still active -- conformers converge and leave the batch
    as an optimization proceeds. Halving ``bsize`` alone then re-slices the
    requeued tail into *exactly* the forward that just OOM'd, so every halving
    down to the active count is a wasted, guaranteed failure: 204 -> 102 -> 51
    -> ... is 7 identical failed forwards for one real shortage, and it
    overshoots to 1 molecule where the batch could carry 2.

    Five molecules, ``bsize`` 204, memory enough for two: one retry at 5 (the
    transient-OOM allowance), then a halving to 2, which sticks.
    """
    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from tests.helpers_adapter import FakeAdapter

    sizes = []

    class _CappedAdapter(FakeAdapter):
        def forward(self, coord, numbers, charges, atom_mask=None):
            sizes.append(coord.shape[0])
            if coord.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return super().forward(coord, numbers, charges, atom_mask)

    B, N = 5, 20
    wrapper = BatchedEnergyForce(_CappedAdapter(), batchsize_atoms=4096)  # bsize = 204
    coord = torch.zeros(B, N, 3)
    numbers = torch.ones(B, N, dtype=torch.long)
    charges = torch.zeros(B)

    e, _ = wrapper.forward_batched(coord, numbers, charges)

    assert e.shape == (B,)
    failed = [s for s in sizes if s > 2]
    assert len(failed) == 2, f"bsize was walked down by halves: {sizes}"
    assert wrapper._bsize_by_n[N] == 2, (
        f"undershot the batch's real capacity; slices attempted were {sizes}"
    )


def test_nan_bisection_does_not_contaminate_bsize_for_a_later_oom():
    """R68/GPU-review F2: the NaN bisection's own cap must not feed the OOM halving.

    A NaN bisection shrinks a LOCAL cap to isolate the offending row and lifts
    it again once that row is placeholdered; it must never touch ``bsize`` or
    ``self._bsize_by_n``, which belong to the OOM policy alone. Here row 0
    (tagged by species) goes non-finite first, forcing several bisection
    levels; once it is isolated, a later, unrelated slice of the still-healthy
    remainder (7 molecules) OOMs exactly once. ``_bsize_by_n[N]`` must land at
    what the OOM policy alone would set from THAT slice's real size (8 -> 4),
    not at some smaller value contaminated by the NaN episode's cap.
    """
    from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
    from Auto3D.foundation.exceptions import NumericalError
    from tests.helpers_adapter import FakeAdapter

    class _NanThenOOMAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.oom_fired = False

        def forward(self, coord, numbers, charges, atom_mask=None):
            if bool((numbers == 2).any()):
                raise NumericalError("simulated non-finite energy")
            if not self.oom_fired and coord.shape[0] > 4:
                self.oom_fired = True
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return super().forward(coord, numbers, charges, atom_mask)

    B, N = 8, 3
    wrapper = BatchedEnergyForce(_NanThenOOMAdapter(), batchsize_atoms=B * N)  # bsize = 8
    coord = torch.randn(B, N, 3)
    numbers = torch.ones(B, N, dtype=torch.long)
    numbers[0] = 2  # tags row 0 so the poison follows it through any slice
    charges = torch.zeros(B)

    e, f = wrapper.forward_batched(coord, numbers, charges)

    assert e.shape == (B,) and f.shape == coord.shape
    assert torch.isnan(e[0]), "the tagged row must still carry the NaN placeholder"
    assert torch.isfinite(e[1:]).all()
    assert wrapper._bsize_by_n[N] == 4, (
        "an OOM after a resolved NaN bisection must halve from bsize (8 -> 4), "
        f"not from a cap the bisection left behind; got {wrapper._bsize_by_n}"
    )
