# tests/test_model_wrapper.py
"""Unit tests for the model_wrapper module.

Tests for the EnForce_ANI class which wraps model adapters
and provides batched forward functionality.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import torch

from Auto3D.batch_opt.model_wrapper import EnForce_ANI


class TestEnForceANIForward:
    """Tests for EnForce_ANI forward method."""

    def test_enforce_ani_forward_delegates_to_adapter(self):
        """EnForce_ANI.forward should delegate to model adapter's forward method."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 5, 3),
        )

        wrapper = EnForce_ANI(mock_adapter)
        coord = torch.randn(2, 5, 3)
        numbers = torch.ones(2, 5, dtype=torch.long)
        charges = torch.zeros(2)

        e, f = wrapper.forward(coord, numbers, charges)

        assert e.shape == (2,)
        assert f.shape == (2, 5, 3)
        mock_adapter.forward.assert_called_once()

    def test_enforce_ani_forward_with_batchsize_kwarg(self):
        """EnForce_ANI should accept batchsize_atoms as keyword argument."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0]),
            torch.randn(1, 5, 3),
        )

        wrapper = EnForce_ANI(mock_adapter, batchsize_atoms=512)

        assert wrapper.batchsize_atoms == 512

    def test_enforce_ani_forward_with_int_second_arg(self):
        """EnForce_ANI should accept int as second argument for batchsize."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0]),
            torch.randn(1, 5, 3),
        )

        wrapper = EnForce_ANI(mock_adapter, 256)

        assert wrapper.batchsize_atoms == 256


class TestEnForceANIForwardBatched:
    """Tests for EnForce_ANI forward_batched method."""

    def test_enforce_ani_forward_batched_splits_large_batches(self):
        """forward_batched should split large batches based on batchsize_atoms."""
        mock_adapter = MagicMock()

        def mock_forward(coords, species, charges, atom_mask=None):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        # Set small batchsize_atoms to force multiple batches
        wrapper = EnForce_ANI(mock_adapter, batchsize_atoms=10)

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
        wrapper = EnForce_ANI(mock_adapter, batchsize_atoms=10000)

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
        wrapper = EnForce_ANI(mock_adapter, batchsize_atoms=10)

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
        wrapper = EnForce_ANI(mock_adapter, batchsize_atoms=1)

        coord = torch.randn(3, 10, 3)  # 3 molecules with 10 atoms each
        numbers = torch.ones(3, 10, dtype=torch.long)
        charges = torch.zeros(3)

        e, f = wrapper.forward_batched(coord, numbers, charges)

        assert e.shape == (3,)
        assert f.shape == (3, 10, 3)
        # Should process one molecule at a time
        assert mock_adapter.forward.call_count == 3


class TestEnForceANIImportPath:
    """``model_wrapper`` is the one home of ``EnForce_ANI``.

    A companion test used to assert the same class was reachable as
    ``from Auto3D.batch_opt.batchopt import EnForce_ANI``, which pinned the
    compat barrel in place. ``batchopt`` still imports the class -- it uses it --
    but no first-party module may reach it through there, which
    ``tests/test_import_boundaries.py`` now enforces statically.
    """

    def test_enforce_ani_import_from_model_wrapper(self):
        """EnForce_ANI should be importable from model_wrapper."""
        from Auto3D.batch_opt.model_wrapper import EnForce_ANI as EnForce_ANI_wrapper

        assert EnForce_ANI_wrapper is EnForce_ANI


class TestEnForceANIModule:
    """Tests for EnForce_ANI as a PyTorch module."""

    def test_enforce_ani_is_nn_module(self):
        """EnForce_ANI should be a torch.nn.Module."""
        mock_adapter = MagicMock()
        wrapper = EnForce_ANI(mock_adapter)

        assert isinstance(wrapper, torch.nn.Module)

    def test_enforce_ani_stores_model(self):
        """EnForce_ANI should store the model adapter."""
        mock_adapter = MagicMock()
        wrapper = EnForce_ANI(mock_adapter)

        assert wrapper.model is mock_adapter

    def test_enforce_ani_default_batchsize(self):
        """EnForce_ANI should use default batchsize_atoms of 16384."""
        mock_adapter = MagicMock()
        wrapper = EnForce_ANI(mock_adapter)

        assert wrapper.batchsize_atoms == 1024 * 16  # 16384


def test_forward_batched_retries_on_oom():
    """A transient CUDA OOM on a multi-molecule batch must be retried with a
    smaller batch (not crash the whole run). The adapter here OOMs on any batch
    larger than 1 molecule and succeeds at batch size 1."""
    import torch

    from Auto3D.batch_opt.model_wrapper import EnForce_ANI

    from tests.helpers_adapter import FakeAdapter

    class _OOMAdapter(FakeAdapter):
        """Conforms to the contract (inherited), then OOMs on purpose.

        Subclassing the shared double rather than re-declaring an ad-hoc one is
        what keeps ``EnForce_ANI``'s contract gate tightenable: a hand-rolled
        stub listing only the members this test happens to exercise goes red for
        a reason that has nothing to do with OOM retry.
        """

        def forward(self, coord, numbers, charges, atom_mask=None):
            if coord.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return coord.pow(2).sum(dim=(1, 2)), torch.zeros_like(coord)

    wrapper = EnForce_ANI(_OOMAdapter(), batchsize_atoms=10_000)
    coord = torch.randn(2, 3, 3)
    numbers = torch.tensor([[1, 6, -1], [1, 6, -1]])
    charges = torch.zeros(2)

    e, f = wrapper.forward_batched(coord, numbers, charges)
    assert e.shape == (2,) and torch.isfinite(e).all()
    assert f.shape == coord.shape


def test_empty_cache_runs_with_exception_context_cleared():
    """M37: empty_cache()/the retry must run AFTER the except block has been
    left, not while the OOM exception (and everything its traceback keeps
    alive, including the failed forward's activations) is still the
    currently-handled exception -- otherwise empty_cache() can only release
    already-free blocks and cannot reclaim what the retry needs."""
    import sys

    from tests.helpers_adapter import FakeAdapter

    seen = {}

    class _OOMAdapter(FakeAdapter):
        def forward(self, coord, numbers, charges, atom_mask=None):
            if coord.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return coord.pow(2).sum(dim=(1, 2)), torch.zeros_like(coord)

    wrapper = EnForce_ANI(_OOMAdapter(), batchsize_atoms=10_000)

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
    wrapper = EnForce_ANI(_OOMAdapter(), batchsize_atoms=12)
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

    With the union removed and nothing else added, `EnForce_ANI(adapter, "AIMNET")`
    would have assigned a string to `batchsize_atoms` and failed much later inside
    batching, as a comparison error naming neither the parameter nor the removal.
    """
    import pytest

    from Auto3D.batch_opt.model_wrapper import EnForce_ANI

    with pytest.raises(TypeError, match="batchsize_atoms"):
        EnForce_ANI(MagicMock(), "AIMNET")


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
            EnForce_ANI(torch.nn.Linear(1, 1))
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
            EnForce_ANI("AIMNET")

    def test_a_conforming_double_is_accepted(self):
        """The gate must not reject a structural (non-subclass) adapter --
        production has always accepted those, which is why annotating the ABC
        instead of the Protocol made the Protocol decorative."""
        from tests.helpers_adapter import FakeAdapter, padded_batch

        adapter = FakeAdapter()
        wrapper = EnForce_ANI(adapter)
        coords, species, charges, atom_mask = padded_batch()
        e, f = wrapper.forward(coords, species, charges, atom_mask=atom_mask)
        assert e.shape == (2,)
        assert f.shape == (2, 3, 3)

    def test_the_missing_member_list_comes_from_the_protocol(self):
        """Derived, not hand-listed: widening ``ModelAdapter`` must widen this
        message in the same edit."""
        import pytest

        from Auto3D.models.contract import ModelAdapter

        class NothingAtAll:
            pass

        with pytest.raises(TypeError) as excinfo:
            EnForce_ANI(NothingAtAll())
        message = str(excinfo.value)
        for name in ModelAdapter.__annotations__:
            assert name in message
