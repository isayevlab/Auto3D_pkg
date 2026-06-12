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
        assert wrapper._use_legacy_forward is False

    def test_enforce_ani_forward_with_int_second_arg(self):
        """EnForce_ANI should accept int as second argument for batchsize."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0]),
            torch.randn(1, 5, 3),
        )

        wrapper = EnForce_ANI(mock_adapter, 256)

        assert wrapper.batchsize_atoms == 256
        assert wrapper._use_legacy_forward is False


class TestEnForceANIForwardBatched:
    """Tests for EnForce_ANI forward_batched method."""

    def test_enforce_ani_forward_batched_splits_large_batches(self):
        """forward_batched should split large batches based on batchsize_atoms."""
        mock_adapter = MagicMock()

        def mock_forward(coords, species, charges):
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

        def mock_forward(coords, species, charges):
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

        def mock_forward(coords, species, charges):
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


class TestEnForceANIBackwardCompatibility:
    """Tests for backward compatibility with legacy API."""

    def test_enforce_ani_legacy_api_emits_deprecation_warning(self):
        """Using string name should emit deprecation warning."""
        # Need a real nn.Module for the legacy API since it uses add_module()
        mock_model = torch.nn.Linear(1, 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapper = EnForce_ANI(mock_model, "AIMNET", 1024)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert wrapper._use_legacy_forward is True
            assert wrapper.name == "AIMNET"

    def test_enforce_ani_import_from_batchopt(self):
        """EnForce_ANI should be importable from batchopt for backward compatibility."""
        from Auto3D.batch_opt.batchopt import EnForce_ANI as EnForce_ANI_batchopt

        # Should be the same class
        assert EnForce_ANI_batchopt is EnForce_ANI

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

    class _OOMAdapter:
        coord_pad = 0.0
        species_pad = -1

        def forward(self, coord, numbers, charges):
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
