# tests/test_batchopt.py
"""Unit tests for the batchopt module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.batch_opt.batchopt import optimizing, EnForce_ANI


class TestOptimizingUsesModelFactory:
    """Tests for optimizing class using ModelFactory."""

    def test_optimizing_uses_model_factory(self):
        """optimizing class should use ModelFactory for model creation."""
        with patch('Auto3D.batch_opt.batchopt.create_model') as mock_factory:
            mock_adapter = MagicMock()
            mock_adapter.coord_pad = 0.0
            mock_adapter.species_pad = 0
            mock_factory.return_value = mock_adapter

            config = {
                'opt_steps': 100,
                'opttol': 0.003,
                'patience': 1000,
                'batchsize_atoms': 1024
            }
            device = torch.device("cpu")
            opt = optimizing("dummy.sdf", "out.sdf", "AIMNET", device, config)

            mock_factory.assert_called_once_with("AIMNET", device)
            # Verify the adapter's properties are used
            assert opt.coord_pad == 0.0
            assert opt.species_pad == 0

    def test_optimizing_uses_adapter_padding_values(self):
        """optimizing should get coord_pad and species_pad from the adapter."""
        with patch('Auto3D.batch_opt.batchopt.create_model') as mock_factory:
            mock_adapter = MagicMock()
            mock_adapter.coord_pad = 1.5
            mock_adapter.species_pad = -2
            mock_factory.return_value = mock_adapter

            config = {
                'opt_steps': 100,
                'opttol': 0.003,
                'patience': 1000,
                'batchsize_atoms': 1024
            }
            device = torch.device("cpu")
            opt = optimizing("dummy.sdf", "out.sdf", "AIMNET", device, config)

            # Verify padding values come from adapter
            assert opt.coord_pad == 1.5
            assert opt.species_pad == -2


class TestEnForceANI:
    """Tests for EnForce_ANI class."""

    def test_enforce_ani_delegates_to_adapter(self):
        """EnForce_ANI.forward should delegate to adapter's forward method."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 5, 3)
        )

        # EnForce_ANI should accept the adapter directly
        model = EnForce_ANI(mock_adapter, batchsize_atoms=1024)

        coords = torch.randn(2, 5, 3)
        species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]])
        charges = torch.tensor([0, 0])

        energy, forces = model.forward(coords, species, charges)

        # Verify adapter's forward was called
        mock_adapter.forward.assert_called_once()
        call_args = mock_adapter.forward.call_args
        assert torch.equal(call_args[0][0], coords)
        assert torch.equal(call_args[0][1], species)
        assert torch.equal(call_args[0][2], charges)

    def test_enforce_ani_forward_batched(self):
        """EnForce_ANI.forward_batched should batch calls correctly."""
        mock_adapter = MagicMock()
        # Return consistent results for batching
        def mock_forward(coords, species, charges):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        model = EnForce_ANI(mock_adapter, batchsize_atoms=10)  # Small batch size

        # Create input that will require multiple batches (5 atoms * 4 batches = 20 atoms > 10)
        coords = torch.randn(4, 5, 3)
        species = torch.ones(4, 5, dtype=torch.long)
        charges = torch.zeros(4, dtype=torch.long)

        energy, forces = model.forward_batched(coords, species, charges)

        # Should have called forward multiple times due to batching
        assert mock_adapter.forward.call_count >= 1
        assert energy.shape == (4,)
        assert forces.shape == (4, 5, 3)
