# tests/test_optimization_engine.py
"""Unit tests for the optimization_engine module.

Tests for the print_stats and n_steps functions which handle the main
optimization loop for batch geometry optimization.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.batch_opt.optimization_engine import n_steps, print_stats


class TestPrintStats:
    """Tests for print_stats function."""

    def test_print_stats_outputs_correctly(self, caplog):
        """print_stats should output convergence info with correct counts."""
        state = {
            'numbers': torch.ones(10, 5),
            'converged_mask': torch.tensor([True, True, False, False, False,
                                            False, False, False, False, False]),
            'oscillating_count': torch.zeros(10, 1),
        }

        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 10" in caplog.text
        assert "Converged: 2" in caplog.text
        assert "Active: 8" in caplog.text

    def test_print_stats_all_converged(self, caplog):
        """print_stats should report all converged when all are done."""
        state = {
            'numbers': torch.ones(5, 3),
            'converged_mask': torch.tensor([True, True, True, True, True]),
            'oscillating_count': torch.zeros(5, 1),
        }

        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 5" in caplog.text
        assert "Converged: 5" in caplog.text
        assert "Active: 0" in caplog.text

    def test_print_stats_with_oscillating(self, caplog):
        """print_stats should report dropped structures correctly."""
        state = {
            'numbers': torch.ones(6, 3),
            'converged_mask': torch.tensor([True, True, True, False, False, False]),
            'oscillating_count': torch.tensor([[150], [150], [50], [0], [0], [0]], dtype=torch.float),
        }

        # patience=100, so structures with count >= 100 are considered dropped
        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 6" in caplog.text
        # 3 converged_mask True, but 2 have oscillating_count >= 100
        # So: converged = 3 - 2 = 1, dropped = 2, active = 3
        assert "Converged: 1" in caplog.text
        assert "Dropped(Oscillating): 2" in caplog.text
        assert "Active: 3" in caplog.text

    def test_print_stats_empty_batch(self, caplog):
        """print_stats should handle empty batches gracefully."""
        state = {
            'numbers': torch.ones(0, 5),
            'converged_mask': torch.tensor([], dtype=torch.bool),
            'oscillating_count': torch.zeros(0, 1),
        }

        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 0" in caplog.text

    def test_print_stats_flushes_output(self, caplog):
        """print_stats should produce log output."""
        state = {
            'numbers': torch.ones(2, 3),
            'converged_mask': torch.tensor([False, False]),
            'oscillating_count': torch.zeros(2, 1),
        }

        # This test verifies the function completes without error
        # and produces log output
        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert len(caplog.text) > 0


class TestNSteps:
    """Tests for n_steps function."""

    def test_n_steps_initializes_state(self):
        """n_steps should initialize oscillating_count in state."""
        mock_nn = MagicMock()
        # Return low forces so structures converge immediately
        mock_nn.forward_batched.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.ones(2, 5, 3) * 0.001  # Very small forces
        )

        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': torch.randn(2, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=10, opttol=0.01, patience=100)

        assert 'oscillating_count' in state

    def test_n_steps_updates_converged_mask(self):
        """n_steps should update converged_mask for structures with low forces."""
        mock_nn = MagicMock()
        # Return forces below opttol for all structures
        mock_nn.forward_batched.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.ones(2, 5, 3) * 0.001  # Forces much smaller than opttol
        )

        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': torch.randn(2, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=10, opttol=0.01, patience=100)

        # Both structures should converge since forces are below threshold
        assert state['converged_mask'].all()

    def test_n_steps_updates_energy(self):
        """n_steps should update energy values in state."""
        mock_nn = MagicMock()
        mock_nn.forward_batched.return_value = (
            torch.tensor([-10.0, -20.0]),  # Energies
            torch.ones(2, 5, 3) * 0.001  # Small forces to converge
        )

        initial_energy = torch.full((2,), 999.0, dtype=torch.double)
        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': torch.randn(2, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': initial_energy.clone(),
        }

        n_steps(state, n=10, opttol=0.01, patience=100)

        # Energies should be updated from initial 999.0
        assert not torch.equal(state['energy'], initial_energy)

    def test_n_steps_updates_coordinates(self):
        """n_steps should update coordinates during optimization."""
        mock_nn = MagicMock()
        # Return non-trivial forces so optimizer moves atoms
        mock_nn.forward_batched.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 5, 3) * 0.1  # Moderate forces
        )

        initial_coord = torch.randn(2, 5, 3)
        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': initial_coord.clone(),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=5, opttol=0.001, patience=100)

        # Coordinates should have been modified
        assert not torch.equal(state['coord'], initial_coord)

    def test_n_steps_stops_when_all_converged(self):
        """n_steps should stop early when all structures converge."""
        call_count = [0]

        def mock_forward(coord, numbers, charges):
            call_count[0] += 1
            batch_size = coord.shape[0]
            # Return very small forces so convergence happens immediately
            return torch.ones(batch_size), torch.ones(batch_size, coord.shape[1], 3) * 0.0001

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': torch.randn(2, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=100, opttol=0.01, patience=1000)

        # Should have stopped early (not run 100 times)
        # First call converges structures, second call finds all converged
        assert call_count[0] < 100

    def test_n_steps_detects_oscillating(self):
        """n_steps should mark oscillating structures as converged (dropped)."""
        step_count = [0]

        def mock_forward(coord, numbers, charges):
            step_count[0] += 1
            batch_size = coord.shape[0]
            # Return forces that never decrease (always same value)
            # This should trigger oscillation detection
            return torch.ones(batch_size), torch.ones(batch_size, coord.shape[1], 3) * 0.5

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': torch.randn(2, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': torch.full((2,), 999.0, dtype=torch.double),
        }

        # With patience=5, structures should be dropped after 5 steps without improvement
        n_steps(state, n=20, opttol=0.01, patience=5)

        # Both structures should be marked as converged (dropped due to oscillation)
        assert state['converged_mask'].all()
        # oscillating_count should be >= patience for dropped structures
        assert (state['oscillating_count'] >= 5).all()

    def test_n_steps_energy_convergence(self):
        """n_steps should converge based on energy stability."""
        call_count = [0]

        def mock_forward(coord, numbers, charges):
            call_count[0] += 1
            batch_size = coord.shape[0]
            # Return constant energy (stable) and moderate forces
            # Energy convergence should kick in after energy_patience steps
            return torch.ones(batch_size) * -10.0, torch.ones(batch_size, coord.shape[1], 3) * 0.05

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            'numbers': torch.ones(2, 5, dtype=torch.long),
            'charges': torch.zeros(2, dtype=torch.long),
            'coord': torch.randn(2, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False]),
            'fmax': torch.full((2,), 999.0),
            'energy': torch.full((2,), 999.0, dtype=torch.double),
        }

        # opttol=0.01, forces=0.05, so force criterion not met
        # But energy is stable, and 0.05 < 0.01 * 10, so energy convergence should trigger
        n_steps(state, n=100, opttol=0.01, patience=1000, energy_tol=1e-4, energy_patience=3)

        # Should converge via energy stability
        assert state['converged_mask'].all()


class TestNStepsIntegration:
    """Integration tests for n_steps with realistic scenarios."""

    def test_n_steps_with_decreasing_forces(self):
        """n_steps should converge structures with decreasing forces."""
        step_count = [0]

        def mock_forward(coord, numbers, charges):
            step_count[0] += 1
            batch_size = coord.shape[0]
            # Simulate forces that decrease with each step
            force_magnitude = max(0.001, 0.5 / step_count[0])
            return torch.ones(batch_size) * (-step_count[0]), torch.ones(batch_size, coord.shape[1], 3) * force_magnitude

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            'numbers': torch.ones(3, 5, dtype=torch.long),
            'charges': torch.zeros(3, dtype=torch.long),
            'coord': torch.randn(3, 5, 3),
            'nn': mock_nn,
            'converged_mask': torch.tensor([False, False, False]),
            'fmax': torch.full((3,), 999.0),
            'energy': torch.full((3,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=100, opttol=0.01, patience=50)

        # Should have converged
        assert state['converged_mask'].all()
        # fmax should be updated to reasonable values
        assert (state['fmax'] < 999.0).all()


class TestNStepsBackwardCompatibility:
    """Tests for backward compatibility of n_steps function."""

    def test_n_steps_importable_from_batchopt(self):
        """n_steps should be importable from batchopt for backward compatibility."""
        from Auto3D.batch_opt.batchopt import n_steps as n_steps_batchopt

        # Should be the same function
        assert n_steps_batchopt is n_steps

    def test_print_stats_importable_from_batchopt(self):
        """print_stats should be importable from batchopt for backward compatibility."""
        from Auto3D.batch_opt.batchopt import print_stats as print_stats_batchopt

        # Should be the same function
        assert print_stats_batchopt is print_stats


class TestPrintStatsBackwardCompatibility:
    """Tests ensuring print_stats maintains backward compatibility."""

    def test_print_stats_uses_oscillating_count_key(self, caplog):
        """print_stats should use 'oscillating_count' key (correct spelling)."""
        state = {
            'numbers': torch.ones(3, 5),
            'converged_mask': torch.tensor([True, False, False]),
            'oscillating_count': torch.tensor([[100], [0], [0]], dtype=torch.float),
        }

        # Should work with correctly spelled key
        with caplog.at_level(logging.INFO):
            print_stats(state, patience=50)

        assert "Dropped(Oscillating): 1" in caplog.text
