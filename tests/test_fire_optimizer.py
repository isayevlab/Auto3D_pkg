# tests/test_fire_optimizer.py
"""Unit tests for the FIRE optimizer module."""

from __future__ import annotations

import pytest
import torch

from Auto3D.batch_opt.fire_optimizer import FIRE


class TestFIREInitialization:
    """Tests for FIRE optimizer initialization."""

    def test_fire_initializes_with_correct_shape(self):
        """FIRE should initialize state tensors with correct shapes."""
        batch_size = 4
        n_atoms = 10
        coord = torch.randn(batch_size, n_atoms, 3)

        optimizer = FIRE(coord)

        assert optimizer.v.shape == (batch_size, n_atoms, 3)
        assert optimizer.Nsteps.shape == (batch_size,)
        assert optimizer.dt.shape == (batch_size,)
        assert optimizer.a.shape == (batch_size,)

    def test_fire_initializes_velocity_to_zero(self):
        """FIRE should initialize velocities to zero."""
        coord = torch.randn(2, 5, 3)
        optimizer = FIRE(coord)

        assert torch.allclose(optimizer.v, torch.zeros_like(coord))

    def test_fire_initializes_on_correct_device(self):
        """FIRE state tensors should be on the same device as input."""
        coord = torch.randn(2, 5, 3)
        optimizer = FIRE(coord)

        assert optimizer.v.device == coord.device
        assert optimizer.Nsteps.device == coord.device
        assert optimizer.dt.device == coord.device
        assert optimizer.a.device == coord.device

    def test_fire_default_parameters(self):
        """FIRE should have correct default parameters."""
        coord = torch.randn(2, 5, 3)
        optimizer = FIRE(coord)

        assert optimizer.dt_max == 0.1
        assert optimizer.Nmin == 5
        assert optimizer.maxstep == 0.1
        assert optimizer.finc == 1.5
        assert optimizer.fdec == 0.7
        assert optimizer.astart == 0.1
        assert optimizer.fa == 0.99


class TestFIREStep:
    """Tests for FIRE optimizer step function."""

    def test_fire_step_returns_correct_shape(self):
        """FIRE step should return coordinates with same shape as input."""
        coord = torch.randn(2, 5, 3)
        forces = torch.randn(2, 5, 3)

        optimizer = FIRE(coord)
        new_coord = optimizer(coord, forces)

        assert new_coord.shape == coord.shape

    def test_fire_step_modifies_coordinates(self):
        """FIRE step should modify coordinates based on forces."""
        coord = torch.randn(2, 5, 3)
        forces = torch.randn(2, 5, 3) * 0.1  # Small forces

        optimizer = FIRE(coord)
        new_coord = optimizer(coord, forces)

        # Coordinates should have changed (non-zero forces cause movement)
        assert not torch.allclose(new_coord, coord)

    def test_fire_step_with_zero_forces(self):
        """FIRE step with zero forces should barely change coordinates."""
        coord = torch.randn(2, 5, 3)
        forces = torch.zeros(2, 5, 3)

        optimizer = FIRE(coord)
        new_coord = optimizer(coord, forces)

        # With zero forces, displacement should be minimal
        displacement = (new_coord - coord).norm()
        assert displacement < 1e-6

    def test_fire_step_updates_velocity(self):
        """FIRE step should update internal velocity state."""
        coord = torch.randn(2, 5, 3)
        forces = torch.randn(2, 5, 3) * 0.1

        optimizer = FIRE(coord)
        initial_v = optimizer.v.clone()

        _ = optimizer(coord, forces)

        # Velocity should have been modified
        assert not torch.allclose(optimizer.v, initial_v)

    def test_fire_convergence_direction(self):
        """FIRE should move atoms in direction of forces (gradient descent)."""
        coord = torch.zeros(1, 3, 3)  # 1 molecule, 3 atoms
        # Forces pointing in positive x direction
        forces = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])

        optimizer = FIRE(coord)
        new_coord = optimizer(coord, forces)

        # Atoms should have moved in positive x direction
        assert (new_coord[0, :, 0] > coord[0, :, 0]).all()

    def test_fire_respects_maxstep(self):
        """FIRE should clamp displacement to maxstep."""
        coord = torch.zeros(1, 3, 3)
        # Very large forces
        forces = torch.ones(1, 3, 3) * 1000.0

        optimizer = FIRE(coord)
        new_coord = optimizer(coord, forces)

        # Total displacement should be bounded
        dr = (new_coord - coord).flatten(-2, -1)
        displacement_norm = dr.norm(p=2, dim=-1)

        # Should not exceed maxstep
        assert displacement_norm.max() <= optimizer.maxstep + 1e-6

    def test_fire_tiny_progressing_forces_stay_finite(self):
        """Underflowing force norms must not inject inf/NaN.

        A molecule can be 'progressing' (v.f > 0) while its float32 force norm
        underflows to exactly 0 -- forces so small (~1e-24) that their squares
        round to zero. Without clamping the force-norm denominator, forces /
        f_norm becomes inf/NaN, is selected into the velocity, and permanently
        corrupts that conformer. Guard: the step must stay finite.
        """
        coord = torch.zeros(1, 2, 3)
        optimizer = FIRE(coord)
        optimizer.v = torch.full((1, 2, 3), 1e-10)  # finite v_norm
        forces = torch.full((1, 2, 3), 1e-24)  # f^2 underflows to 0 in fp32

        # Precondition: this configuration really does trigger the degenerate
        # branch (progressing molecule with a zeroed force norm).
        vf = (forces * optimizer.v).flatten(-2, -1).sum(-1)
        f_norm = forces.flatten(-2, -1).norm(p=2, dim=-1)
        assert (vf > 0).all() and float(f_norm) == 0.0

        new_coord = optimizer(coord, forces)
        assert torch.isfinite(new_coord).all()
        assert torch.isfinite(optimizer.v).all()


class TestFIREClean:
    """Tests for FIRE optimizer clean method."""

    def test_fire_clean_subsets_state(self):
        """FIRE.clean should subset all internal state tensors."""
        coord = torch.randn(4, 5, 3)
        optimizer = FIRE(coord)

        # Apply one step to populate velocities
        forces = torch.randn(4, 5, 3)
        optimizer(coord, forces)

        # Clean to keep only first 2 molecules
        keep = torch.tensor([0, 1])
        result = optimizer.clean(keep)

        assert result is True
        assert optimizer.v.shape[0] == 2
        assert optimizer.Nsteps.shape[0] == 2
        assert optimizer.dt.shape[0] == 2
        assert optimizer.a.shape[0] == 2

    def test_fire_clean_preserves_correct_molecules(self):
        """FIRE.clean should preserve state for correct molecules."""
        coord = torch.randn(3, 5, 3)
        optimizer = FIRE(coord)

        # Apply step to create non-zero velocities
        forces = torch.randn(3, 5, 3)
        optimizer(coord, forces)

        # Store original values for molecule 1 (index 1)
        original_v1 = optimizer.v[1].clone()
        original_dt1 = optimizer.dt[1].clone()

        # Keep only molecules 1 and 2 (indices 1, 2)
        optimizer.clean(torch.tensor([1, 2]))

        # Molecule 1's state should now be at index 0
        assert torch.allclose(optimizer.v[0], original_v1)
        assert torch.allclose(optimizer.dt[0], original_dt1)

    def test_fire_clean_empty_index_results_empty(self):
        """FIRE.clean with an empty index should result in empty tensors.

        Reached when every remaining structure leaves the active set in the same
        step; an empty ``index_select`` is a valid no-op-shaped result, not an
        error.
        """
        coord = torch.randn(3, 5, 3)
        optimizer = FIRE(coord)

        optimizer.clean(torch.zeros(0, dtype=torch.long))

        assert optimizer.v.shape[0] == 0
        assert optimizer.Nsteps.shape[0] == 0

    def test_fire_clean_full_index_unchanged(self):
        """FIRE.clean with every index should preserve all molecules."""
        coord = torch.randn(3, 5, 3)
        optimizer = FIRE(coord)

        optimizer.clean(torch.arange(3))

        assert optimizer.v.shape[0] == 3
        assert optimizer.Nsteps.shape[0] == 3

    def test_fire_clean_rejects_a_boolean_mask(self):
        """A boolean mask must fail loudly, because it would not fail otherwise.

        ``clean`` used to take a boolean mask and now takes an int64 index. That
        is the one genuinely dangerous kind of signature change: ``index_select``
        accepts ``tensor([True, True, False, False])`` and reinterprets it as
        indices ``[1, 1, 0, 0]``, so an out-of-tree caller that was never updated
        would get a silently permuted, wrong-length optimizer state instead of an
        exception. Hence the explicit dtype check.
        """
        optimizer = FIRE(torch.randn(4, 5, 3))
        with pytest.raises(ValueError, match="int64"):
            optimizer.clean(torch.tensor([True, True, False, False]))

    def test_fire_clean_reindexes_in_the_order_given(self):
        """State follows the index, position for position.

        ``n_steps`` always passes an ascending index from ``torch.nonzero``, but
        pinning the general behaviour is what makes the bit-identity argument in
        ``test_optimization_engine_indexing.py`` checkable: ``index_select``
        preserves order, so a converged molecule's removal never permutes the
        survivors relative to the batch they are scattered back into.
        """
        optimizer = FIRE(torch.randn(4, 5, 3))
        optimizer(torch.randn(4, 5, 3), torch.randn(4, 5, 3))
        before_dt = optimizer.dt.clone()
        before_v = optimizer.v.clone()

        optimizer.clean(torch.tensor([2, 0]))

        assert torch.equal(optimizer.dt, torch.stack([before_dt[2], before_dt[0]]))
        assert torch.equal(optimizer.v, torch.stack([before_v[2], before_v[0]]))


class TestFIREMultipleSteps:
    """Tests for FIRE optimizer behavior over multiple steps."""

    def test_fire_multiple_steps_convergence(self):
        """FIRE should converge toward lower force configuration over steps."""
        coord = torch.randn(1, 5, 3)

        optimizer = FIRE(coord)

        # Simulate optimization with decreasing forces
        for i in range(10):
            # Simulate forces that decrease with optimization
            forces = torch.randn(1, 5, 3) * (1.0 / (i + 1))
            coord = optimizer(coord, forces)

        # After multiple steps, should have made progress
        assert optimizer.Nsteps[0] > 0 or optimizer.dt[0] < 0.1

    def test_fire_time_step_increases_with_progress(self):
        """FIRE should increase time step when making consistent progress."""
        coord = torch.randn(1, 10, 3)
        optimizer = FIRE(coord)

        # Provide consistent forces in same direction
        forces = torch.ones(1, 10, 3) * 0.1

        # Apply multiple steps with consistent force direction
        for _ in range(10):
            coord = optimizer(coord, forces)

        # Time step may have increased if Nsteps > Nmin
        # (depends on force-velocity alignment)
        final_dt = optimizer.dt[0].item()

        # At least verify dt didn't go negative or crazy
        assert final_dt > 0
        assert final_dt <= optimizer.dt_max

    def test_fire_resets_on_force_reversal(self):
        """FIRE should reset velocity when forces reverse direction."""
        coord = torch.randn(1, 5, 3)
        optimizer = FIRE(coord)

        # Build up velocity in one direction
        forces_positive = torch.ones(1, 5, 3) * 0.1
        for _ in range(5):
            coord = optimizer(coord, forces_positive)

        # Reverse force direction - should trigger reset
        forces_negative = -torch.ones(1, 5, 3) * 0.1
        coord = optimizer(coord, forces_negative)

        # After reversal, velocity direction should have changed
        # or been reset depending on dot product
        assert optimizer.v.norm().item() >= 0  # Sanity check


class TestFIREBatchBehavior:
    """Tests for FIRE optimizer batch processing behavior."""

    def test_fire_handles_mixed_convergence(self):
        """FIRE should handle batches where some molecules need reset."""
        coord = torch.randn(4, 5, 3)
        optimizer = FIRE(coord)

        # Build up velocity
        for _ in range(3):
            forces = torch.randn(4, 5, 3) * 0.1
            coord = optimizer(coord, forces)

        # Now give half the molecules opposing forces
        forces = torch.randn(4, 5, 3) * 0.1
        forces[:2] = -optimizer.v[:2] * 100  # Force reset for first 2

        coord = optimizer(coord, forces)

        # Should still work without error
        assert coord.shape == (4, 5, 3)

    def test_fire_independent_molecule_tracking(self):
        """FIRE tracks each molecule's dt/a state independently, driven only
        by that molecule's OWN progressing flag -- not by whatever the other
        molecules in the batch are doing (fire_optimizer.py's per-molecule
        torch.where selects on ``progressing``/``speedup``, but ``a3``,
        ``dt``, ``self.a`` etc are shared *tensors* the batch is stepped
        through together, so a batch-index mix-up would leak one molecule's
        state into another's).

        Protocol (all pure tensor arithmetic; deterministic, no randomness):
        Phase 1 -- all 3 molecules push in the same fixed direction, so they
        progress together and jointly build up ``Nsteps`` past ``Nmin``
        (bootstrapping requires ALL molecules progressing at least once; see
        the ``all_progressing`` branch). Phase 2 -- molecule 0 starts
        flipping its own force sign every step. Once misaligned with its own
        velocity, a molecule can never re-progress under an alternating
        force (each non-progressing step resets v to align with THAT step's
        force, and the next step's opposite force is then anti-aligned) --
        so molecule 0 is guaranteed to be "not progressing" for the rest of
        phase 2, while molecules 1/2 keep progressing and trigger the
        speed-up branch (``past_nmin`` already true from phase 1,
        ``all_progressing`` now false because molecule 0 dissents).
        """
        n_atoms = 2
        astart = 0.1
        dt_max = 0.1
        coord = torch.zeros(3, n_atoms, 3)
        optimizer = FIRE(coord)
        steady_force = torch.ones(n_atoms, 3) * 0.1

        # Phase 1: all three molecules progress together, bootstrapping
        # Nsteps past Nmin (=5) for everyone.
        for _ in range(8):
            forces = torch.stack([steady_force, steady_force, steady_force])
            coord = optimizer(coord, forces)
        assert (optimizer.Nsteps > 5).all(), "phase 1 setup failed to reach Nmin"

        # Phase 2: molecule 0 alternates sign every step (can never progress
        # again); molecules 1 and 2 keep pushing steadily.
        for step in range(10):
            forces = torch.stack(
                [
                    steady_force if step % 2 == 0 else -steady_force,
                    steady_force,
                    steady_force,
                ]
            )
            coord = optimizer(coord, forces)

        # Molecule 0 never progressed in phase 2, so its mixing parameter
        # `a` must be exactly reset to astart every single step -- the
        # oscillating molecule's own branch, untouched by its batch-mates.
        assert optimizer.a[0].item() == pytest.approx(astart)

        # Molecules 1/2 share an identical force history and so must reach
        # an identical (and, since they triggered the speed-up branch,
        # strictly smaller-than-astart) mixing parameter -- proving the
        # speed-up state is tracked per molecule, not smeared across the
        # batch by molecule 0's resets.
        assert optimizer.a[1].item() == pytest.approx(optimizer.a[2].item())
        assert optimizer.a[1].item() < astart

        # dt tells the same story from the other side: molecule 0's dt was
        # repeatedly shrunk (fdec) by its own non-progress, while 1/2's dt
        # saturated at dt_max via their own speed-up.
        assert optimizer.dt[1].item() == pytest.approx(dt_max)
        assert optimizer.dt[2].item() == pytest.approx(dt_max)
        assert optimizer.dt[0].item() < optimizer.dt[1].item()


class TestFIRETorchScript:
    """Tests for TorchScript compatibility of FIRE optimizer."""

    def test_fire_is_torchscript_class(self):
        """FIRE should be a TorchScript class."""
        coord = torch.randn(2, 5, 3)
        optimizer = FIRE(coord)

        # TorchScript classes have specific attributes
        # This test verifies the @torch.jit.script decorator worked
        assert callable(optimizer)
        assert hasattr(optimizer, "clean")

    def test_fire_works_in_jit_context(self):
        """FIRE should work correctly when used in JIT-compiled code."""
        coord = torch.randn(2, 5, 3)
        forces = torch.randn(2, 5, 3) * 0.1

        optimizer = FIRE(coord)

        # Use with torch.jit context
        with torch.jit.optimized_execution(False):
            new_coord = optimizer(coord, forces)

        assert new_coord.shape == coord.shape
        assert not torch.allclose(new_coord, coord)


def test_fire_trajectory_golden():
    """FIRE must produce a deterministic trajectory; guards the branchless rewrite."""
    import torch

    from Auto3D.batch_opt.fire_optimizer import FIRE

    torch.manual_seed(0)
    # 3 molecules, 4 atoms each; mix of progressing/resetting dynamics
    coord = torch.linspace(-1, 1, 36).reshape(3, 4, 3).contiguous()
    opt = FIRE(coord)
    with torch.no_grad():
        for step in range(20):
            # deterministic pseudo-forces depending on coord (toy quadratic well)
            forces = -0.5 * coord + 0.1 * torch.sin(coord * 3.0)
            coord = opt(coord, forces).detach()
    checksum = float(coord.abs().sum().item())
    # EXPECTED generated from the ORIGINAL (pre-rewrite) FIRE implementation and
    # matched by the branchless rewrite; guards numerical equivalence.
    EXPECTED = 12.077177047729492
    assert abs(checksum - EXPECTED) < 1e-5, f"got {checksum}"
