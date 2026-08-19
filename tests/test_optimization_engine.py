# tests/test_optimization_engine.py
"""Unit tests for the optimization_engine module.

Tests for the print_stats and n_steps functions which handle the main
optimization loop for batch geometry optimization.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import torch

from Auto3D.engines.batch_opt.optimization_engine import n_steps, print_stats


class TestPrintStats:
    """Tests for print_stats function."""

    def test_print_stats_outputs_correctly(self, caplog):
        """print_stats should output convergence info with correct counts."""
        state = {
            "numbers": torch.ones(10, 5),
            "converged_mask": torch.tensor(
                [True, True, False, False, False, False, False, False, False, False]
            ),
            "oscillating_count": torch.zeros(10, 1),
        }

        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 10" in caplog.text
        assert "Converged: 2" in caplog.text
        assert "Active: 8" in caplog.text

    def test_print_stats_all_converged(self, caplog):
        """print_stats should report all converged when all are done."""
        state = {
            "numbers": torch.ones(5, 3),
            "converged_mask": torch.tensor([True, True, True, True, True]),
            "oscillating_count": torch.zeros(5, 1),
        }

        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 5" in caplog.text
        assert "Converged: 5" in caplog.text
        assert "Active: 0" in caplog.text

    def test_print_stats_with_oscillating(self, caplog):
        """print_stats should report dropped structures correctly."""
        state = {
            "numbers": torch.ones(6, 3),
            "converged_mask": torch.tensor([True, True, True, False, False, False]),
            "oscillating_count": torch.tensor(
                [[150], [150], [50], [0], [0], [0]], dtype=torch.float
            ),
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
            "numbers": torch.ones(0, 5),
            "converged_mask": torch.tensor([], dtype=torch.bool),
            "oscillating_count": torch.zeros(0, 1),
        }

        with caplog.at_level(logging.INFO):
            print_stats(state, patience=100)

        assert "Total 3D structures: 0" in caplog.text

    def test_print_stats_flushes_output(self, caplog):
        """print_stats should produce log output."""
        state = {
            "numbers": torch.ones(2, 3),
            "converged_mask": torch.tensor([False, False]),
            "oscillating_count": torch.zeros(2, 1),
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
            torch.ones(2, 5, 3) * 0.001,  # Very small forces
        )

        state = {
            "numbers": torch.ones(2, 5, dtype=torch.long),
            "charges": torch.zeros(2, dtype=torch.long),
            "coord": torch.randn(2, 5, 3),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False]),
            "fmax": torch.full((2,), 999.0),
            "energy": torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=10, opttol=0.01, patience=100)

        assert "oscillating_count" in state

    def test_n_steps_updates_converged_mask(self):
        """n_steps should update converged_mask for structures with low forces."""
        mock_nn = MagicMock()
        # Return forces below opttol for all structures
        mock_nn.forward_batched.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.ones(2, 5, 3) * 0.001,  # Forces much smaller than opttol
        )

        state = {
            "numbers": torch.ones(2, 5, dtype=torch.long),
            "charges": torch.zeros(2, dtype=torch.long),
            "coord": torch.randn(2, 5, 3),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False]),
            "fmax": torch.full((2,), 999.0),
            "energy": torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=10, opttol=0.01, patience=100)

        # Both structures should converge since forces are below threshold
        assert state["converged_mask"].all()

    def test_n_steps_updates_energy(self):
        """n_steps should update energy values in state."""
        mock_nn = MagicMock()
        mock_nn.forward_batched.return_value = (
            torch.tensor([-10.0, -20.0]),  # Energies
            torch.ones(2, 5, 3) * 0.001,  # Small forces to converge
        )

        initial_energy = torch.full((2,), 999.0, dtype=torch.double)
        state = {
            "numbers": torch.ones(2, 5, dtype=torch.long),
            "charges": torch.zeros(2, dtype=torch.long),
            "coord": torch.randn(2, 5, 3),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False]),
            "fmax": torch.full((2,), 999.0),
            "energy": initial_energy.clone(),
        }

        n_steps(state, n=10, opttol=0.01, patience=100)

        # Energies should be updated from initial 999.0
        assert not torch.equal(state["energy"], initial_energy)

    def test_n_steps_updates_coordinates(self):
        """n_steps should update coordinates during optimization."""
        mock_nn = MagicMock()
        # Return non-trivial forces so optimizer moves atoms
        mock_nn.forward_batched.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 5, 3) * 0.1,  # Moderate forces
        )

        initial_coord = torch.randn(2, 5, 3)
        state = {
            "numbers": torch.ones(2, 5, dtype=torch.long),
            "charges": torch.zeros(2, dtype=torch.long),
            "coord": initial_coord.clone(),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False]),
            "fmax": torch.full((2,), 999.0),
            "energy": torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=5, opttol=0.001, patience=100)

        # Coordinates should have been modified
        assert not torch.equal(state["coord"], initial_coord)

    def test_n_steps_stops_when_all_converged(self):
        """n_steps should stop early when all structures converge."""
        call_count = [0]

        def mock_forward(coord, numbers, charges, atom_mask=None):
            call_count[0] += 1
            batch_size = coord.shape[0]
            # Return very small forces so convergence happens immediately
            return torch.ones(batch_size), torch.ones(batch_size, coord.shape[1], 3) * 0.0001

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            "numbers": torch.ones(2, 5, dtype=torch.long),
            "charges": torch.zeros(2, dtype=torch.long),
            "coord": torch.randn(2, 5, 3),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False]),
            "fmax": torch.full((2,), 999.0),
            "energy": torch.full((2,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=100, opttol=0.01, patience=1000)

        # Should have stopped early (not run 100 times)
        # First call converges structures, second call finds all converged
        assert call_count[0] < 100

    def test_n_steps_detects_oscillating(self):
        """n_steps should mark oscillating structures as converged (dropped)."""
        step_count = [0]

        def mock_forward(coord, numbers, charges, atom_mask=None):
            step_count[0] += 1
            batch_size = coord.shape[0]
            # Return forces that never decrease (always same value)
            # This should trigger oscillation detection
            return torch.ones(batch_size), torch.ones(batch_size, coord.shape[1], 3) * 0.5

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            "numbers": torch.ones(2, 5, dtype=torch.long),
            "charges": torch.zeros(2, dtype=torch.long),
            "coord": torch.randn(2, 5, 3),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False]),
            "fmax": torch.full((2,), 999.0),
            "energy": torch.full((2,), 999.0, dtype=torch.double),
        }

        # With patience=5, structures should be dropped after 5 steps without improvement
        n_steps(state, n=20, opttol=0.01, patience=5)

        # Both structures should be marked as converged (dropped due to oscillation)
        assert state["converged_mask"].all()
        # oscillating_count should be >= patience for dropped structures
        assert (state["oscillating_count"] >= 5).all()


class _ConstantForceNN:
    """Stub NNP: a constant force field plus a caller-chosen energy sequence.

    Every atom's force vector points along x, so ``f.norm(dim=-1)`` is the
    x-component exactly (``sqrt(v**2) == v`` with no rounding for the
    power-of-two magnitudes used below). The magnitude never changes, so
    ``fmax`` sits at a chosen, exact position relative to ``opttol`` for the
    whole run regardless of how far FIRE moves the atoms.

    ``energy_stable=True`` reports the same energy at every call (a perfectly
    stable energy -- the most favourable possible input for an energy-based
    convergence criterion); ``False`` shifts it by 1 eV per call, far above
    any plausible energy tolerance.

    ``force_per_atom`` may be a scalar (every molecule feels the same force) or
    a ``{species: force}`` map for a heterogeneous batch. The map is keyed on
    species rather than on row index because ``n_steps`` gathers a SUBSET of
    the batch once some molecules converge (``_step_active_subset`` gathers
    ``coord`` and ``numbers`` with the same ``active_idx``), so row 1 of a later
    step is not molecule 1. Keying on ``numbers`` -- gathered
    alongside ``coord`` -- makes each force follow its own molecule.
    """

    def __init__(self, force_per_atom: float | dict[int, float], energy_stable: bool):
        self.force_per_atom = force_per_atom
        self.energy_stable = energy_stable
        self.calls = 0

    def forward_batched(self, coord, numbers, charges, atom_mask=None):
        self.calls += 1
        batch = coord.shape[0]
        value = -10.0 if self.energy_stable else -10.0 - self.calls
        e = torch.full((batch,), value)
        f = torch.zeros_like(coord)
        if isinstance(self.force_per_atom, dict):
            for row in range(batch):
                f[row, :, 0] = self.force_per_atom[int(numbers[row, 0])]
        else:
            f[..., 0] = self.force_per_atom
        return e, f


def _run_constant_force(
    force_per_atom: float | dict[int, float],
    energy_stable: bool,
    opttol: float,
    species: tuple[int, int] = (1, 1),
):
    """Run n_steps against _ConstantForceNN and return the convergence mask.

    ``species`` labels the two molecules so a ``{species: force}`` map can give
    them different forces; the default keeps both at species 1 (a homogeneous
    batch, where the mask is always all-True or all-False).
    """
    nn = _ConstantForceNN(force_per_atom, energy_stable)
    numbers = torch.stack(
        [
            torch.full((4,), species[0], dtype=torch.long),
            torch.full((4,), species[1], dtype=torch.long),
        ]
    )
    state = {
        "numbers": numbers,
        "charges": torch.zeros(2, dtype=torch.long),
        "coord": torch.zeros(2, 4, 3),
        "nn": nn,
        "converged_mask": torch.zeros(2, dtype=torch.bool),
        "fmax": torch.full((2,), 999.0),
        "energy": torch.full((2,), 999.0, dtype=torch.double),
    }
    # patience far above n so nothing is dropped as oscillating: the constant
    # force never decreases, so the oscillation counter climbs every step and
    # would otherwise mask the effect under test.
    n_steps(state, n=20, opttol=opttol, patience=1000)
    return state["converged_mask"].tolist()


def test_convergence_outcome_never_depends_on_energy_stability():
    """Energy stability alone can never converge a structure (audit M1).

    Algebraic proof, for the code as it stood before M1:

        not_converged_post1 = fmax > opttol
        energy_converged    = (energy_stable_subset >= energy_patience) & (fmax < opttol)
        not_converged_post  = not_converged_post1 & not_oscillating & ~energy_converged

    Where ``not_converged_post1`` is true, ``fmax > opttol``, so ``fmax <
    opttol`` is false, so ``energy_converged`` is false and
    ``~energy_converged`` is true -- the identity of ``&``. Where
    ``not_converged_post1`` is false the conjunction is false regardless. At
    the ``fmax == opttol`` boundary both comparisons are false, so the same
    holds. The term could therefore never change an outcome, and deleting it
    changes no geometry this package produces.

    This test is that proof rather than a spot check, but expressed through
    the production loop instead of restated in Python booleans: a lattice test
    over three locally-defined ``bool``s would pass no matter what
    ``optimization_engine`` does. Here the lattice is exhaustive over the two
    axes that reach the deleted term -- ``fmax`` below / exactly at / above
    ``opttol`` (its only three orderings), crossed with energy perfectly
    stable / never stable -- and each cell is decided by running ``n_steps``.

    It is also a live tripwire, not just documentation. The ``above`` cells
    assert that a structure whose energy has been constant for twenty
    consecutive steps is still *not* reported converged while its force is
    twice the tolerance. Reintroducing early termination on a relaxed force
    gate -- e.g. the ``fmax < opttol * 10`` this loop once used -- turns those
    cells green-to-red immediately.
    """
    opttol = 0.0625  # exact in binary, so the "at" cell is a true tie
    regimes = {"below": opttol / 2, "at": opttol, "above": opttol * 2}

    outcomes = {
        (label, stable): _run_constant_force(force, stable, opttol)
        for label, force in regimes.items()
        for stable in (True, False)
    }

    # The claim: at every force regime, a perfectly stable energy and a wildly
    # unstable one produce the identical convergence outcome.
    for label in regimes:
        assert outcomes[(label, True)] == outcomes[(label, False)], (
            f"energy stability changed the outcome at fmax {label} opttol: "
            f"stable={outcomes[(label, True)]} unstable={outcomes[(label, False)]}"
        )

    # And the outcome is exactly the force criterion, in both energy regimes.
    for stable in (True, False):
        assert all(outcomes[("below", stable)]), "fmax < opttol must converge"
        assert all(outcomes[("at", stable)]), "fmax == opttol must converge"
        assert not any(outcomes[("above", stable)]), (
            "fmax > opttol must not converge, however stable the energy"
        )

    # A heterogeneous batch, so the step loop actually performs a PARTIAL
    # subset gather. Every cell above uses one force for both molecules, so
    # `not_converged` is always all-True or all-False and the gathers in
    # `_step_active_subset` are no-ops. That leaves a reintroduction
    # bug invisible: a criterion whose per-molecule buffer is gathered with a
    # stale mask would let molecule 1, after molecule 0 converges and drops
    # out, read molecule 0's row and early-terminate at the wrong geometry.
    # Here molecule 0 (species 1) is below the tolerance and molecule 1
    # (species 6) is above it, so molecule 0 converges first and every
    # subsequent step runs on a one-row subset.
    for stable in (True, False):
        mixed = _run_constant_force({1: opttol / 2, 6: opttol * 2}, stable, opttol, species=(1, 6))
        assert mixed == [True, False], (
            "in a mixed batch each molecule must be judged on its own force "
            f"after the subset gather, got {mixed} (energy_stable={stable})"
        )


def test_fmax_ignores_padded_atoms():
    """The padded-atom force is ignored via an explicit atom_mask (audit C13).

    Species 0 is a real atom (hydrogen, in ANI2xt's convention) at index 0 of
    this molecule, and species 0 is *also* the value historically used to
    flag the padded last slot -- a genuine collision. Deriving the mask from
    `numbers == species_pad` would therefore mark BOTH the real atom at index
    0 and the padded slot at index 2 as padding, incorrectly zeroing the real
    atom's force too. The explicit atom_mask keeps the two unambiguous: only
    index 2 is padding, so the huge force placed on the real atom at index 0
    must be retained (not zeroed) while the padded slot's huge force is
    dropped.
    """
    import torch

    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    class MockNN:
        def forward_batched(self, coord, numbers, charges, atom_mask=None):
            e = torch.zeros(coord.shape[0])
            f = torch.zeros_like(coord)
            f[:, 0, :] = 100.0  # huge force on the real atom at index 0 (species 0)
            f[:, -1, :] = 100.0  # huge force on the (padded) last slot (also species 0)
            return e, f

    coord = torch.zeros(1, 3, 3)
    state = {
        "coord": coord,
        "numbers": torch.tensor([[0, 8, 0]]),  # index 0 is real H; last is pad, both species 0
        "charges": torch.zeros(1, dtype=torch.long),
        "nn": MockNN(),
        "converged_mask": torch.zeros(1, dtype=torch.bool),
        "fmax": torch.full((1,), 999.0),
        "energy": torch.full((1,), float("inf"), dtype=torch.double),
    }
    atom_mask = torch.tensor([[True, True, False]])  # only the last slot is padding
    n_steps(state, n=1, opttol=0.01, patience=5, atom_mask=atom_mask)
    # The real atom at index 0 keeps its force; only the padded slot's huge
    # force is dropped. A value-derived mask (numbers == 0) would have zeroed
    # index 0 too, collapsing fmax toward 0 instead.
    assert state["fmax"].item() > 50.0  # real slot-0 force retained, not zeroed


def test_stored_energy_matches_stored_coord():
    import torch

    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    class MockNN:
        def forward_batched(self, coord, numbers, charges, atom_mask=None):
            e = (coord**2).sum(dim=(1, 2))
            f = -2.0 * coord
            return e, f

    coord = torch.full((1, 2, 3), 0.5, dtype=torch.float)
    state = {
        "coord": coord.clone(),
        "numbers": torch.ones(1, 2, dtype=torch.long),
        "charges": torch.zeros(1, dtype=torch.long),
        "nn": MockNN(),
        "converged_mask": torch.zeros(1, dtype=torch.bool),
        "fmax": torch.full((1,), 999.0),
        "energy": torch.full((1,), float("inf"), dtype=torch.double),
    }
    n_steps(state, n=50, opttol=0.01, patience=40)
    recomputed = (state["coord"] ** 2).sum().item()
    assert abs(state["energy"].item() - recomputed) < 1e-3


def test_stored_fmax_matches_stored_coord():
    """Reported fmax must be the force norm at the reported (final) geometry.

    The loop measures forces at the pre-step geometry then always takes one more
    FIRE step, so the in-loop fmax lagged the stored post-step coordinates by a
    full step. fmax is now recomputed at the final geometry alongside energy.
    Running a still-moving molecule for only a few steps (no convergence) makes
    the one-step lag large, so this fails if fmax is not recomputed: with the
    harmonic potential E = sum(coord^2), F = -2*coord, the reported fmax must
    equal max-over-atoms of |2*coord| at the final coordinates.
    """
    import torch

    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    class MockNN:
        def forward_batched(self, coord, numbers, charges, atom_mask=None):
            e = (coord**2).sum(dim=(1, 2))
            f = -2.0 * coord
            return e, f

    # Start far from the minimum and take only a few steps with an
    # unreachable tolerance, so the molecule is still moving fast at the end
    # and the pre-step/post-step force norms differ well above the tolerance.
    coord = torch.full((1, 2, 3), 2.0, dtype=torch.float)
    state = {
        "coord": coord.clone(),
        "numbers": torch.ones(1, 2, dtype=torch.long),
        "charges": torch.zeros(1, dtype=torch.long),
        "nn": MockNN(),
        "converged_mask": torch.zeros(1, dtype=torch.bool),
        "fmax": torch.full((1,), 999.0),
        "energy": torch.full((1,), float("inf"), dtype=torch.double),
    }
    n_steps(state, n=3, opttol=1e-9, patience=1000)
    recomputed_fmax = (2.0 * state["coord"]).norm(dim=-1).max(dim=-1)[0]
    assert abs(state["fmax"].item() - recomputed_fmax.item()) < 1e-5


class TestConvergedAndFmaxDescribeTheSameGeometry:
    """``Converged=True`` and the reported ``fmax`` must not contradict.

    ``batchopt`` writes both to every output record: ``Converged`` from
    ``converged_mask``, and ``fmax`` recomputed at the final geometry. The loop
    decided convergence from the force measured *before* its last FIRE step and
    then took that step anyway, so the flag described one geometry and the number
    beside it described another. A consumer filtering on ``fmax <= opt_tol`` and
    one filtering on ``Converged == "True"`` got different sets out of the same
    file, and the file asserted both.

    The discrepancy scales with stiffness, which is why it was easy to dismiss: a
    soft potential moves little in one step and stays inside the tolerance. On a
    stiff one, measured before the fix, ``Converged=True`` came back with fmax up
    to 6.9x the tolerance.
    """

    class _Harmonic:
        """E = k*sum(coord^2), F = -2*k*coord. Exact, hermetic, no NNP."""

        def __init__(self, k: float):
            self.k = k

        def forward_batched(self, coord, numbers, charges, atom_mask=None):
            return self.k * (coord**2).sum(dim=(1, 2)), -2.0 * self.k * coord

    @staticmethod
    def _run(k: float, start: float, opttol: float) -> tuple[bool, float]:
        state = {
            "coord": torch.full((1, 2, 3), start, dtype=torch.float),
            "numbers": torch.ones(1, 2, dtype=torch.long),
            "charges": torch.zeros(1, dtype=torch.long),
            "nn": TestConvergedAndFmaxDescribeTheSameGeometry._Harmonic(k),
            "converged_mask": torch.zeros(1, dtype=torch.bool),
            "fmax": torch.full((1,), 999.0),
            "energy": torch.full((1,), float("inf"), dtype=torch.double),
        }
        # patience above n so nothing leaves the active set as oscillating:
        # converged_mask would then be True for a structure that never met the
        # force criterion, and this assertion would be about the wrong thing.
        n_steps(state, n=2000, opttol=opttol, patience=5000)
        return bool(state["converged_mask"][0]), float(state["fmax"][0])

    # Stiffnesses spanning the soft case (where the inconsistency hides below
    # the tolerance) and the stiff case (where it was 2x-7x the tolerance).
    @pytest.mark.parametrize("k", [1.0, 10.0, 100.0])
    @pytest.mark.parametrize("start", [0.5, 1.0, 1.5, 3.0])
    def test_a_structure_reported_converged_reports_a_converged_force(self, k, start):
        opttol = 0.01
        converged, fmax = self._run(k, start, opttol)

        assert converged, "test premise: this configuration should converge"
        assert fmax <= opttol, (
            f"reported Converged=True beside fmax={fmax:.6f}, which is "
            f"{fmax / opttol:.1f}x the {opttol} eV/A tolerance: the flag and the "
            f"force in the same record describe different geometries"
        )


class TestNStepsIntegration:
    """Integration tests for n_steps with realistic scenarios."""

    def test_n_steps_with_decreasing_forces(self):
        """n_steps should converge structures with decreasing forces."""
        step_count = [0]

        def mock_forward(coord, numbers, charges, atom_mask=None):
            step_count[0] += 1
            batch_size = coord.shape[0]
            # Simulate forces that decrease with each step
            force_magnitude = max(0.001, 0.5 / step_count[0])
            return torch.ones(batch_size) * (-step_count[0]), torch.ones(
                batch_size, coord.shape[1], 3
            ) * force_magnitude

        mock_nn = MagicMock()
        mock_nn.forward_batched.side_effect = mock_forward

        state = {
            "numbers": torch.ones(3, 5, dtype=torch.long),
            "charges": torch.zeros(3, dtype=torch.long),
            "coord": torch.randn(3, 5, 3),
            "nn": mock_nn,
            "converged_mask": torch.tensor([False, False, False]),
            "fmax": torch.full((3,), 999.0),
            "energy": torch.full((3,), 999.0, dtype=torch.double),
        }

        n_steps(state, n=100, opttol=0.01, patience=50)

        # Should have converged
        assert state["converged_mask"].all()
        # fmax should be updated to reasonable values
        assert (state["fmax"] < 999.0).all()


class TestOptimizationEngineImportPaths:
    """This module is the home of ``n_steps`` and ``print_stats``.

    Two tests here used to assert both names were also reachable through
    ``Auto3D.engines.batch_opt.batchopt``, which is what kept that compat barrel alive.
    ``print_stats`` is no longer imported there at all (nothing in ``batchopt``
    called it); ``n_steps`` still is, because ``batchopt`` uses it, but reaching
    it that way is forbidden by ``tests/test_import_boundaries.py``.
    """

    def test_n_steps_and_print_stats_live_here(self):
        from Auto3D.engines.batch_opt.optimization_engine import (
            n_steps as n_steps_home,
        )
        from Auto3D.engines.batch_opt.optimization_engine import (
            print_stats as print_stats_home,
        )

        assert n_steps_home is n_steps
        assert print_stats_home is print_stats


class TestPrintStatsBackwardCompatibility:
    """Tests ensuring print_stats maintains backward compatibility."""

    def test_print_stats_uses_oscillating_count_key(self, caplog):
        """print_stats should use 'oscillating_count' key (correct spelling)."""
        state = {
            "numbers": torch.ones(3, 5),
            "converged_mask": torch.tensor([True, False, False]),
            "oscillating_count": torch.tensor([[100], [0], [0]], dtype=torch.float),
        }

        # Should work with correctly spelled key
        with caplog.at_level(logging.INFO):
            print_stats(state, patience=50)

        assert "Dropped(Oscillating): 1" in caplog.text
