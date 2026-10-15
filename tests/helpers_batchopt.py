# tests/helpers_batchopt.py
"""The ensemble_opt double shared by the optimizer tests.

``test_batchopt.py`` and ``test_e_tot_units.py`` each monkeypatch
``Auto3D.engines.batch_opt.batchopt.ensemble_opt`` with a double that must
return exactly the shape the real function does -- the seven keys
``test_ensemble_opt_returns_exactly_the_keys_its_callers_read`` pins. Before
this module the two doubles were byte-alike copies that both still built the
pre-trim ``ids``/``he``/``close``/``timing`` keys the real function dropped,
so a re-added reader of one of those keys would have passed against the
doubles while failing against ``ensemble_opt`` itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def make_fake_ensemble_opt(
    energies: Sequence[float] | None = None,
) -> Callable:
    """Build a conforming ``ensemble_opt`` double.

    Every structure is reported converged, with zero force, not oscillating,
    and finite -- callers that need a different convergence/force shape stub
    something else. ``energies`` (eV, indexed by position) defaults to all
    zeros; the double slices it to the batch size it is actually called with.
    """

    def fake_ensemble_opt(
        net, coord, numbers, charges, param, device, atom_mask=None, progress_cb=None, mults=None
    ):
        n = len(coord)
        values = list(energies[:n]) if energies is not None else [0.0] * n
        return dict(
            coord=coord.tolist(),
            energy=values,
            fmax=[0.0] * n,
            numbers=numbers.tolist(),
            converged_mask=[True] * n,
            oscillating_count=[0] * n,
            non_finite=[False] * n,
        )

    return fake_ensemble_opt
