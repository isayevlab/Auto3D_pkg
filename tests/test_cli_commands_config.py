"""Tests for Auto3D.cli.commands.config's module-level preset data.

Presets are static data nobody re-reads once shipped -- exactly how the
`thorough` preset shipped with both `k=10` and `window=5.0` set (M28), a
combination `ConformerRanker.run` silently resolved by always preferring `k`,
making `window` inert with no error, warning, or test failure. This module
is the guard that would have caught that: every preset must be internally
self-consistent.
"""

from __future__ import annotations

from Auto3D.cli.commands.config import PRESETS


def test_no_preset_sets_both_k_and_window():
    """No shipped preset may set both k and window.

    They are alternative conformer-selection strategies (top-k vs. an
    energy window); ConformerRanker.run only consults one, so a preset
    setting both would silently make one of them a dead key -- the exact
    defect this test exists to prevent from shipping again.
    """
    for name, preset in PRESETS.items():
        assert not ("k" in preset and "window" in preset), (
            f"preset {name!r} sets both k and window, so one is silently inert: {preset!r}"
        )
