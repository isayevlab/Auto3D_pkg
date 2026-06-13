# tests/test_progress.py
"""Unit tests for the live-progress plumbing: the count helper that feeds events
and the display's multi-job aggregation. No NNP/optimization runs here."""
from __future__ import annotations

import torch

from Auto3D.batch_opt.optimization_engine import optimization_counts
from Auto3D.cli.progress import OptimizationDisplay


def test_optimization_counts():
    # 5 structures; converged_mask marks 3 as done; one of those (osc>=patience)
    # is a drop, so converged=2, dropped=1, active=5-3=2.
    state = {
        "numbers": torch.zeros(5, 3),
        "converged_mask": torch.tensor([True, True, False, False, True]),
        "oscillating_count": torch.tensor([0, 5, 0, 0, 1]),
    }
    assert optimization_counts(state, patience=3) == (5, 2, 1, 2)


def test_optimization_counts_none_converged():
    state = {
        "numbers": torch.zeros(4, 3),
        "converged_mask": torch.tensor([False, False, False, False]),
        "oscillating_count": torch.tensor([0, 0, 0, 0]),
    }
    assert optimization_counts(state, patience=3) == (4, 0, 0, 4)


def test_display_single_job():
    d = OptimizationDisplay(0)
    d.update_from_jobs({1: {"total": 5, "converged": 2, "dropped": 1, "active": 2, "step": 30}})
    assert (d.total, d.converged, d.dropped, d.active, d.step) == (5, 2, 1, 2, 30)
    d.make_panel()  # must not raise


def test_display_multi_job_aggregates():
    d = OptimizationDisplay(0)
    d.update_from_jobs({
        1: {"total": 5, "converged": 2, "dropped": 1, "active": 2, "step": 30},
        2: {"total": 3, "converged": 3, "dropped": 0, "active": 0, "step": 50},
    })
    assert d.total == 8
    assert d.converged == 5
    assert d.dropped == 1
    assert d.active == 2
    assert d.step == 50  # furthest along


def test_display_empty_jobs_is_noop():
    d = OptimizationDisplay(0)
    d.update_from_jobs({})
    assert (d.total, d.converged, d.dropped, d.active, d.step) == (0, 0, 0, 0, 0)
