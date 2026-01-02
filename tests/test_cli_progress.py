# tests/test_cli_progress.py
"""Tests for progress display components."""

import pytest


def test_optimization_display_exists():
    """OptimizationDisplay should exist."""
    from Auto3D.cli.progress import OptimizationDisplay
    assert OptimizationDisplay is not None


def test_optimization_display_update():
    """OptimizationDisplay should track stats."""
    from Auto3D.cli.progress import OptimizationDisplay

    display = OptimizationDisplay(total_structures=100)
    display.update(converged=10, active=85, dropped=5, step=100, best_energy=-342.5)

    assert display.converged == 10
    assert display.active == 85
    assert display.dropped == 5


def test_optimization_display_panel():
    """OptimizationDisplay should create a Rich panel."""
    from Auto3D.cli.progress import OptimizationDisplay
    from rich.panel import Panel

    display = OptimizationDisplay(total_structures=100)
    panel = display.make_panel()

    assert isinstance(panel, Panel)


def test_create_progress():
    """create_progress should return Progress object."""
    from Auto3D.cli.progress import create_progress
    from rich.progress import Progress

    progress = create_progress()
    assert isinstance(progress, Progress)


def test_isomer_progress_callback():
    """IsomerProgressCallback should exist and be callable."""
    from Auto3D.cli.progress import IsomerProgressCallback, create_progress

    progress = create_progress()
    task_id = progress.add_task("test", total=100)
    callback = IsomerProgressCallback(progress, task_id)

    # Should not raise
    callback(50, 100)
