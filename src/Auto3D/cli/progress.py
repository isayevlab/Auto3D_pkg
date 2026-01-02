# src/Auto3D/cli/progress.py
"""Progress display components for Auto3D CLI.

This module provides Rich-based progress bars and live status displays
for long-running operations.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from Auto3D.cli.console import console


def create_progress() -> Progress:
    """Create a Rich progress bar with standard columns.

    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


class OptimizationDisplay:
    """Live-updating display during geometry optimization.

    This class tracks optimization statistics and renders them as a
    Rich panel suitable for use with Live display.

    Attributes:
        total: Total number of structures.
        converged: Number of converged structures.
        active: Number of actively optimizing structures.
        dropped: Number of dropped (oscillating) structures.
        step: Current optimization step.
        best_energy: Best (lowest) energy found.
    """

    def __init__(self, total_structures: int) -> None:
        """Initialize display with total structure count.

        Args:
            total_structures: Total number of structures to optimize.
        """
        self.total = total_structures
        self.converged = 0
        self.active = total_structures
        self.dropped = 0
        self.step = 0
        self.best_energy: float | None = None

    def update(
        self,
        converged: int,
        active: int,
        dropped: int,
        step: int,
        best_energy: float | None = None,
    ) -> None:
        """Update optimization statistics.

        Args:
            converged: Number of converged structures.
            active: Number of active structures.
            dropped: Number of dropped structures.
            step: Current step number.
            best_energy: Best energy found (optional).
        """
        self.converged = converged
        self.active = active
        self.dropped = dropped
        self.step = step
        if best_energy is not None:
            self.best_energy = best_energy

    def make_panel(self) -> Panel:
        """Create a Rich panel showing current status.

        Returns:
            Panel with progress bar and statistics.
        """
        # Calculate progress
        completed = self.converged + self.dropped
        pct = completed / self.total if self.total > 0 else 0

        # Create progress bar
        filled = int(pct * 30)
        bar = "━" * filled + ("╺" if filled < 30 else "") + "─" * (29 - filled)

        # Stats grid
        stats = Table.grid(padding=(0, 3))
        stats.add_row(
            "[green]Converged[/green]", f"[green]{self.converged}[/green]",
            "[yellow]Active[/yellow]", f"[yellow]{self.active}[/yellow]",
            "[red]Dropped[/red]", f"[red]{self.dropped}[/red]",
        )

        # Build content
        content = f"{bar} {pct:.0%}  Step {self.step}\n\n"

        if self.best_energy is not None:
            content += f"[dim]Best energy: {self.best_energy:.2f} kcal/mol[/dim]"

        return Panel(content, title="[bold]Optimizing[/bold]", border_style="blue")


class IsomerProgressCallback:
    """Callback for isomer enumeration progress."""

    def __init__(self, progress: Progress, task_id) -> None:
        """Initialize callback.

        Args:
            progress: Rich Progress instance.
            task_id: Progress task ID.
        """
        self.progress = progress
        self.task_id = task_id

    def __call__(self, current: int, total: int) -> None:
        """Update progress.

        Args:
            current: Current item number.
            total: Total items.
        """
        self.progress.update(self.task_id, completed=current, total=total)
