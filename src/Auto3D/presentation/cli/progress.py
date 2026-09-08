"""Live optimization status panel for the Auto3D CLI.

This module renders the counts the optimizer workers emit. It deliberately
renders **no percentage and no progress bar**: see
:meth:`OptimizationDisplay.make_panel` for why there is no honest denominator
available at this layer.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table


class OptimizationDisplay:
    """Live-updating display of the optimizer batch currently in flight.

    Fed by :meth:`update_from_jobs` from the per-step events
    ``Auto3D.engines.batch_opt.optimization_engine.n_steps`` emits, and rendered into a
    Rich ``Live`` by ``Auto3D.presentation.cli.commands.run``.

    Attributes:
        total: Structures in the batch(es) currently being optimized.
        converged: Structures in those batches that have converged.
        active: Structures in those batches still being optimized.
        dropped: Structures in those batches dropped for oscillating.
        step: Furthest optimizer step reached by any reporting worker.
    """

    def __init__(self, total_structures: int) -> None:
        """Initialize display with a starting structure count.

        Args:
            total_structures: Structures in the first batch, if already known;
                callers that learn the count only from the first event pass 0.
        """
        self.total = total_structures
        self.converged = 0
        self.active = total_structures
        self.dropped = 0
        self.step = 0

    def update_from_jobs(self, jobs: dict) -> None:
        """Aggregate the latest per-job progress event into the display.

        Each value in ``jobs`` is the most recent event from one optimizer
        worker (``{"total", "converged", "dropped", "active", "step"}``). Those
        counts describe **the batch that worker currently has in the
        optimizer**, not its whole share of the run: a worker pulls chunk after
        chunk off the queue and ``n_steps`` starts over -- new ``total``, step
        back to 1 -- for each one. Summing across workers therefore yields the
        set of structures in flight right now, and that is all this display
        claims.

        An earlier version of this method described the result as "exactly its
        own progress" for a single optimizer and rendered it as a percentage of
        ``total``. It was neither: the percentage ran ``25% -> 75% -> 100% ->
        6% -> 100% -> 2%`` across successive batches, because each new batch
        reset both halves of the fraction.
        """
        if not jobs:
            return
        self.total = sum(j.get("total", 0) for j in jobs.values())
        self.converged = sum(j.get("converged", 0) for j in jobs.values())
        self.dropped = sum(j.get("dropped", 0) for j in jobs.values())
        self.active = sum(j.get("active", 0) for j in jobs.values())
        self.step = max((j.get("step", 0) for j in jobs.values()), default=0)

    def as_batch_counts(self) -> dict[str, int]:
        """Return the current counts as a plain dict (for the interrupt report)."""
        return {
            "total": self.total,
            "converged": self.converged,
            "active": self.active,
            "dropped": self.dropped,
            "step": self.step,
        }

    def make_panel(self) -> Panel:
        """Create a Rich panel showing the in-flight batch.

        No bar and no percentage, on purpose. A fraction needs a denominator
        that is the whole job, and this layer does not have one: the number of
        structures a run will optimize is not known until stereoisomer and
        tautomer enumeration have finished, which happens *while* the optimizer
        is already consuming the earlier chunks. The two denominators that were
        available were both lies -- the optimizer's own ``tqdm`` divided by the
        *step budget* (so a run converging at step 300 of 2000 showed 15% and
        then vanished, and a run where nothing converged showed 100%), and this
        panel divided by the current batch size (so the percentage sawtoothed
        back down on every new chunk).

        Returns:
            Panel with the current counts and step.
        """
        stats = Table.grid(padding=(0, 3))
        stats.add_row(
            "[green]Converged[/green]",
            f"[green]{self.converged}[/green]",
            "[yellow]Active[/yellow]",
            f"[yellow]{self.active}[/yellow]",
            "[red]Dropped[/red]",
            f"[red]{self.dropped}[/red]",
        )

        content = Table.grid()
        content.add_row(stats)
        content.add_row(f"\n[dim]{self.total} structures in this batch, step {self.step}[/dim]")

        return Panel(
            content,
            title="[bold]Optimizing (current batch)[/bold]",
            border_style="blue",
        )
