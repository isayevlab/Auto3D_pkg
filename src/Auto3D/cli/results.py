# src/Auto3D/cli/results.py
"""Results display components for Auto3D CLI."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.panel import Panel
from rich.table import Table

from Auto3D.cli.console import console, emit_json


@dataclass
class FailedMolecule:
    """Information about a failed molecule."""

    name: str
    error: str


@dataclass
class WorkflowResults:
    """Results from a workflow run."""

    success_count: int
    failed_count: int
    total_conformers: int
    output_path: str
    elapsed_seconds: float
    failures: list[FailedMolecule] = field(default_factory=list)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 2m 3s" or "45s".
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def print_results_summary(results: WorkflowResults) -> None:
    """Print a summary panel of workflow results.

    Args:
        results: Workflow results to display.
    """
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()

    stats.add_row("Molecules:", f"[green]{results.success_count}[/green] succeeded")
    if results.failed_count > 0:
        stats.add_row("", f"[red]{results.failed_count}[/red] failed")
    stats.add_row("Conformers:", f"{results.total_conformers} generated")
    stats.add_row("Output:", f"[cyan]{results.output_path}[/cyan]")
    stats.add_row("Time:", format_duration(results.elapsed_seconds))

    border_style = "green" if results.failed_count == 0 else "yellow"
    title_style = "bold green" if results.failed_count == 0 else "bold yellow"

    console.print(Panel(stats, title=f"[{title_style}]Results[/{title_style}]", border_style=border_style))


def print_failures(failures: list[FailedMolecule], verbose: bool = False) -> None:
    """Print information about failed molecules.

    Args:
        failures: List of failed molecules.
        verbose: If True, show detailed error table.
    """
    if not failures:
        return

    console.print(f"\n[yellow]Warning: {len(failures)} molecules failed[/yellow]")

    if verbose:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Molecule")
        table.add_column("Error")

        for f in failures[:20]:
            table.add_row(f.name, f"[dim]{f.error}[/dim]")

        if len(failures) > 20:
            table.add_row("...", f"[dim]+{len(failures) - 20} more[/dim]")

        console.print(table)
    else:
        console.print("[dim]Run with -v to see details[/dim]")


def count_from_output(output_path: str) -> tuple[int, int]:
    """Return (unique_molecule_count, conformer_count) from an output SDF.

    Thin back-compat wrapper around :func:`Auto3D.results.count_output` (the
    single source of truth). ``main()`` now returns a ``WorkflowResult`` that
    carries these counts, so the CLI reads them off the result instead.
    """
    from Auto3D.results import count_output

    return count_output(output_path)


def output_json(results: WorkflowResults) -> None:
    """Output results as JSON.

    Args:
        results: Workflow results to output.
    """
    emit_json({
        "success": results.failed_count == 0,
        "molecules": results.success_count,
        "failed": results.failed_count,
        "conformers": results.total_conformers,
        "output_file": results.output_path,
        "elapsed_seconds": results.elapsed_seconds,
        "failures": [{"name": f.name, "error": f.error} for f in results.failures],
    })
