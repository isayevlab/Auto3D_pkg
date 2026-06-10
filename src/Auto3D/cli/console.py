"""Rich console utilities for Auto3D CLI.

This module provides a console singleton with auto-detection of terminal
capabilities and helper functions for consistent output formatting.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    pass


def create_console() -> Console:
    """Create a Rich console with auto-detected terminal capabilities.

    Returns:
        Console configured for the current terminal.
    """
    # Force plain text if piped or redirected
    force_terminal = None
    if not sys.stdout.isatty():
        force_terminal = False

    return Console(
        force_terminal=force_terminal,
        highlight=True,
        emoji=False,
    )


# Global console singleton
console = create_console()

# Separate console for stderr output
error_console = Console(stderr=True, highlight=True, emoji=False)


def print_banner(input_path: str, engine: str, gpu_info: str, output_info: str) -> None:
    """Print the startup banner with run configuration.

    Args:
        input_path: Path to input file.
        engine: Optimization engine name.
        gpu_info: GPU configuration string.
        output_info: Output configuration (k=N or window=X).
    """
    import Auto3D

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()

    grid.add_row("Input:", f"[cyan]{input_path}[/cyan]")
    grid.add_row("Engine:", engine)
    grid.add_row("GPU:", gpu_info)
    grid.add_row("Output:", output_info)

    console.print(Panel(grid, title=f"[bold]Auto3D v{Auto3D.__version__}[/bold]", border_style="blue"))


def print_success(message: str) -> None:
    """Print a success message with green checkmark.

    Args:
        message: Success message to display.
    """
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow indicator.

    Args:
        message: Warning message to display.
    """
    error_console.print(f"[yellow]⚠[/yellow] {message}")


def print_error(message: str, hint: str | None = None) -> None:
    """Print an error message in a panel.

    Args:
        message: Error message to display.
        hint: Optional hint for resolving the error.
    """
    content = f"[red]{message}[/red]"
    if hint:
        content += f"\n\n[dim]{hint}[/dim]"

    panel = Panel(
        content,
        title="[red]Error[/red]",
        border_style="red",
    )
    error_console.print(panel)
