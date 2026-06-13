# src/Auto3D/cli/errors.py
"""Error handling for Auto3D CLI."""

from __future__ import annotations

from rich.panel import Panel

from Auto3D.cli.console import error_console
from Auto3D.exceptions import (
    Auto3DError,
    ConfigurationError,
    DependencyError,
    GPUError,
    InputValidationError,
    ModelError,
    ModelNotFoundError,
)

# Differentiated exit codes for scripting/CI. 1 = generic; the rest let callers
# branch on error class. (Click reserves 2 for usage errors, which aligns with
# our configuration/input errors below.)
EXIT_CODES: dict[type, int] = {
    ConfigurationError: 2,
    InputValidationError: 2,
    DependencyError: 3,
    GPUError: 4,
    ModelError: 5,  # includes ModelNotFoundError / ModelLoadError / NumericalError
}


def exit_code_for(error: Exception) -> int:
    """Return the differentiated exit code for an exception (1 if unmapped)."""
    for exc_type, code in EXIT_CODES.items():
        if isinstance(error, exc_type):
            return code
    return 1


def get_error_hint(error: Auto3DError) -> str | None:
    """Get a helpful hint for an error.

    Args:
        error: The error that occurred.

    Returns:
        Helpful hint string, or None.
    """
    if isinstance(error, ConfigurationError):
        return "Run 'auto3d config init' to generate a valid config file"

    if isinstance(error, InputValidationError):
        return "Run 'auto3d validate <file>' to check your input file"

    if isinstance(error, ModelNotFoundError):
        return "Available engines: AIMNET, ANI2x, ANI2xt\nRun 'auto3d models list' for details"

    if isinstance(error, GPUError):
        return "Try --no-gpu to run on CPU, or check CUDA installation"

    if isinstance(error, DependencyError):
        dep = getattr(error, "dependency_name", "unknown")
        hints = {
            "openeye": "Install: conda install -c openeye openeye-toolkits",
            "torchani": "Install: pip install torchani",
            "ase": "Install: pip install ase",
        }
        return hints.get(dep, f"Install the missing dependency: {dep}")

    return None


def handle_error(error: Exception) -> None:
    """Handle an error with Rich formatting.

    Args:
        error: The error to handle.
    """
    if isinstance(error, Auto3DError):
        error_type = type(error).__name__.replace("Error", " Error")
        hint = get_error_hint(error)

        content = f"[red]{error}[/red]"
        if hint:
            content += f"\n\n[dim]{hint}[/dim]"

        error_console.print(Panel(
            content,
            title=f"[red]{error_type}[/red]",
            border_style="red",
        ))
    else:
        error_console.print(Panel(
            f"[red]{error}[/red]",
            title="[red]Error[/red]",
            border_style="red",
        ))

    raise SystemExit(exit_code_for(error))
