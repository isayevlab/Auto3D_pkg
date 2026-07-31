# src/Auto3D/cli/errors.py
"""Error handling for Auto3D CLI."""

from __future__ import annotations

from rich.panel import Panel
from rich.traceback import Traceback

from Auto3D.cli.console import error_console
from Auto3D.exceptions import (
    Auto3DError,
    ConfigurationError,
    DependencyError,
    GPUError,
    InputValidationError,
    ModelError,
)

# Differentiated exit codes for scripting/CI. 1 = generic; the rest let callers
# branch on error class. (Click reserves 2 for usage errors, which aligns with
# our configuration/input errors below.)
EXIT_CODES: dict[type, int] = {
    ConfigurationError: 2,
    InputValidationError: 2,
    DependencyError: 3,
    GPUError: 4,
    ModelError: 5,  # includes ModelLoadError / NumericalError
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


def handle_error(error: Exception, verbose: int = 0) -> None:
    """Handle an error with Rich formatting.

    Args:
        error: The error to handle.
        verbose: CLI verbosity level (the ``-v``/``--verbose`` count). At 0,
            a known ``Auto3DError`` still gets only its clean message + hint
            -- that panel *is* the intended, actionable presentation. Any
            value above 0 additionally prints a full traceback to stderr, so
            an internal failure (a bare ``KeyError('ID')`` from a missing SDF
            property, say) is debuggable without editing source (M30).
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
        # Anything that isn't an Auto3DError is by definition unexpected --
        # an internal bug, not a recognized user-facing failure mode. Before
        # this fix the panel showed only `str(error)`, so a missing-property
        # KeyError rendered as a bare, unactionable 'ID'. Always name the
        # exception type and always point at --verbose, even at verbose=0:
        # the user has nothing to act on otherwise, and nothing worth
        # reporting in a bug (see get_error_hint's docstring for the
        # Auto3DError case, which already has a real hint).
        error_console.print(Panel(
            f"[red]{type(error).__name__}: {error}[/red]"
            "\n\n[dim]Run with -v/--verbose for a full traceback.[/dim]",
            title="[red]Unexpected Error[/red]",
            border_style="red",
        ))

    if verbose > 0:
        # A fixed, generous width avoids the traceback panel wrapping at
        # whatever narrow width the ambient terminal/pipe happens to report
        # (source lines and file paths get split across lines otherwise).
        error_console.print(
            Traceback.from_exception(type(error), error, error.__traceback__),
            width=200,
        )

    raise SystemExit(exit_code_for(error))
