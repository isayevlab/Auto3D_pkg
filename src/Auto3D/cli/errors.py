# src/Auto3D/cli/errors.py
"""Error handling for Auto3D CLI."""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel
from rich.traceback import Traceback

from Auto3D.cli.console import emit_json, error_console
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

# Ctrl-C. 128 + SIGINT(2) is the shell convention for "terminated by a signal",
# so `auto3d run ...; echo $?` after Ctrl-C answers what every other well-behaved
# program answers. Deliberately not folded into EXIT_CODES: that table maps
# *exception classes* through exit_code_for, and KeyboardInterrupt is a
# BaseException that never reaches an `except Exception` handler -- which is
# exactly why an interrupted run used to report nothing at all.
#
# The value is not novel for the Typer commands: typer/core.py already turns an
# escaping KeyboardInterrupt into click's Exit(130). Naming it here is what lets
# the deprecated `auto3d config.yaml` runner -- which is not a Typer command, and
# so dumped a raw traceback and exited 1 -- agree with them, and what lets both
# report something before they go.
EXIT_INTERRUPTED = 130


def exit_code_for(error: Exception) -> int:
    """Return the differentiated exit code for an exception (1 if unmapped)."""
    for exc_type, code in EXIT_CODES.items():
        if isinstance(error, exc_type):
            return code
    return 1


def get_error_hint(error: Auto3DError) -> str | None:
    """Get a helpful hint for an error.

    A per-raise ``hint`` set on the exception wins over the per-class hints
    below, including when it is empty: the class hints are broad guesses from
    the exception *type* alone, and a raise site that knows better must be
    able to say "nothing useful to add" as well as "say this instead". The
    output-overwrite refusal is the motivating case -- it is a
    ``ConfigurationError``, so it inherited "Run 'auto3d config init' to
    generate a valid config file", which is a non-sequitur for
    ``-o precious.sdf`` and would have become one of the most frequently
    printed hints in the CLI.

    Args:
        error: The error that occurred.

    Returns:
        Helpful hint string, or None.
    """
    explicit_hint = getattr(error, "hint", None)
    if explicit_hint is not None:
        return explicit_hint or None

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


def handle_error(error: Exception, verbose: int = 0, json_output: bool = False) -> None:
    """Handle an error with Rich formatting.

    Args:
        error: The error to handle.
        verbose: CLI verbosity level (the ``-v``/``--verbose`` count). At 0,
            a known ``Auto3DError`` still gets only its clean message + hint
            -- that panel *is* the intended, actionable presentation. Any
            value above 0 additionally prints a full traceback to stderr, so
            an internal failure (a bare ``KeyError('ID')`` from a missing SDF
            property, say) is debuggable without editing source (M30).
        json_output: The command was invoked with ``--json``. The human panel
            below still goes to stderr -- diagnostics belong there and a
            terminal user must not lose them -- but stdout additionally gets a
            machine-readable failure document, so a caller that parses stdout
            gets a parseable answer on the failure path too instead of an
            empty stream. Emitted *before* ``SystemExit`` is raised, for the
            same reason ``execute_run`` emits its results document before
            ``_exit_if_incomplete``.
    """
    if json_output:
        emit_json({
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "hint": get_error_hint(error) if isinstance(error, Auto3DError) else None,
            "exit_code": exit_code_for(error),
        })

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


def job_directory_hint(path: str | Path | None, job_name: str | None) -> str | None:
    """Where this run's job directory is, as far as the CLI can know it.

    ``WorkflowOrchestrator`` builds the job directory as
    ``<input parent>/<input stem>_<job_name>`` and, when ``job_name`` is empty,
    invents a timestamp for it *inside its own private copy of the config*
    (``run()`` does ``replace(self.config)`` before ``_validate_input`` fills the
    name in). So the caller can name the directory exactly when it supplied
    ``--job-name``/``job_name:`` and can only name the pattern otherwise; this
    returns whichever of the two is true rather than guessing a timestamp or
    globbing for a directory that a *different* run may have left behind.

    Returns None when there is no input path to derive anything from.
    """
    if not path:
        return None
    input_path = Path(path)
    stem = f"{input_path.stem}_{job_name}" if job_name else f"{input_path.stem}_<timestamp>"
    return str(input_path.parent / stem)


def handle_interrupt(
    job_hint: str | None = None,
    batch: dict[str, int] | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Report what is known about a Ctrl-C'd run, then exit :data:`EXIT_INTERRUPTED`.

    An interrupted run used to print nothing whatsoever -- ``KeyboardInterrupt``
    is a ``BaseException``, so neither ``execute_run``'s ``except Exception`` nor
    the legacy runner's saw it, and the user was left with a bare prompt, no idea
    how far the run had got and no idea whether anything reached disk.

    Every field is optional and omitted when unknown, because the point of this
    panel is to state what is known and nothing else:

    Args:
        job_hint: Job directory (or the pattern it follows) from
            :func:`job_directory_hint`.
        batch: The most recent optimizer progress counts, if any arrived --
            ``{"converged", "active", "dropped", "step"}``. These describe the
            batch that was in the optimizer, not the whole run (see
            ``Auto3D.cli.progress.OptimizationDisplay``), and are labelled that
            way; claiming them as run totals would be the same defect this
            module's neighbours were just cleared of.
        elapsed_seconds: Wall-clock time before the signal arrived.

    Raises:
        SystemExit: Always, with :data:`EXIT_INTERRUPTED`.
    """
    from Auto3D.cli.results import format_duration

    lines = ["[yellow]Interrupted by the user (Ctrl-C).[/yellow]"]
    if elapsed_seconds is not None:
        lines.append(f"Ran for {format_duration(elapsed_seconds)} before the signal arrived.")
    if batch:
        lines.append(
            "\nOptimizer batch in flight: "
            f"[green]{batch.get('converged', 0)}[/green] converged, "
            f"[yellow]{batch.get('active', 0)}[/yellow] active, "
            f"[red]{batch.get('dropped', 0)}[/red] dropped, "
            f"at step {batch.get('step', 0)}."
            "\n[dim]Counts describe that batch, not the whole run.[/dim]"
        )
    if job_hint:
        lines.append(
            f"\nAnything already written is under the job directory:\n  [cyan]{job_hint}[/cyan]"
            "\n[dim]No output SDF is combined for an interrupted run.[/dim]"
        )

    error_console.print(Panel(
        "\n".join(lines),
        title="[yellow]Interrupted[/yellow]",
        border_style="yellow",
    ))
    raise SystemExit(EXIT_INTERRUPTED)
