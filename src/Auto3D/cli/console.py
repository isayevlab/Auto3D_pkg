"""Rich console utilities for Auto3D CLI.

This module owns *where CLI output goes*: the stdout stream reserved for
Auto3D's own output, the stderr stream diagnostics go to, and the containment
that keeps third-party libraries out of the first one.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import IO, TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# stdout containment
#
# `auto3d ... --json` promises a stdout stream carrying nothing but the JSON
# document, and `--quiet` promises no chatter. Neither promise is ours alone to
# keep. Resolving an engine name (cli/config_schema.py's `_validate_engine`,
# which runs while the configuration is being built -- before any output
# decision is made) imports `aimnet.calculators`, which pulls in `warp`, which
# prints a 734-byte device banner to *stdout* at import time. There is no
# Auto3D `print` to put a `if not json_output` in front of: the write is not
# ours, so guarding our own output cannot fix it. That is why
# `auto3d run in.smi --k 1 --json | jq .` could never parse.
#
# So the CLI reserves stdout for its own output for the whole life of a
# command: `sys.stdout` is pointed at stderr and the real stdout is kept here,
# behind `document_stream()` -- which is what `console` and `emit_json` write
# to. Redirecting rather than discarding is deliberate: a library that prints a
# genuine failure message to stdout must still reach the user, and stderr is
# where diagnostics belong. `--quiet` layers `suppress_foreign_stdout` on top,
# which holds that output back and releases it to stderr only if the command
# fails -- so "quiet" never means "silently drop the one line that explained
# the crash".
#
# Installed at the two places that own a whole command body:
# `Auto3D.cli.app._ReservedStdoutCommand` (every Typer command, however the
# app is driven -- console script, `python -m`, an embedding caller, click's
# CliRunner) and `Auto3D.auto3Dcli._run_legacy_yaml` (the deprecated
# `auto3d config.yaml` form, which reaches the same engine resolution). Both
# begin before the command does any work and therefore before the first
# import that can reach `aimnet`, which is the point: containment installed
# *after* the transitive import already happened would be a fix that works in
# a fresh process and silently does nothing in one where something imported
# `aimnet` first. Nesting is a no-op, so overlapping installs are safe.
#
# Deliberately *not* installed around the whole process (nor in the Typer
# group callback): click prints `--help` and usage errors while parsing a
# command's parameters, before its body runs. Those are legitimate stdout
# output, and an earlier version of this fix redirected them all to stderr.
# The spawned optimizer workers are a third process-level case and handle it
# themselves -- see `Auto3D.workflow_workers._worker_stdout_to_stderr`.
# ---------------------------------------------------------------------------

_real_stdout: IO[str] | None = None


def document_stream() -> IO[str]:
    """Return the stream Auto3D's own output belongs on.

    The process's real stdout while :func:`reserve_stdout` is active, plain
    ``sys.stdout`` otherwise. Resolved on every call rather than captured
    once, because both pytest and click's ``CliRunner`` replace ``sys.stdout``
    per invocation -- a stream captured when this module was first imported
    would be a stale buffer belonging to some earlier test.
    """
    return _real_stdout if _real_stdout is not None else sys.stdout


@contextlib.contextmanager
def reserve_stdout() -> Iterator[None]:
    """Reserve stdout for Auto3D's own output for the duration of a command.

    Writes any third party makes to ``sys.stdout`` are routed to stderr;
    Auto3D's own output goes to :func:`document_stream`, which still points at
    the real stdout.

    Re-entrant: a nested activation yields without touching anything, so the
    outermost caller owns the real stdout and an inner one cannot mistake the
    already-redirected stream for it.
    """
    global _real_stdout
    if _real_stdout is not None:
        yield
        return
    _real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        _real_stdout = None


@contextlib.contextmanager
def suppress_foreign_stdout(enabled: bool = True) -> Iterator[None]:
    """Hold third-party stdout back instead of forwarding it to stderr.

    ``--quiet`` promises no chatter, and a library banner on stderr is still
    chatter. What ``--quiet`` does not promise is losing a diagnosis, so
    anything captured here is written to stderr if the command fails -- any
    exception, including the ``SystemExit`` that ``handle_error`` and
    ``_exit_if_incomplete`` raise. On a clean run it is dropped.

    A no-op when ``enabled`` is False, so callers can wrap unconditionally.
    """
    if not enabled:
        yield
        return

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            yield
    except BaseException:
        held = captured.getvalue()
        if held:
            sys.stderr.write(held)
            sys.stderr.flush()
        raise


class _DocumentStream(io.TextIOBase):
    """The file object ``console`` writes through: always the reserved stdout.

    Resolved per write rather than bound once, for the same reason
    :func:`document_stream` resolves per call -- ``console`` below is a
    module-level singleton built at import time, long before
    :func:`reserve_stdout` runs and before ``CliRunner`` installs its own
    buffer.
    """

    def write(self, text: str) -> int:
        return document_stream().write(text)

    def flush(self) -> None:
        stream = document_stream()
        if not getattr(stream, "closed", False):
            stream.flush()

    def isatty(self) -> bool:
        return document_stream().isatty()

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        """Never close the underlying stream; this object does not own it.

        ``io.IOBase.__del__`` closes on garbage collection, which would
        otherwise flush (or close) whatever stream happened to be current at
        interpreter shutdown.
        """


def emit_json(payload: Any) -> None:
    """Write ``payload`` to stdout as a plain, uncolored JSON document.

    Deliberately not ``console.print_json``: rich highlights JSON with ANSI
    escapes whenever stdout is a terminal, so ``auto3d ... --json`` in an
    interactive shell emitted ``ESC[1;34m"success"ESC[0m`` -- fine on screen,
    unparseable the moment it is copied or captured through a pty. A
    machine-readable document gets no styling, no highlighting and no
    width-dependent wrapping.
    """
    stream = document_stream()
    stream.write(json.dumps(payload, indent=2))
    stream.write("\n")
    stream.flush()


def create_console() -> Console:
    """Create a Rich console with auto-detected terminal capabilities.

    Returns:
        Console configured for the current terminal.
    """
    stream = _DocumentStream()

    # Force plain text if piped or redirected
    force_terminal = None
    if not stream.isatty():
        force_terminal = False

    return Console(
        file=stream,
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

    console.print(
        Panel(grid, title=f"[bold]Auto3D v{Auto3D.__version__}[/bold]", border_style="blue")
    )


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
