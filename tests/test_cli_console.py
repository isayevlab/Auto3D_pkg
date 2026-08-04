"""Tests for CLI console utilities."""

import io
import sys

import pytest


def test_console_exists():
    """Console singleton should exist."""
    from Auto3D.cli.console import console
    assert console is not None


def test_console_auto_detects_tty(monkeypatch):
    """Non-interactive stdout must force plain (non-colored) output.

    ``c.is_terminal`` recomputes from the live stream's own ``isatty()``
    whenever ``_force_terminal`` is ``None`` (see ``rich.console.Console
    .is_terminal``), so an ``... or c.is_terminal is False`` assertion would
    still pass even if ``create_console()`` never set ``force_terminal`` at
    all: the monkeypatched ``isatty`` would make that disjunct true on its
    own. Assert on ``_force_terminal`` directly instead.
    """
    from Auto3D.cli.console import create_console

    # Simulate non-TTY
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    c = create_console()
    assert c._force_terminal is False


def test_console_does_not_force_plain_on_a_real_tty(monkeypatch):
    """A real terminal must not be forced to plain output.

    Complements the non-TTY case above: catches a mutation that always sets
    ``force_terminal = False`` (or inverts the ``isatty()`` check) regardless
    of the stream's actual capability.
    """
    from Auto3D.cli.console import create_console

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    c = create_console()
    assert c._force_terminal is None


def test_print_success(capsys):
    """print_success should output green checkmark."""
    from Auto3D.cli.console import print_success, console

    # Force no markup for testing
    console._force_terminal = False
    print_success("Test passed")
    # Rich strips markup in non-terminal mode
    captured = capsys.readouterr()
    assert "Test passed" in captured.out


def test_print_error(capsys):
    """print_error should write the message and hint to stderr, not stdout."""
    from Auto3D.cli.console import error_console, print_error

    error_console._force_terminal = False
    print_error("Test error", hint="Try this")
    captured = capsys.readouterr()
    assert "Test error" in captured.err
    assert "Try this" in captured.err
    assert captured.out == ""


def test_print_warning(capsys):
    """print_warning should write the message to stderr, not stdout."""
    from Auto3D.cli.console import error_console, print_warning

    error_console._force_terminal = False
    print_warning("Test warning")
    captured = capsys.readouterr()
    assert "Test warning" in captured.err
    assert captured.out == ""
