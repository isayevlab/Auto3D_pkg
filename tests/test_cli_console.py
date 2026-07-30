"""Tests for CLI console utilities."""

import io
import sys

import pytest


def test_console_exists():
    """Console singleton should exist."""
    from Auto3D.cli.console import console
    assert console is not None


def test_console_auto_detects_tty(monkeypatch):
    """Console should auto-detect terminal capabilities."""
    from Auto3D.cli.console import create_console

    # Simulate non-TTY
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    c = create_console()
    assert c._force_terminal is False or c.is_terminal is False


def test_print_success(capsys):
    """print_success should output green checkmark."""
    from Auto3D.cli.console import print_success, console

    # Force no markup for testing
    console._force_terminal = False
    print_success("Test passed")
    # Rich strips markup in non-terminal mode
    captured = capsys.readouterr()
    assert "Test passed" in captured.out


def test_print_error():
    """print_error should output to stderr."""
    from Auto3D.cli.console import print_error
    # Just ensure it doesn't crash
    print_error("Test error", hint="Try this")


def test_print_warning():
    """print_warning should output warning."""
    from Auto3D.cli.console import print_warning
    print_warning("Test warning")
