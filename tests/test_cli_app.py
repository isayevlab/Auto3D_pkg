# tests/test_cli_app.py
"""Tests for main Typer application."""

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_app_exists():
    """Main app should exist."""
    from Auto3D.cli.app import app
    assert app is not None


def test_help_works(runner):
    """--help should show available commands."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "config" in result.stdout
    assert "models" in result.stdout


def test_version_works(runner):
    """--version should show version."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0


def test_run_help(runner):
    """run --help should show run options."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout or "-c" in result.stdout


def test_config_help(runner):
    """config --help should show subcommands."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout


def test_models_help(runner):
    """models --help should show subcommands."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout
