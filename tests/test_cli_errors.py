# tests/test_cli_errors.py
"""Tests for M30: an internal (non-Auto3DError) CLI failure must be
debuggable without editing source.

Before this fix, `handle_error` took no verbosity argument and printed only
`str(error)` at every verbosity -- an unexpected `KeyError('ID')` (e.g. from a
missing SDF property) rendered as a bare red box reading `'ID'`, with no file,
line, or stack, no matter how many `-v` a user passed (there was nowhere for
them to go).

Two layers of coverage:
- Direct calls to `handle_error` pin its formatting contract (message, hint,
  traceback presence/absence) cheaply, without a full CLI invocation.
- `CliRunner` invocations pin that `-v/--verbose` actually reaches
  `handle_error` through each command's real wiring -- a `handle_error` that
  works correctly in isolation but is never passed the CLI's verbosity would
  still pass the direct tests above and silently ship broken.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from Auto3D.cli.app import app
from Auto3D.cli.errors import handle_error
from Auto3D.exceptions import ConfigurationError

runner = CliRunner()


# --- direct handle_error tests: formatting contract --------------------------

def test_auto3derror_verbose0_shows_message_and_hint_no_traceback(capsys):
    """The existing clean presentation for a known Auto3DError is the
    feature, and must not change at verbose=0."""
    with pytest.raises(SystemExit) as exc_info:
        handle_error(ConfigurationError("bad config"), verbose=0)
    assert exc_info.value.code == 2  # ConfigurationError -> 2 (unchanged mapping)

    captured = capsys.readouterr()
    assert "bad config" in captured.err
    assert "config init" in captured.err  # get_error_hint's hint text
    assert "Traceback" not in captured.err


def test_auto3derror_verbose1_shows_traceback(capsys):
    """The same known error, at verbose=1, must additionally show a
    traceback -- the hint/message presentation is unchanged, just extended."""
    try:
        raise ConfigurationError("bad config")
    except ConfigurationError as e:
        with pytest.raises(SystemExit) as exc_info:
            handle_error(e, verbose=1)
    assert exc_info.value.code == 2  # mapping still unchanged at verbose=1

    captured = capsys.readouterr()
    assert "bad config" in captured.err
    assert "config init" in captured.err
    assert "Traceback" in captured.err
    assert "ConfigurationError" in captured.err


def test_unexpected_error_verbose0_identifies_type_and_points_to_verbose(capsys):
    """Judgment call: even at verbose=0, an unexpected (non-Auto3DError)
    failure must name the exception type and say how to get more, because
    the bare message alone ('ID') gives the user nothing to act on."""
    with pytest.raises(SystemExit) as exc_info:
        handle_error(KeyError("ID"), verbose=0)
    assert exc_info.value.code == 1  # unmapped -> generic exit code, unchanged

    captured = capsys.readouterr()
    assert "KeyError" in captured.err
    assert "'ID'" in captured.err
    assert "Traceback" not in captured.err
    assert "-v" in captured.err or "--verbose" in captured.err


def test_unexpected_error_verbose1_traceback_identifies_origin(capsys):
    """Pinned scenario from the brief: an unexpected KeyError('ID') must, at
    verbose=1, produce something that identifies where it came from -- the
    failing function name and the file it lives in, not just the message."""
    def _raise_keyerror_id():
        d: dict = {}
        return d["ID"]

    try:
        _raise_keyerror_id()
    except KeyError as e:
        with pytest.raises(SystemExit) as exc_info:
            handle_error(e, verbose=1)
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Traceback" in captured.err
    assert "KeyError" in captured.err
    assert "_raise_keyerror_id" in captured.err  # names the failing frame
    assert "test_cli_errors" in captured.err  # names the file it's in


def test_handle_error_default_verbose_is_zero(capsys):
    """Existing call sites (and the pre-existing direct test in
    test_cli_app.py) call handle_error with no verbose argument at all; the
    default must keep today's no-traceback behavior."""
    with pytest.raises(SystemExit):
        handle_error(KeyError("ID"))
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


# --- CliRunner: verbosity threads through each command's real wiring --------

def _raise_keyerror_id(*args, **kwargs):
    raise KeyError("ID")


def test_run_verbose_flag_reaches_handle_error(tmp_path, monkeypatch):
    """`auto3d run` funnels an unexpected internal error through handle_error
    only if `-v` actually reaches it via execute_run's own `verbose` param."""
    import Auto3D.auto3D as a3d

    smi = tmp_path / "in.smi"
    smi.write_text("CCO mol1\n")
    monkeypatch.setattr(a3d, "main", _raise_keyerror_id)

    quiet = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "--quiet"])
    assert quiet.exit_code == 1
    assert "Traceback" not in quiet.stdout
    assert "Traceback" not in quiet.stderr
    assert "KeyError" in quiet.stderr

    verbose = runner.invoke(
        app, ["run", str(smi), "--k", "1", "--no-gpu", "--quiet", "-v"]
    )
    assert verbose.exit_code == 1
    # Judgment call: error_console is stderr-bound, so the traceback must
    # never land on stdout -- Task 4 made --json stdout load-bearing for a
    # partial-run consumer, and a stray traceback there would corrupt it.
    assert "Traceback" not in verbose.stdout
    assert "Traceback" in verbose.stderr
    assert "_raise_keyerror_id" in verbose.stderr


def test_run_json_stdout_stays_clean_on_unexpected_error_with_verbose(
    tmp_path, monkeypatch
):
    """Same pin as above, specifically with --json: the traceback (stderr)
    must never contaminate the --json stdout stream."""
    import Auto3D.auto3D as a3d

    smi = tmp_path / "in.smi"
    smi.write_text("CCO mol1\n")
    monkeypatch.setattr(a3d, "main", _raise_keyerror_id)

    res = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "--json", "-v"])
    assert res.exit_code == 1
    assert "Traceback" not in res.stdout
    assert "Traceback" in res.stderr


def test_energy_verbose_flag_reaches_handle_error(tmp_path):
    """A properties.py command: these had no verbosity option at all before
    this change, so this pins the newly-added wiring specifically."""
    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")

    with patch("Auto3D.SPE.calc_spe", side_effect=KeyError("ID")):
        quiet = runner.invoke(app, ["energy", str(sdf), "--no-gpu"])
    assert quiet.exit_code == 1
    assert "Traceback" not in quiet.stderr

    with patch("Auto3D.SPE.calc_spe", side_effect=KeyError("ID")):
        verbose = runner.invoke(app, ["energy", str(sdf), "--no-gpu", "-v"])
    assert verbose.exit_code == 1
    assert "Traceback" not in verbose.stdout
    assert "Traceback" in verbose.stderr


def test_models_test_verbose_flag_reaches_handle_error(monkeypatch):
    """`models test` also gained -v in this change; a DependencyError is a
    known Auto3DError, so verbose=1 must add a traceback on top of the
    existing clean panel (not replace it)."""
    from Auto3D.exceptions import DependencyError

    def _boom(*a, **k):
        raise DependencyError("torchani not installed")

    monkeypatch.setattr(
        "Auto3D.model_factory.get_device",
        lambda *a, **k: __import__("torch").device("cpu"),
    )
    monkeypatch.setattr("Auto3D.model_factory.create_model", _boom)

    quiet = runner.invoke(app, ["models", "test", "ANI2x", "--no-gpu"])
    assert quiet.exit_code == 3  # DependencyError -> 3, unchanged mapping
    assert "Traceback" not in quiet.stderr
    assert "torchani not installed" in quiet.stderr

    verbose = runner.invoke(app, ["models", "test", "ANI2x", "--no-gpu", "-v"])
    assert verbose.exit_code == 3
    assert "torchani not installed" in verbose.stderr
    assert "Traceback" not in verbose.stdout
    assert "Traceback" in verbose.stderr
