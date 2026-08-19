"""Tests for ``Auto3D.presentation.auto3Dcli`` -- the legacy ``auto3d <config.yaml>`` entry point.

Review Minor #27: the legacy path used to construct ``Auto3DOptions(**parameters)``
directly from a raw ``yaml.safe_load``, so an unrecognized key (a typo like
``optimising_engine``) surfaced as a bare
``TypeError: __init__() got an unexpected keyword argument '...'`` with no exit
code and no hint -- while the modern ``auto3d run INPUT -c config.yaml`` path
already reported the same mistake as a pydantic-backed ``ConfigurationError``
naming the field, at exit code 2.

``_run_legacy_yaml`` (see ``tests/test_legacy_yaml_parity.py`` for the broader
ingestion-parity suite covering the four *malformed-shape* cases) now ingests
through ``Auto3D.presentation.cli.config_schema.load_yaml_config``/``build_cli_config`` --
the exact ``CLIConfig`` (``extra="forbid"``) the modern path already used --
so this specific "unknown key" case is fixed as a side effect. This module
pins that directly: it is a distinct case from the four already covered
(empty file / top-level list / top-level scalar / YAML syntax error), since an
unknown key is a well-formed *mapping* that only pydantic's ``extra="forbid"``
rejects.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_unknown_key_reports_named_configuration_error(tmp_path, monkeypatch):
    """A typo'd key must surface as ``ConfigurationError`` naming the field,
    not a bare ``TypeError``.

    Mutation check (performed by hand, not re-run here since it would require
    editing production code): reverting ``_run_legacy_yaml`` to construct
    ``Auto3DOptions(**parameters)`` directly from the raw YAML dict -- the
    pre-fix shape -- makes ``Auto3DOptions(path=..., k=1,
    optimising_engine=...)`` raise a bare ``TypeError`` instead (verified
    interactively: ``Auto3DOptions.__init__() got an unexpected keyword
    argument 'optimising_engine'``), which this test would not accept as
    passing.
    """
    from Auto3D.foundation.exceptions import ConfigurationError
    from Auto3D.presentation.auto3Dcli import _run_legacy_yaml
    from Auto3D.presentation.cli import errors as errors_mod

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("path: mols.smi\nk: 1\noptimising_engine: AIMNET\n")

    seen: list[BaseException] = []
    real_handle_error = errors_mod.handle_error

    def spy(error, *args, **kwargs):
        seen.append(error)
        return real_handle_error(error, *args, **kwargs)

    monkeypatch.setattr(errors_mod, "handle_error", spy)

    with pytest.raises(SystemExit) as exc_info:
        _run_legacy_yaml(str(cfg))

    assert seen, "expected the failure to route through handle_error"
    error = seen[-1]
    assert isinstance(error, ConfigurationError), (
        f"expected ConfigurationError, got {type(error).__name__}: {error}"
    )
    assert "optimising_engine" in str(error), str(error)
    assert exc_info.value.code == 2


def test_unknown_key_reports_same_verdict_as_the_modern_path(tmp_path, monkeypatch):
    """Same shape, same verdict, through ``auto3d run INPUT -c cfg.yaml``.

    Complements ``test_legacy_yaml_parity.py``'s malformed-shape parity suite
    with the "well-formed mapping, unknown key" shape it does not cover.
    """
    from Auto3D.presentation.cli.commands import run as run_mod

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("k: 1\noptimising_engine: AIMNET\n")

    seen: list[BaseException] = []
    real_handle_error = run_mod.handle_error

    def spy(error, *args, **kwargs):
        seen.append(error)
        return real_handle_error(error, *args, **kwargs)

    monkeypatch.setattr(run_mod, "handle_error", spy)

    with pytest.raises(SystemExit) as exc_info:
        run_mod.execute_run(input_file=smi, config_file=cfg, quiet=True)

    assert seen, "expected the failure to route through handle_error"
    from Auto3D.foundation.exceptions import ConfigurationError

    assert isinstance(seen[-1], ConfigurationError), type(seen[-1])
    assert exc_info.value.code == 2
