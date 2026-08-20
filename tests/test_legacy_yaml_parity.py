"""One YAML ingestion path: the two entry points must agree on malformed files.

Auto3D has two ways to hand it a YAML configuration:

* the deprecated ``auto3d <config.yaml>`` form (``auto3Dcli._run_legacy_yaml``);
* the modern ``auto3d run INPUT -c <config.yaml>`` form
  (``cli.commands.run.execute_run`` -> ``cli.config_schema.load_yaml_config``).

Both already shared every *value* validator -- ``build_cli_config``, and through
it ``FIELD_BOUNDS``, ``extra="forbid"``, ``parse_gpu_idx`` and the engine
registry lookup. What they did **not** share was the *ingestion* layer: the
three shape guards in ``load_yaml_config`` (empty file, non-mapping top level,
unparseable YAML). ``_run_legacy_yaml`` carried its own ``yaml.safe_load`` and
its own ``"None"``-string loop, so an empty or list-topped file reached
``parameters.items()`` and surfaced as ``AttributeError``/``TypeError`` under
the generic "Unexpected Error" panel at **exit 1**, while the identical file
through ``-c`` gave a ``ConfigurationError`` at **exit 2** with a hint.

Two exit codes for one file is exactly what ``build_cli_config``'s docstring
says it exists to prevent, so this module asserts the property directly rather
than asserting either path's behavior in isolation: for each malformed shape,
both entry points must report the **same exception class** and the **same exit
code**.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Each shape is a file the user could plausibly write by mistake. The values are
# the literal file contents; see test_config_parity.py for the well-formed
# counterpart of this comparison (same configuration, every entry point).
MALFORMED_SHAPES: dict[str, str] = {
    # yaml.safe_load returns None for an empty document.
    "empty_file": "",
    # A top-level sequence: valid YAML, but not a mapping of option names.
    "top_level_list": "- k: 1\n- path: mols.smi\n",
    # A top-level scalar -- the shape you get from a file holding only a comment
    # line's worth of text, and the other half of the "not a mapping" case.
    "top_level_scalar": "k=1\n",
    # Genuinely unparseable: yaml.safe_load raises yaml.YAMLError.
    "yaml_syntax_error": "k: 1\n  window: 5.0\n",
}


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _spy_on(monkeypatch, module, sink: list) -> None:
    """Record every exception ``module.handle_error`` is handed, then let the
    real handler run so the exit code it chooses is the one under test.

    Patched per-module rather than on ``Auto3D.presentation.cli.errors`` alone because the
    two entry points bind the name differently: ``cli/commands/run.py`` imports
    ``handle_error`` at module level (so the binding must be replaced on *that*
    module), while ``_run_legacy_yaml`` imports it inside the function body on
    every call.
    """
    real = module.handle_error

    def spy(error, *args, **kwargs):
        sink.append(error)
        return real(error, *args, **kwargs)

    monkeypatch.setattr(module, "handle_error", spy)


def _verdict_legacy(yaml_path: Path, monkeypatch) -> tuple[str, int]:
    """(exception class name, exit code) from ``auto3d <config.yaml>``."""
    from Auto3D.presentation.auto3Dcli import _run_legacy_yaml
    from Auto3D.presentation.cli import errors as errors_mod

    seen: list[BaseException] = []
    _spy_on(monkeypatch, errors_mod, seen)

    with pytest.raises(SystemExit) as exc_info:
        _run_legacy_yaml(str(yaml_path))

    assert seen, "the legacy path exited without routing through handle_error"
    return type(seen[-1]).__name__, exc_info.value.code


def _verdict_modern(input_file: Path, yaml_path: Path, monkeypatch) -> tuple[str, int]:
    """(exception class name, exit code) from ``auto3d run INPUT -c cfg``."""
    from Auto3D.presentation.cli.commands import run as run_mod

    seen: list[BaseException] = []
    _spy_on(monkeypatch, run_mod, seen)

    with pytest.raises(SystemExit) as exc_info:
        run_mod.execute_run(input_file=input_file, config_file=yaml_path, quiet=True)

    assert seen, "the modern path exited without routing through handle_error"
    return type(seen[-1]).__name__, exc_info.value.code


@pytest.mark.parametrize("shape", sorted(MALFORMED_SHAPES))
def test_malformed_yaml_is_judged_identically_by_both_entry_points(shape, tmp_path, monkeypatch):
    """A malformed config file must produce the same error class and the same
    exit code whichever entry point reads it.

    Before the fix all four shapes gave ``ConfigurationError``/exit 2 through
    ``-c`` and an internal-looking ``AttributeError``/``TypeError``/
    ``yaml.YAMLError`` at exit 1 through ``auto3d <config.yaml>``.
    """
    cfg = _write(tmp_path, "cfg.yaml", MALFORMED_SHAPES[shape])
    smi = _write(tmp_path, "mols.smi", "CCO m1\n")

    legacy = _verdict_legacy(cfg, monkeypatch)
    modern = _verdict_modern(smi, cfg, monkeypatch)

    assert legacy == modern, (
        f"{shape}: legacy 'auto3d cfg.yaml' reported {legacy} but modern "
        f"'auto3d run in.smi -c cfg.yaml' reported {modern}"
    )
    # Pin the shared verdict too, not just the agreement: a future change that
    # made *both* paths crash with "Unexpected Error" at exit 1 would satisfy
    # the equality above while destroying the property it exists to protect.
    assert modern == ("ConfigurationError", 2), modern


def test_settings_only_config_still_refused_by_the_legacy_form(tmp_path, monkeypatch):
    """Behavior that must be *preserved*, not changed, by the ingestion merge.

    ``CLIConfig.path`` is optional so a settings-only config file is reusable
    across runs (``auto3d run INPUT -c cfg.yaml`` supplies the input). The
    deprecated form has no other source of an input path, so
    ``to_auto3d_options()`` must still refuse it -- as a ``ConfigurationError``
    at exit 2, naming the missing key.
    """
    # ANI2xt, not the default AIMNET: resolving the engine name for a *valid*
    # config actually happens here, and AIMNET would import the optional
    # `aimnet` package (see tests/test_config_parity.py's matching note).
    cfg = _write(tmp_path, "settings_only.yaml", "k: 1\noptimizing_engine: 'ANI2xt'\n")

    assert _verdict_legacy(cfg, monkeypatch) == ("ConfigurationError", 2)
