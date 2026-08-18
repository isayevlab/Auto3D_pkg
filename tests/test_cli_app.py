# tests/test_cli_app.py
"""Tests for main Typer application."""

import re

import pytest
from typer.testing import CliRunner

import Auto3D.auto3D


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
    """--version should show version.

    ``exit_code == 0`` alone comes from ``raise typer.Exit()`` in
    ``version_callback`` independently of whatever it printed (or didn't) --
    deleting the ``console.print`` call entirely would still exit 0. Check
    the actual printed content.
    """
    import Auto3D
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Auto3D version" in result.stdout
    assert Auto3D.__version__ in result.stdout


def test_run_help(runner):
    """run --help should show run options.

    ``"--config" in out or "-c" in out`` is trivially satisfied by
    ``--max-confs`` (which contains the substring ``-c``) even if ``-c``/
    ``--config`` were deleted outright. Anchor on the full ``--config`` flag
    name, which is not a substring of any other option.
    """
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert "--config" in out


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


def test_models_list_shows_engines(runner):
    """models list should show available engines."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "AIMNET" in result.stdout


def test_models_info_aimnet(runner):
    """models info should show engine details."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "AIMNET"])
    assert result.exit_code == 0
    assert "AIMNet2" in result.stdout


def test_models_info_ani2x(runner):
    """models info should show ANI2x details."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "ANI2x"])
    assert result.exit_code == 0
    assert "ANI-2x" in result.stdout


def test_models_info_unknown_engine(runner):
    """models info should fail for unknown engine.

    Exit 2 (ConfigurationError), not the hard-coded 1 this used to raise: an
    unrecognized engine name is the same user mistake `resolve_engine_name`
    already reports as exit 2 from `run` and `energy`.
    """
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "UNKNOWN"])
    assert result.exit_code == 2
    assert "Unknown engine" in " ".join(result.stderr.split())


def test_models_list_shows_aimnet_registry(runner):
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    out = result.stdout
    assert "AIMNET" in out
    assert "aimnet2-2025" in out  # registry families surfaced
    assert "ANI2x" in out


def test_models_info_aimnet_element_set(runner):
    """A bare "B" substring is satisfied by "Best for organic molecules" --
    and every other single-letter symbol risks the same kind of accidental
    match somewhere in the panel's prose. Parse the actual "Supported
    Elements:" line into discrete tokens and check membership there.
    """
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "AIMNET"])
    assert result.exit_code == 0

    match = re.search(r"Supported Elements:\s*([A-Za-z, ]+)", result.stdout)
    assert match, f"no 'Supported Elements:' line found in:\n{result.stdout}"
    elements = {tok.strip() for tok in match.group(1).split(",")}

    expected = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"}
    assert elements == expected, f"element set changed: {elements}"


def test_validate_valid_smi(runner, tmp_path):
    """validate should pass for valid SMILES file."""
    from Auto3D.cli.app import app

    smi_file = tmp_path / "test.smi"
    smi_file.write_text("CCO ethanol\nCC(=O)O acetic_acid\n")

    result = runner.invoke(app, ["validate", str(smi_file)])
    assert result.exit_code == 0
    assert "Valid" in result.stdout or "valid" in result.stdout.lower()


def test_validate_invalid_smi(runner, tmp_path):
    """validate should fail for invalid SMILES.

    Exit 2 (InputValidationError), not the hard-coded 1 this used to raise:
    the pre-flight checker must return the code `auto3d run` returns for the
    same file, which is 2.
    """
    from Auto3D.cli.app import app

    smi_file = tmp_path / "test.smi"
    smi_file.write_text("INVALID_SMILES mol1\n")

    result = runner.invoke(app, ["validate", str(smi_file)])
    assert result.exit_code == 2
    assert "Validation Failed" in " ".join(result.stdout.split())


@pytest.fixture
def tmp_path_cwd(tmp_path, monkeypatch):
    """Change to tmp_path and return it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------
# Out-of-process harness for the stdout contract.
#
# `--json` promises a stdout stream carrying nothing but the document, and
# what broke that promise was a third-party import-time `print`. Asserting it
# in-process cannot work: `CliRunner` swaps `sys.stdout` for a buffer, and any
# earlier test in the same interpreter that already imported `aimnet` makes
# the offending banner disappear -- an order-dependent guard that passes for
# the wrong reason. So the tests below drive a real process and read its real
# stdout and stderr.
#
# The pipeline itself is stubbed out (`Auto3D.auto3D` is replaced in
# `sys.modules` before the CLI imports it): a real run would download and load
# a neural network potential, which these tests have no business doing. The
# stub prints FOREIGN_STDOUT_MARKER, standing in for the writes a real run's
# libraries make to stdout after the command has started -- notably the same
# warp banner, re-printed by every *spawned* optimizer worker.
# --------------------------------------------------------------------------

FOREIGN_STDOUT_MARKER = "FOREIGN-STDOUT-WRITE"

_STUBBED_RUN_BOOTSTRAP = '''\
"""Invoke the `auto3d` console-script entry point with the pipeline stubbed.

argv: <output.sdf> <marker> <auto3d args...>
"""
import sys
import types

out_sdf, marker, cli_args = sys.argv[1], sys.argv[2], sys.argv[3:]

from Auto3D.results import WorkflowResult


def fake_main(options, **kwargs):
    print(marker)
    return WorkflowResult(out_sdf)


stub = types.ModuleType("Auto3D.auto3D")
stub.main = fake_main
sys.modules["Auto3D.auto3D"] = stub

sys.argv = ["auto3d", *cli_args]
from Auto3D.auto3Dcli import cli

cli()
'''


def _warp_is_installed() -> bool:
    """Whether the library whose banner motivated all of this is present."""
    import importlib.util

    return importlib.util.find_spec("warp") is not None


def _write_single_conformer_sdf(path) -> None:
    """Write a one-molecule SDF for the stubbed run to report on."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol.SetProp("_Name", "mol1")
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


@pytest.fixture
def auto3d_process(tmp_path):
    """Return a callable running `auto3d <args>` in a fresh subprocess.

    The input file is appended after the subcommand automatically, so a caller
    writes ``auto3d_process("run", "--k", "1", "--json")``.
    """
    import subprocess
    import sys

    bootstrap = tmp_path / "stubbed_auto3d.py"
    bootstrap.write_text(_STUBBED_RUN_BOOTSTRAP)
    smi = tmp_path / "in.smi"
    smi.write_text("CCO mol1\n")
    out_sdf = tmp_path / "in_out.sdf"
    _write_single_conformer_sdf(out_sdf)

    def run(subcommand: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(bootstrap),
                str(out_sdf),
                FOREIGN_STDOUT_MARKER,
                subcommand,
                str(smi),
                *args,
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=300,
            check=False,
        )

    return run


def test_config_init_creates_file(runner, tmp_path_cwd):
    """config init should create YAML file."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-o", "test.yaml"])

    assert result.exit_code == 0
    assert (tmp_path_cwd / "test.yaml").exists()


def test_config_init_with_preset(runner, tmp_path_cwd):
    """config init with preset should use preset values."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-o", "test.yaml", "-p", "quick"])

    assert result.exit_code == 0
    content = (tmp_path_cwd / "test.yaml").read_text()
    assert "opt_steps: 500" in content


def test_config_init_default_path(runner, tmp_path_cwd):
    """config init without -o should create auto3d.yaml."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert (tmp_path_cwd / "auto3d.yaml").exists()


def test_config_init_invalid_preset(runner, tmp_path_cwd):
    """config init with an invalid preset is rejected by Typer's own enum
    validation, not merely routed to execute_config_init's fallback guard.

    execute_config_init's own preset check (cli/commands/config.py) is
    "unreachable from the CLI" by its own docstring, but it ALSO exits 2 and
    ALSO mentions "quick" in its hint -- so if the CLI's ``Preset`` enum
    annotation were ever weakened to a plain ``str``, this test would still
    pass via that fallback path, silently losing the enum-level guard it
    claims to test (the two-paths-same-integer trap). Click's own
    choice-validation wording ("is not one of") is specific to the enum
    rejection and cannot come from the fallback's "Unknown preset" message.
    """
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-p", "invalid"])

    assert result.exit_code == 2
    assert "is not one of" in result.output
    # Click reports the valid choices for the enum option.
    assert "quick" in result.output


def test_config_show_displays_file(runner, tmp_path_cwd):
    """config show should display YAML file."""
    from Auto3D.cli.app import app

    # Create a config file first
    config_file = tmp_path_cwd / "test.yaml"
    config_file.write_text("path: input.smi\nk: 5\n")

    result = runner.invoke(app, ["config", "show", str(config_file)])

    assert result.exit_code == 0
    assert "path" in result.stdout


def test_config_show_not_found(runner, tmp_path_cwd):
    """config show should fail if file not found.

    Exit 2 (ConfigurationError), not the hard-coded 1 this used to raise --
    the same code `config validate` gives for a missing file (there via
    Typer's own `exists=True` check).
    """
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "show", "nonexistent.yaml"])

    assert result.exit_code == 2
    assert "not found" in result.stderr


def test_config_validate_valid(runner, tmp_path_cwd):
    """config validate should pass for valid config."""
    from Auto3D.cli.app import app

    # Create a valid config file
    config_file = tmp_path_cwd / "valid.yaml"
    config_file.write_text("path: input.smi\nk: 5\noptimizing_engine: AIMNET\nuse_gpu: true\n")

    result = runner.invoke(app, ["config", "validate", str(config_file)])

    assert result.exit_code == 0
    assert "Valid" in result.stdout or "Validation Passed" in result.stdout


def test_config_validate_invalid(runner, tmp_path_cwd):
    """config validate should fail for invalid config.

    Exit 2 (ConfigurationError), not the hard-coded 1 this used to raise:
    `auto3d run -c` rejects this same file with 2, and a pre-flight checker
    that answers a different number than the run it predicts is useless as a
    script gate.

    The config here is invalid on its *values* (`k: 0` violates FIELD_BOUNDS).
    It used to be a file whose only defect was a missing `path`, which is no
    longer a defect at all -- see
    test_config_validate_accepts_a_settings_only_config below.
    """
    from Auto3D.cli.app import app

    config_file = tmp_path_cwd / "invalid.yaml"
    config_file.write_text("path: mols.smi\nk: 0\noptimizing_engine: AIMNET\n")

    result = runner.invoke(app, ["config", "validate", str(config_file)])

    assert result.exit_code == 2
    assert "Validation Passed" not in result.output


def test_config_validate_accepts_a_settings_only_config(runner, tmp_path_cwd):
    """A config with no `path` is valid, because `auto3d run INPUT -c` is.

    Every modern entry point supplies the input on the command line and
    overrides whatever `path` the file carries, so the reusable settings-only
    config is the shape users actually want. `config validate` used to reject
    it with `path / Field required` -- a pre-flight checker calling invalid
    the one file shape that runs fine.

    The paired `run` invocation is what makes this a parity assertion rather
    than a claim about the validator alone: both must agree.
    """
    from unittest.mock import patch

    from Auto3D.cli.app import app

    cfg = tmp_path_cwd / "settings_only.yaml"
    cfg.write_text("k: 5\noptimizing_engine: AIMNET\nuse_gpu: false\n")

    result = runner.invoke(app, ["config", "validate", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Validation Passed" in result.output

    # And the run it predicts really does accept it.
    smi = tmp_path_cwd / "mols.smi"
    smi.write_text("CCO m1\n")
    with patch.object(Auto3D.auto3D, "main", return_value="out.sdf") as m:
        run_result = runner.invoke(app, ["run", str(smi), "-c", str(cfg), "--no-gpu"])
    assert run_result.exit_code == 0, run_result.output
    assert m.called, "run rejected the config that config validate approved"


def test_run_requires_input(runner):
    """run should require input file argument."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_run_with_nonexistent_file(runner):
    """run should fail with nonexistent file."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run", "nonexistent.smi"])
    assert result.exit_code != 0


def test_json_output_is_pure_json(auto3d_process):
    """`auto3d run --json` must write the JSON document and nothing else to stdout.

    Run as a real subprocess, not through ``CliRunner``, because the bytes on
    stdout *are* the contract and the write that broke it was not ours: the
    engine-name check imports ``aimnet`` -> ``warp``, which prints a 734-byte
    device banner to stdout at import time. The version of this test that this
    one replaces asserted the same guarantee in-process and did not provide
    it -- it failed when run alone and passed in the full suite only because
    some earlier test had already paid for that import, so the banner was
    spent by the time it ran. A broken `--json` therefore shipped green.

    The assertion is on the exact bytes rather than "json.loads succeeded":
    ``json.loads`` accepts trailing whitespace and would also accept a
    document that some future change decided to colorize with ANSI on a
    terminal, so re-serializing and comparing is what actually pins "the
    document, the whole document, and nothing but the document".
    """
    import json

    result = auto3d_process("run", "--k", "1", "--json")

    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert result.stdout == json.dumps(document, indent=2) + "\n", (
        f"stdout carried something other than the JSON document:\n{result.stdout!r}"
    )
    assert document["success"] is True
    assert document["molecules"] == 1

    # Contained, not silenced: both third-party writes are still readable,
    # they are just on the stream diagnostics belong on. A fix that dropped
    # them would also drop a library's genuine failure message.
    assert FOREIGN_STDOUT_MARKER in result.stderr
    if _warp_is_installed():
        assert "Warp" in result.stderr


def test_json_output_is_written_before_nonzero_exit_when_molecules_missing(
    runner, tmp_path_cwd, monkeypatch
):
    """C6/B8: a run that loses a molecule must still emit parseable JSON, then
    exit non-zero.

    Hermetic: `Auto3D.auto3D.main` is monkeypatched to return a
    `WorkflowResult` carrying a non-empty `failures` list (Task 3's
    reconciliation carrier), so this exercises the full `execute_run` ->
    `output_json` -> `_exit_if_incomplete` path without a pipeline run or a
    loaded potential.
    """
    import json

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\nCCCO mol2\n")

    import Auto3D.auto3D as a3d

    out = tmp_path_cwd / "in_out.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with Chem.SDWriter(str(out)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "mol1")
        w.write(m)
    from Auto3D.results import WorkflowResult

    # mol2 has no corresponding output structure -- a lost molecule.
    monkeypatch.setattr(a3d, "main", lambda options: WorkflowResult(str(out), failures=["mol2"]))

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--json"])

    assert result.exit_code != 0, f"exited 0 despite a reported failure; output:\n{result.output}"
    # No slicing from the first '{' any more: the CLI now reserves stdout for
    # its own output for the whole command, so the third-party banner that
    # used to need routing around lands on stderr instead. A workaround left
    # next to the fix would tell the next reader the stream is still dirty.
    stdout = result.stdout
    data = json.loads(stdout)  # must not raise
    assert stdout == json.dumps(data, indent=2) + "\n"
    assert data["success"] is False
    assert data["failed"] == 1
    assert data["molecules"] == 1
    assert [f["name"] for f in data["failures"]] == ["mol2"]


def test_no_nonzero_exit_when_no_molecules_missing(runner, tmp_path_cwd, monkeypatch):
    """A complete run (no reconciled failures) must keep exiting 0."""
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    import Auto3D.auto3D as a3d

    out = tmp_path_cwd / "in_out.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with Chem.SDWriter(str(out)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "mol1")
        w.write(m)
    from Auto3D.results import WorkflowResult

    monkeypatch.setattr(a3d, "main", lambda options: WorkflowResult(str(out)))

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--json"])
    assert result.exit_code == 0


def test_quiet_suppresses_third_party_stdout(auto3d_process):
    """`--quiet` must silence output Auto3D does not write, too.

    Before this, `auto3d run in.smi --k 1 -q` printed the 14-line warp device
    banner it had promised to suppress -- `quiet` only ever gated Auto3D's own
    `console.print` calls, and the banner is not one of them.
    """
    result = auto3d_process("run", "--k", "1", "--quiet")

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    # Not merely moved to the other stream -- that is what happens *without*
    # `--quiet`, and it is what makes `stdout == ""` on its own a vacuous
    # check. stderr is not asserted empty, though: `--quiet` suppresses
    # chatter, not diagnostics, and a library reporting a real problem there
    # is exactly what must keep working (with no CUDA device visible, warp
    # writes "no CUDA-capable device is detected" to stderr on this path).
    assert FOREIGN_STDOUT_MARKER not in result.stdout
    assert FOREIGN_STDOUT_MARKER not in result.stderr


def test_quiet_releases_held_third_party_output_when_the_run_fails(
    runner, tmp_path_cwd, monkeypatch
):
    """A library's stdout write is held back by `--quiet`, not thrown away.

    Suppressing a banner must not also swallow the one line that explained a
    crash, so anything held is written to stderr if the command fails.

    The assertion is on *ordering*, not on "the text reached stderr", because
    those are not the same claim: simply forwarding third-party stdout to
    stderr live -- which is what happens without `--quiet` -- also puts the
    text on stderr and would satisfy the weaker check while providing none of
    the behavior. Held output can only be released once the failure is known,
    so it necessarily lands *after* the error panel; forwarded output
    necessarily lands before it.
    """
    from Auto3D.cli.app import app
    from Auto3D.exceptions import ModelLoadError

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    import Auto3D.auto3D as a3d

    def exploding_main(options, **kwargs):
        print("libfoo: could not reach the model server")
        raise ModelLoadError("model could not be obtained")

    monkeypatch.setattr(a3d, "main", exploding_main)

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--quiet"])

    assert result.exit_code == 5
    assert result.stdout == ""
    released_at = result.stderr.index("libfoo: could not reach the model server")
    panel_at = result.stderr.index("model could not be obtained")
    assert released_at > panel_at, (
        "the library's message was forwarded live rather than held and "
        f"released on failure:\n{result.stderr}"
    )


def test_json_error_document_is_emitted_when_the_command_fails(runner, tmp_path_cwd):
    """`--json` must leave a parseable document on stdout on the failure path too.

    A caller that parses stdout got an empty stream on every error, so it
    could not tell "the command failed" from "the command produced nothing to
    say". The Rich panel still goes to stderr -- this adds a stdout document,
    it does not move the diagnostics.
    """
    import json

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    result = runner.invoke(
        app, ["run", str(smi), "--k", "1", "--json", "--engine", "aimnet2-2025x"]
    )

    assert result.exit_code == 2
    document = json.loads(result.stdout)
    assert result.stdout == json.dumps(document, indent=2) + "\n"
    assert document["success"] is False
    assert document["error_type"] == "ConfigurationError"
    assert document["exit_code"] == 2
    assert "aimnet2-2025x" in document["error"]
    # The human-readable panel is unchanged and still on stderr.
    assert "Configuration Error" in result.stderr


def test_validate_json_reports_a_clean_file(runner, tmp_path_cwd):
    """`auto3d validate --json`: every sibling command had --json, this one did not."""
    import json

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\nCCC mol2\n")

    result = runner.invoke(app, ["validate", str(smi), "--json"])

    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert result.stdout == json.dumps(document, indent=2) + "\n"
    assert document == {
        "success": True,
        "command": "validate",
        "input_file": str(smi),
        "format": "SMI",
        "molecules": 2,
        "valid_molecules": 2,
        "errors": [],
    }


def test_validate_json_reports_every_bad_entry_and_exits_nonzero(runner, tmp_path_cwd):
    """The JSON document lists all failures, not the ten the human table shows."""
    import json

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("".join(f"not_a_smiles_{i} mol{i}\n" for i in range(12)))

    result = runner.invoke(app, ["validate", str(smi), "--json"])

    # Exit 2 (InputValidationError), matching every other input rejection in
    # the CLI. The document on stdout is still validate's own, richer one --
    # `handle_error`'s failure document is deliberately suppressed here,
    # because two JSON documents on one stream is not parseable JSON.
    assert result.exit_code == 2
    assert result.stdout == json.dumps(json.loads(result.stdout), indent=2) + "\n"
    document = json.loads(result.stdout)
    assert document["success"] is False
    assert document["molecules"] == 12
    assert document["valid_molecules"] == 0
    assert len(document["errors"]) == 12


def test_json_document_is_not_colorized_on_a_terminal(tmp_path):
    """Rich highlights JSON with ANSI whenever stdout is a tty.

    That made `auto3d ... --json` in an interactive shell emit
    ``ESC[1;34m"success"ESC[0m``: fine to look at, unparseable the moment it
    is copied or captured. A pty is the only way to see it -- under pytest,
    and under any pipe, Rich renders plain and the bug is invisible, which is
    how it survived.
    """
    import json
    import os
    import pty
    import subprocess
    import sys

    smi = tmp_path / "in.smi"
    smi.write_text("CCO mol1\n")

    master, slave = pty.openpty()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['auto3d', *sys.argv[1:]];"
            " from Auto3D.auto3Dcli import cli; cli()",
            "validate",
            str(smi),
            "--json",
        ],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=subprocess.DEVNULL,
    )
    os.close(slave)
    chunks = []
    try:
        while True:
            try:
                data = os.read(master, 4096)
            except OSError:  # EIO: the child closed its end of the pty
                break
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(master)
    assert process.wait(timeout=300) == 0

    on_terminal = b"".join(chunks)
    assert b"\x1b" not in on_terminal, f"ANSI escapes in a --json document: {on_terminal!r}"
    # A pty turns "\n" into "\r\n"; the document itself must still parse.
    assert json.loads(on_terminal.decode().replace("\r\n", "\n"))["success"] is True


# Rich styles help output when it thinks the stream supports colour, which CI
# forces on via FORCE_COLOR. Tests that match on the text must ignore styling.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def test_help_goes_to_stdout_and_carries_no_import_banner(auto3d_process):
    """Two things at once, and both matter.

    Help is legitimate stdout output, so the stdout reservation must not
    capture it -- an earlier attempt installed the reservation in the group
    callback and sent every subcommand's `--help` to stderr. And because the
    reservation only covers a command's *body*, everything imported before
    that (`Auto3D.cli.app` and its module-level imports) has to stay silent on
    stdout on its own; a banner printed at import time would land here, ahead
    of the usage text.
    """
    result = auto3d_process("run", "--help")

    assert result.returncode == 0, result.stderr

    # Strip ANSI before matching. Rich emits a bold escape ahead of the usage
    # line whenever it believes the stream is colour-capable, and GitHub
    # Actions sets FORCE_COLOR, so `lstrip()` alone left a leading `\x1b[1m`
    # and this assertion failed on every CI runner while passing locally --
    # the same local/CI split that has produced several defects in this
    # effort. The subject here is that no banner precedes the usage text, not
    # how the usage text is styled.
    plain = _ANSI_RE.sub("", result.stdout).lstrip()

    assert plain.startswith("Usage: auto3d run"), result.stdout
    assert "Warp" not in result.stdout


def test_run_cli_k_override_substitutes_file_window(runner, tmp_path_cwd, monkeypatch):
    """`auto3d run in.smi -c cfg.yaml --k 1`, where cfg.yaml sets
    `window: 5.0`, must succeed with k=1 winning -- not hard-fail the
    mutual-exclusion rule because the CLI override was merged alongside the
    file's selector instead of substituting for it.

    `Auto3D.auto3D.main` is stubbed (captures the `Auto3DOptions` it would
    have received) so this exercises the full `execute_run` ->
    `load_yaml_config`/`merge_configs` -> `to_auto3d_options()` path without
    a pipeline run or a loaded potential. `optimizing_engine: ANI2xt` avoids
    importing the optional `aimnet` package (same reasoning as
    `test_cli.py`'s `_LEGACY_YAML`).
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")
    cfg = tmp_path_cwd / "cfg.yaml"
    cfg.write_text(
        "path: placeholder.smi\nwindow: 5.0\noptimizing_engine: ANI2xt\nuse_gpu: false\n"
    )

    import Auto3D.auto3D as a3d
    from Auto3D.results import WorkflowResult

    captured: dict = {}

    def fake_main(options, **kwargs):
        captured["options"] = options
        return WorkflowResult("fake_out.sdf")

    monkeypatch.setattr(a3d, "main", fake_main)

    result = runner.invoke(app, ["run", str(smi), "-c", str(cfg), "--k", "1", "--json"])

    assert result.exit_code == 0, result.output
    assert captured["options"].k == 1
    assert not captured["options"].window  # file's window=5.0 must be cleared


def test_run_cli_explicit_k_and_window_conflict_is_configuration_error(
    runner, tmp_path_cwd, monkeypatch
):
    """`--k` and `--window` both passed explicitly on the CLI is a genuine
    user conflict (not a file-vs-CLI merge artifact) and must still exit 2
    as a ConfigurationError with a hint -- not exit 1 as a raw pydantic
    ValidationError under "Unexpected Error".
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    import Auto3D.auto3D as a3d

    monkeypatch.setattr(
        a3d,
        "main",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("main() must not run when config validation fails")
        ),
    )

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--window", "2.0"])

    assert result.exit_code == 2, result.output
    assert "Unexpected Error" not in result.output


def test_run_cli_yaml_config_bounds_violation_is_configuration_error(
    runner, tmp_path_cwd, monkeypatch
):
    """`auto3d run in.smi -c cfg.yaml` with `cfg.yaml` setting `k: 0` must
    exit 2 as a ConfigurationError with a hint -- not exit 1 under
    "Unexpected Error" as a raw pydantic ValidationError.

    This is the load_yaml_config construction site specifically (as opposed
    to test_run_cli_explicit_k_and_window_conflict_is_configuration_error,
    which exercises merge_configs): the bad value comes from the config file
    itself, before any CLI override is merged in. Before this fix,
    `load_yaml_config` raised the pydantic error unwrapped, which
    `execute_run`'s `except Auto3DError` clause does not catch.
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")
    cfg = tmp_path_cwd / "cfg.yaml"
    cfg.write_text("path: placeholder.smi\nk: 0\noptimizing_engine: ANI2xt\nuse_gpu: false\n")

    import Auto3D.auto3D as a3d

    monkeypatch.setattr(
        a3d,
        "main",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("main() must not run when config validation fails")
        ),
    )

    result = runner.invoke(app, ["run", str(smi), "-c", str(cfg)])

    assert result.exit_code == 2, result.output
    assert "Unexpected Error" not in result.output


def test_run_cli_yaml_uncoercible_gpu_idx_is_configuration_error(runner, tmp_path_cwd, monkeypatch):
    """`auto3d run in.smi -c cfg.yaml` with `gpu_idx: {a: 1}` must exit 2 as
    a ConfigurationError -- not exit 1 under "Unexpected Error".

    Sibling of the `k: 0` case above, for the failure mode that case does not
    reach: `k: 0` violates a bound and so becomes a pydantic
    `ValidationError`, which `build_cli_config` already translated. A mapping
    in `gpu_idx` instead makes `CLIConfig.parse_gpu_idx`'s own `int(v)` raise
    `TypeError`, which pydantic re-raises untouched -- so it escaped
    `build_cli_config`'s `except ValidationError`, escaped `execute_run`'s
    `except Auto3DError`, and landed in the generic "Unexpected Error" panel
    at exit 1. Same user mistake (a bad value in a config file), same
    treatment required.
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")
    cfg = tmp_path_cwd / "cfg.yaml"
    cfg.write_text(
        "path: placeholder.smi\nk: 1\ngpu_idx:\n  a: 1\noptimizing_engine: ANI2xt\nuse_gpu: false\n"
    )

    import Auto3D.auto3D as a3d

    monkeypatch.setattr(
        a3d,
        "main",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("main() must not run when config validation fails")
        ),
    )

    result = runner.invoke(app, ["run", str(smi), "-c", str(cfg)])

    assert result.exit_code == 2, result.output
    assert "Unexpected Error" not in result.output


# Unit tests for the exit-code decision itself (Auto3D.cli.commands.run),
# pinned without going through the CLI or a pipeline run at all.


def test_exit_if_incomplete_raises_nonzero_when_failures_present():
    from Auto3D.cli.commands.run import EXIT_PARTIAL_SUCCESS, _exit_if_incomplete
    from Auto3D.cli.results import FailedMolecule, RunSummary

    results = RunSummary(
        success_count=1,
        failed_count=1,
        total_conformers=1,
        output_path="out.sdf",
        elapsed_seconds=0.1,
        failures=[FailedMolecule(name="mol2", error="missing from output")],
    )

    with pytest.raises(SystemExit) as exc_info:
        _exit_if_incomplete(results)
    assert exc_info.value.code == EXIT_PARTIAL_SUCCESS
    assert EXIT_PARTIAL_SUCCESS != 0


def test_exit_if_incomplete_does_not_raise_when_no_failures():
    from Auto3D.cli.commands.run import _exit_if_incomplete
    from Auto3D.cli.results import RunSummary

    results = RunSummary(
        success_count=2,
        failed_count=0,
        total_conformers=2,
        output_path="out.sdf",
        elapsed_seconds=0.1,
        failures=[],
    )

    _exit_if_incomplete(results)  # must not raise


# Error handling tests


def test_error_hint_configuration_error():
    """get_error_hint should return hint for ConfigurationError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import ConfigurationError

    hint = get_error_hint(ConfigurationError("test"))
    assert hint is not None
    assert "config init" in hint


def test_error_hint_input_validation_error():
    """get_error_hint should return hint for InputValidationError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import InputValidationError

    hint = get_error_hint(InputValidationError("test"))
    assert hint is not None
    assert "validate" in hint


def test_error_hint_gpu_error():
    """get_error_hint should return hint for GPUError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import GPUError

    hint = get_error_hint(GPUError("test"))
    assert hint is not None
    assert "--no-gpu" in hint


def test_handle_error_exits():
    """handle_error should raise SystemExit."""
    from Auto3D.cli.errors import handle_error

    with pytest.raises(SystemExit) as exc_info:
        handle_error(Exception("test error"))
    assert exc_info.value.code == 1


def test_models_info_aimnet2_pd(runner):
    """models info aimnet2-pd works and shows Pd in its actual element set.

    "Pd" also appears in three other lines of prose in this panel ("Best for
    Pd organometallic...", "Replaces As with Pd...", "...supported is Pd"),
    so a bare substring check passes even if the Supported Elements line
    itself dropped it. Parse that line specifically.
    """
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "aimnet2-pd"])
    assert result.exit_code == 0

    match = re.search(r"Supported Elements:\s*([A-Za-z, ]+)", result.stdout)
    assert match, f"no 'Supported Elements:' line found in:\n{result.stdout}"
    elements = {tok.strip() for tok in match.group(1).split(",")}
    assert "Pd" in elements


def test_models_info_aimnet2_alias(runner):
    """models info aimnet2 (the canonical default) must resolve to the AIMNET entry."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "aimnet2"])
    assert result.exit_code == 0
    assert "AIMNet2" in result.stdout


def test_models_info_unkeyed_aimnet2_variant_resolves(runner):
    """Any aimnet2-* registry name -- including a future variant without its own
    ENGINE_INFO block -- must resolve to the base AIMNet2 entry, not 'Unknown
    engine' (check_valid_configuration already accepts any aimnet2* name)."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "aimnet2-future"])
    assert result.exit_code == 0, result.stdout
    assert "AIMNet2" in result.stdout


def test_config_validate_missing_file(runner, tmp_path_cwd):
    """config validate on a missing file is now a Typer path-existence error (exit 2)."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "validate", "nonexistent.yaml"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


@pytest.mark.parametrize(
    "flag,value,field,expected",
    [
        ("--max-confs", "5", "max_confs", 5),
        ("--threshold", "0.7", "threshold", 0.7),
        ("--mpi-np", "2", "mpi_np", 2),
        ("--opt-steps", "77", "opt_steps", 77),
        ("--opt-tol", "0.05", "convergence_threshold", 0.05),
        ("--patience", "33", "patience", 33),
        ("--batchsize-atoms", "512", "batchsize_atoms", 512),
        ("--isomer-engine", "rdkit", "isomer_engine", "rdkit"),
        ("--tauto-engine", "rdkit", "tauto_engine", "rdkit"),
    ],
)
def test_run_forwards_each_new_flag_to_auto3d_options(
    runner, tmp_path_cwd, flag, value, field, expected
):
    """Each flag must reach the field it names, not merely be accepted.

    `run` exposed 7 of 23 Auto3DOptions fields; the rest were YAML-only even
    though `optimize`/`thermo` exposed their equivalents. Adding a flag that
    Typer accepts but `execute_run` never forwards would be invisible to a
    test that only checks the exit code -- the command would still exit 0 and
    the option would silently do nothing, which is the defect class this
    codebase keeps finding. Asserting on the constructed Auto3DOptions is what
    makes dropping the flag from the override dict fail this test.
    """
    from unittest.mock import patch

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "mols.smi"
    smi.write_text("CCO m1\n")

    with patch.object(Auto3D.auto3D, "main", return_value="out.sdf") as m:
        result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", flag, value])

    assert result.exit_code == 0, result.output
    assert m.called, f"{flag} prevented the run from starting"
    options = m.call_args[0][0]
    assert getattr(options, field) == expected, (
        f"{flag}={value} did not reach Auto3DOptions.{field}"
    )
