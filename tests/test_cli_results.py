# tests/test_cli_results.py
"""Tests for results display components."""

import pytest


def test_format_duration():
    """format_duration should format seconds nicely."""
    from Auto3D.presentation.cli.results import format_duration

    assert format_duration(65) == "1m 5s"
    assert format_duration(3661) == "1h 1m 1s"
    assert format_duration(45) == "45s"


def test_format_duration_zero():
    """format_duration should handle zero."""
    from Auto3D.presentation.cli.results import format_duration

    assert format_duration(0) == "0s"


def test_workflow_results_dataclass():
    """RunSummary should be a valid dataclass."""
    from Auto3D.presentation.cli.results import RunSummary

    results = RunSummary(
        success_count=10,
        failed_count=2,
        total_conformers=50,
        output_path="output.sdf",
        elapsed_seconds=120.5,
    )

    assert results.success_count == 10
    assert results.failed_count == 2
    assert results.failures == []


def test_failed_molecule_dataclass():
    """FailedMolecule should be a valid dataclass."""
    from Auto3D.presentation.cli.results import FailedMolecule

    failure = FailedMolecule(name="mol1", error="Invalid SMILES")
    assert failure.name == "mol1"
    assert failure.error == "Invalid SMILES"


def test_print_results_summary(capsys):
    """print_results_summary must render the actual result fields -- and hide
    the failed-count row entirely when nothing failed, not merely avoid
    crashing on either input.
    """
    from Auto3D.presentation.cli.results import RunSummary, print_results_summary

    with_failures = RunSummary(
        success_count=10,
        failed_count=2,
        total_conformers=50,
        output_path="output.sdf",
        elapsed_seconds=120.5,
        failures=[],
    )
    print_results_summary(with_failures)
    out = capsys.readouterr().out
    assert "10" in out and "succeeded" in out
    assert "2" in out and "failed" in out
    assert "50" in out and "generated" in out
    assert "output.sdf" in out
    assert "2m" in out  # format_duration(120.5)

    no_failures = RunSummary(
        success_count=10,
        failed_count=0,
        total_conformers=50,
        output_path="output.sdf",
        elapsed_seconds=120.5,
    )
    print_results_summary(no_failures)
    out_clean = capsys.readouterr().out
    assert "failed" not in out_clean, "the failed-count row must not appear when nothing failed"


def test_print_failures_empty(capsys):
    """print_failures must be a true no-op for an empty list -- no output at
    all, not merely a call that happens not to raise.
    """
    from Auto3D.presentation.cli.results import print_failures

    print_failures([])
    assert capsys.readouterr().out == ""


def test_print_failures_verbose(capsys):
    """print_failures(verbose=True) must render the actual failure table."""
    from Auto3D.presentation.cli.results import FailedMolecule, print_failures

    failures = [FailedMolecule(name=f"mol{i}", error="Error") for i in range(5)]
    print_failures(failures, verbose=True)

    out = capsys.readouterr().out
    assert "5 molecules failed" in out
    for i in range(5):
        assert f"mol{i}" in out
    assert "Error" in out
    assert "Run with -v" not in out, "verbose mode must show the table, not the -v hint"


def test_output_json(capsys):
    """output_json must emit the exact result fields as parseable JSON."""
    import json

    from Auto3D.presentation.cli.results import FailedMolecule, RunSummary, output_json

    results = RunSummary(
        success_count=10,
        failed_count=1,
        total_conformers=50,
        output_path="output.sdf",
        elapsed_seconds=120.5,
        failures=[FailedMolecule(name="mol1", error="boom")],
    )

    output_json(results)

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "success": False,
        "molecules": 10,
        "failed": 1,
        "conformers": 50,
        "output_file": "output.sdf",
        "elapsed_seconds": 120.5,
        "failures": [{"name": "mol1", "error": "boom"}],
    }
