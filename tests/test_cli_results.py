# tests/test_cli_results.py
"""Tests for results display components."""

import pytest


def test_format_duration():
    """format_duration should format seconds nicely."""
    from Auto3D.cli.results import format_duration

    assert format_duration(65) == "1m 5s"
    assert format_duration(3661) == "1h 1m 1s"
    assert format_duration(45) == "45s"


def test_format_duration_zero():
    """format_duration should handle zero."""
    from Auto3D.cli.results import format_duration

    assert format_duration(0) == "0s"


def test_workflow_results_dataclass():
    """WorkflowResults should be a valid dataclass."""
    from Auto3D.cli.results import WorkflowResults

    results = WorkflowResults(
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
    from Auto3D.cli.results import FailedMolecule

    failure = FailedMolecule(name="mol1", error="Invalid SMILES")
    assert failure.name == "mol1"
    assert failure.error == "Invalid SMILES"


def test_print_results_summary():
    """print_results_summary should not crash."""
    from Auto3D.cli.results import print_results_summary, WorkflowResults

    results = WorkflowResults(
        success_count=10,
        failed_count=2,
        total_conformers=50,
        output_path="output.sdf",
        elapsed_seconds=120.5,
        failures=[],
    )

    # Should not raise
    print_results_summary(results)


def test_print_failures_empty():
    """print_failures should handle empty list."""
    from Auto3D.cli.results import print_failures

    # Should not raise
    print_failures([])


def test_print_failures_verbose():
    """print_failures should show table in verbose mode."""
    from Auto3D.cli.results import print_failures, FailedMolecule

    failures = [FailedMolecule(name=f"mol{i}", error="Error") for i in range(5)]
    # Should not raise
    print_failures(failures, verbose=True)


def test_output_json():
    """output_json should not crash."""
    from Auto3D.cli.results import output_json, WorkflowResults

    results = WorkflowResults(
        success_count=10,
        failed_count=0,
        total_conformers=50,
        output_path="output.sdf",
        elapsed_seconds=120.5,
    )

    # Should not raise
    output_json(results)
