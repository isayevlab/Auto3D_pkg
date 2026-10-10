"""The workflow orchestrator is composed from three mixins; its old module keeps its public names."""

from __future__ import annotations

import importlib

import pytest


def test_the_three_mixin_modules_exist():
    for name in ("run_logging", "supervisor", "finalize"):
        importlib.import_module(f"Auto3D.orchestration.{name}")


def test_the_orchestrator_is_composed_from_the_three_mixins():
    from Auto3D.orchestration.finalize import FinalizeMixin
    from Auto3D.orchestration.run_logging import RunLoggingMixin
    from Auto3D.orchestration.supervisor import SupervisorMixin
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    assert WorkflowOrchestrator.__bases__ == (RunLoggingMixin, SupervisorMixin, FinalizeMixin)


@pytest.mark.parametrize(
    ("name", "home"),
    [
        ("_shutdown_logging", "Auto3D.orchestration.run_logging"),
        ("_log_both", "Auto3D.orchestration.run_logging"),
        ("_join_workers", "Auto3D.orchestration.supervisor"),
        ("_supervise_with_progress", "Auto3D.orchestration.supervisor"),
        ("_finalize_output", "Auto3D.orchestration.finalize"),
        ("_combine_chunk_outputs", "Auto3D.orchestration.finalize"),
        ("_reconcile_output", "Auto3D.orchestration.finalize"),
        ("_run_pipeline", "Auto3D.orchestration.workflow"),
        ("_setup_logging", "Auto3D.orchestration.workflow"),
        ("_start_manager", "Auto3D.orchestration.workflow"),
    ],
)
def test_each_method_is_defined_in_its_home_module(name, home):
    from Auto3D.orchestration.workflow import WorkflowOrchestrator

    assert getattr(WorkflowOrchestrator, name).__module__ == home


def test_the_old_module_keeps_its_public_names():
    from Auto3D.orchestration import workflow

    for name in ("WorkflowOrchestrator", "_raise_on_sigterm", "_sigterm_raises"):
        assert hasattr(workflow, name), name


def test_no_mixin_module_imports_the_workers_or_the_sequencer():
    import sys

    for name in ("run_logging", "supervisor", "finalize"):
        importlib.import_module(f"Auto3D.orchestration.{name}")
    src = {
        name: open(importlib.import_module(f"Auto3D.orchestration.{name}").__file__).read()
        for name in ("run_logging", "supervisor", "finalize")
    }
    for name, text in src.items():
        assert "orchestration.workflow_workers" not in text, name
        assert "orchestration.workflow import" not in text, name
    assert "Auto3D.orchestration.workflow" in sys.modules
