"""The workflow orchestrator is composed from three mixins; its old module keeps its public names."""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

#: The sequencer and the workers module: no mixin may import either at runtime.
_FORBIDDEN_MODULES = frozenset(
    {"Auto3D.orchestration.workflow", "Auto3D.orchestration.workflow_workers"}
)
#: Every mixin module lives directly in this package, so the package for
#: resolving a relative import's level is fixed rather than derived per-file.
_MIXIN_PACKAGE = ("Auto3D", "orchestration")


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Whether an ``If`` node's test is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def _absolute_module(node: ast.ImportFrom) -> str:
    """Absolute module name an ``ImportFrom`` names, resolving relative levels."""
    if not node.level:
        return node.module or ""
    package = _MIXIN_PACKAGE if node.level == 1 else _MIXIN_PACKAGE[: -(node.level - 1)]
    return ".".join((*package, node.module)) if node.module else ".".join(package)


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
    """No mixin may import ``workflow`` (the sequencer it composes into) or
    ``workflow_workers`` (the spawn targets) at runtime.

    Checked by AST rather than by substring: a textual check over the
    source over- and under-blocks (PR 4 architecture review, M2) -- it
    cannot tell a real import from a comment or a docstring, and it misses
    ``from Auto3D.orchestration import workflow_workers`` and ``from
    .workflow_workers import X``, neither of which contains the forbidden
    substring. ``if TYPE_CHECKING:`` imports are exempt: they create no
    runtime edge (``from __future__ import annotations`` means the names
    they bring in are never evaluated outside a type checker), which is
    exactly why ``supervisor.py`` imports ``ProgressEvent`` that way.
    """
    import sys

    offenders = []
    for name in ("run_logging", "supervisor", "finalize"):
        module = importlib.import_module(f"Auto3D.orchestration.{name}")
        path = pathlib.Path(module.__file__)
        tree = ast.parse(path.read_text(), filename=str(path))

        type_checking_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        type_checking_ids.add(id(sub))

        for node in ast.walk(tree):
            if id(node) in type_checking_ids:
                continue
            if isinstance(node, ast.ImportFrom):
                base = _absolute_module(node)
                targets = {base} | {
                    f"{base}.{alias.name}" if base else alias.name for alias in node.names
                }
                if targets & _FORBIDDEN_MODULES:
                    offenders.append(f"{name}.py:{node.lineno}: from {base} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULES:
                        offenders.append(f"{name}.py:{node.lineno}: import {alias.name}")
    assert not offenders, (
        "a mixin module imports the workers module or the sequencer it composes "
        "into at runtime:\n" + "\n".join(offenders)
    )
    assert "Auto3D.orchestration.workflow" in sys.modules
