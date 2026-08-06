"""``auto3d models``'s dependency probe must answer about the name it is given.

``check_dependency_status`` special-cased ``"torchani"`` and fell through to an
unconditional ``(True, "Available")`` for every other name -- an answer reached
without importing anything. The fallthrough was unreachable in practice (the one
production call site passes ``"torchani"``), so it never reported a wrong status;
but a probe that says "Available" for a package it never looked for is a trap for
the next engine added to the table, which is exactly what the models command is
for.
"""
from __future__ import annotations

from Auto3D.cli.commands.models import check_dependency_status


def test_an_installed_dependency_is_reported_available():
    """The premise: a package that really is importable reports available."""
    available, status = check_dependency_status("rdkit")

    assert available is True
    assert "not installed" not in status.lower()


def test_a_missing_dependency_is_not_reported_available():
    """A name the probe cannot import must not come back as available."""
    available, status = check_dependency_status("a_package_that_is_not_installed")

    assert available is False, (
        "the probe claimed a package was available without importing it"
    )
    assert "not installed" in status.lower(), status


def test_torchani_is_still_probed_by_import():
    """The one production call site; its answer must track the real import."""
    try:
        import torchani  # noqa: F401
    except ImportError:
        expected = False
    else:
        expected = True

    available, _ = check_dependency_status("torchani")

    assert available is expected
