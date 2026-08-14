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

    assert available is False, "the probe claimed a package was available without importing it"
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


def test_a_dependency_that_raises_on_import_is_reported_not_available(monkeypatch):
    """Installed-but-broken is a third state, and it must not kill the command.

    CUDA-linked packages raise ``OSError``/``RuntimeError`` rather than
    ``ImportError`` when the driver is wrong. ``auto3d models list`` is the
    command a user runs to find out what is broken, so a probe that propagates
    takes down the diagnosis along with the dependency.
    """
    import Auto3D.cli.commands.models as models_mod

    def _explode(name):
        raise OSError("libcudart.so.12: cannot open shared object file")

    monkeypatch.setattr(models_mod.importlib, "import_module", _explode)

    available, status = check_dependency_status("torchani")

    assert available is False
    assert "OSError" in status, status


def test_engine_info_covers_exactly_the_registered_engines():
    """`ENGINE_INFO` and the engine registry must name the same set.

    These are two lists of engine names in different layers, and until now
    nothing connected them. The failure they allow is silent in the worst
    direction: register an engine and forget the display entry, and it works
    perfectly while never appearing in ``auto3d models list`` or ``models
    info``. A user has no way to discover it, and no error says why.

    The design spec for the registry proposed moving this table into the
    registry entries as ``info=``. That is the wrong direction on reflection:
    ``ENGINE_INFO`` is presentation content -- descriptions, references, prose
    notes -- and ``model_factory`` is the engine layer. Pushing display strings
    down two layers to remove a duplication would trade a checkable problem for
    a structural one, and ``tests/test_layer_boundaries.py`` exists to stop
    exactly that kind of drift downward.

    What the move was actually for is this property, and the property does not
    require the move. Asserted in both directions: an engine without display
    metadata fails, and metadata for an engine that no longer exists fails too.
    """
    from Auto3D.cli.commands.models import ENGINE_INFO
    from Auto3D.model_factory import ModelFactory

    registered = {name.upper() for name in ModelFactory.available_models()}
    documented = set(ENGINE_INFO)

    assert documented - registered == set(), (
        f"ENGINE_INFO describes engines that are not registered: {sorted(documented - registered)}"
    )
    assert registered - documented == set(), (
        "these engines are registered but have no ENGINE_INFO entry, so they "
        "are invisible in `auto3d models list`/`models info`: "
        f"{sorted(registered - documented)}"
    )
