"""Dependency-hiding helpers for every optional package Auto3D can run without.

Dev and CI environments install aimnet (a required pip dependency) and, on
most hosts, torchani too, so a test of the code path that runs without one of
them must block the import rather than relying on absence.
``sys.modules[name] = None`` makes ``import name`` raise
``ModuleNotFoundError`` with ``exc.name`` set -- exactly what a genuinely
missing package raises -- and ``monkeypatch`` restores everything afterwards.
These helpers also work in the ``no-aimnet`` CI job, where aimnet really is
absent: the override is then a no-op on top of reality.
"""

from __future__ import annotations

import sys


def hide_optional_dependency(monkeypatch, name: str, *, reset_model_cache: bool = False) -> None:
    """Make ``import <name>`` raise ``ModuleNotFoundError(name=<name>)``.

    Cached submodules are evicted first (``from name.sub import X`` resolves
    through ``sys.modules["name.sub"]`` and would never consult the blocked
    parent). ``reset_model_cache`` swaps ``ModelFactory._cache`` for an empty
    dict so a model built by an earlier test cannot satisfy a request without
    reaching the import; monkeypatch restores the populated cache afterwards.
    Without it, a cached model resolves the request without ever reaching the
    blocked import, so a test asserting the dependency error fails -- but
    only when the suite happens to run a model build first, which is why this
    surfaced as an intermittent failure under ``pytest-randomly`` rather than
    a reliable one.
    """
    for cached in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, cached)
    monkeypatch.setitem(sys.modules, name, None)
    if reset_model_cache:
        from Auto3D.engines.model_factory import ModelFactory

        monkeypatch.setattr(ModelFactory, "_cache", {})


def hide_aimnet(monkeypatch) -> None:
    """``hide_optional_dependency`` for aimnet; the name the no-aimnet tests import."""
    hide_optional_dependency(monkeypatch, "aimnet")


class BrokenTransitiveDepFinder:
    """Meta-path finder making ``import aimnet`` fail the way a broken
    *environment* does: aimnet present/findable, but its import dies on a
    missing transitive dependency (``warp`` here). Used to assert that case
    is NOT translated into "install aimnet".

    Use with::

        for name in [m for m in sys.modules if m == "aimnet" or m.startswith("aimnet.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [BrokenTransitiveDepFinder()] + sys.meta_path)
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aimnet":
            raise ModuleNotFoundError("No module named 'warp'", name="warp")
        return None
