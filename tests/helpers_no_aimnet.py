"""Simulate an environment without the ``aimnet`` package.

Dev and CI environments install aimnet (it is a required pip dependency), so
tests of the no-aimnet path must block the import instead of relying on
absence. Same technique as ``_hide_torchani`` in ``test_cli_exit_codes.py``:
``sys.modules[name] = None`` makes ``import name`` raise
``ModuleNotFoundError`` with ``exc.name`` set -- exactly what a genuinely
missing package raises -- and ``monkeypatch`` restores everything afterwards.
These helpers also work in the ``no-aimnet`` CI job, where aimnet really is
absent: the override is then a no-op on top of reality.
"""

from __future__ import annotations

import sys


def hide_aimnet(monkeypatch) -> None:
    """Make ``import aimnet`` and ``from aimnet... import ...`` raise
    ``ModuleNotFoundError(name="aimnet")``.

    Cached submodules must be evicted too: ``from aimnet.calculators import
    X`` resolves via ``sys.modules["aimnet.calculators"]`` when that entry is
    cached and would never consult the blocked parent entry.
    """
    for name in [m for m in sys.modules if m == "aimnet" or m.startswith("aimnet.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "aimnet", None)


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
