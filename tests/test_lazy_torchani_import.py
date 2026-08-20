"""The optional torchani extra must not be required just to import thermo."""

from __future__ import annotations

import builtins
import sys

# Module names this test evicts from sys.modules so their module-level code
# re-runs under the import block.
_RELOADED = ("Auto3D.entry.ASE.thermo", "Auto3D.engines.models.ani2xt")


def test_thermo_imports_with_torchani_blocked(monkeypatch):
    """Importing Auto3D.entry.ASE.thermo (and using its pure-Python helpers) must work
    even when torchani cannot be imported. torchani is only needed to *construct*
    an ANI2xt model, not to import the module that references the class.
    """
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torchani" or name.startswith("torchani."):
            raise ImportError("torchani blocked for this test")
        return real_import(name, *args, **kwargs)

    # Drop cached copies so the re-import re-runs module-level code under the
    # block -- then put the originals back, because a re-import does not just
    # refresh a module, it creates a *second* module object with its own
    # globals. Other test modules hold `from Auto3D.entry.ASE.thermo import helper`
    # references bound to the first object, so leaving the second one in
    # sys.modules splits the module in two: a later test that patches
    # `thermo._symmetry_default_warned` patches the new module's global while
    # the helper it then calls reads the old module's. That is invisible here
    # and fails somewhere else -- it cost seed 12345 a failure in
    # test_thermo_helpers.py, 182 tests downstream of this one, with nothing in
    # the output pointing back here.
    saved_modules = {name: sys.modules[name] for name in _RELOADED if name in sys.modules}
    # A re-import also rebinds the leaf attribute on the parent package
    # (`Auto3D.entry.ASE.thermo`), which is how `import Auto3D.entry.ASE; Auto3D.entry.ASE.thermo`
    # resolves, so that needs restoring too.
    saved_parent_attrs = {}
    for name in saved_modules:
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, leaf):
            saved_parent_attrs[(parent, leaf)] = getattr(parent, leaf)

    for name in saved_modules:
        del sys.modules[name]

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    try:
        import Auto3D.entry.ASE.thermo.driver as thermo  # must NOT raise ModuleNotFoundError  # noqa: I001

        from rdkit import Chem

        # A pure-Python helper that does not touch torchani must work.
        assert thermo._symmetry_number(Chem.MolFromSmiles("CCO")) == 1
    finally:
        for name, module in saved_modules.items():
            sys.modules[name] = module
        for (parent, leaf), module in saved_parent_attrs.items():
            setattr(parent, leaf, module)
