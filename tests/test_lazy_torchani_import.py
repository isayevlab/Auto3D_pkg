"""The optional torchani extra must not be required just to import thermo."""
from __future__ import annotations

import builtins
import sys


def test_thermo_imports_with_torchani_blocked(monkeypatch):
    """Importing Auto3D.ASE.thermo (and using its pure-Python helpers) must work
    even when torchani cannot be imported. torchani is only needed to *construct*
    an ANI2xt model, not to import the module that references the class.
    """
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torchani" or name.startswith("torchani."):
            raise ImportError("torchani blocked for this test")
        return real_import(name, *args, **kwargs)

    # Drop cached copies so the re-import re-runs module-level code under the block.
    for mod in list(sys.modules):
        if mod.startswith("Auto3D.ASE.thermo") or mod.startswith(
            "Auto3D.batch_opt.ANI2xt_no_rep"
        ):
            sys.modules.pop(mod, None)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    import Auto3D.ASE.thermo as thermo  # must NOT raise ModuleNotFoundError  # noqa: I001

    from rdkit import Chem

    # A pure-Python helper that does not touch torchani must work.
    assert thermo._symmetry_number(Chem.MolFromSmiles("CCO")) == 1
