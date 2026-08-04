# tests/test_public_api.py
"""Lock the public API surface: everything in Auto3D.__all__ resolves, the
generate_conformers alias points at main, the tautomer functions are public,
and the property/tautomer modules declare their own __all__."""
from __future__ import annotations

import importlib


def test_all_public_names_resolve():
    """Every name in ``Auto3D.__all__`` must resolve to a callable/class.

    Iterating ``Auto3D.__all__`` itself and checking each entry resolves is
    self-referential: it catches a name left in ``__all__`` with nothing
    behind it, but a name silently *dropped* from ``__all__`` (the public
    surface quietly shrinking) would still pass, since the loop would just
    iterate over less. Pin the expected membership independently so that
    regression is caught too.
    """
    import Auto3D

    expected = {
        "__version__",
        "main",
        "generate_conformers",
        "smiles2mols",
        "Auto3DOptions",
        "OptimizationConfig",
        "create_model",
        "ModelFactory",
        "calc_spe",
        "opt_geometry",
        "calc_thermo",
        "get_stable_tautomers",
        "select_tautomers",
    }
    assert set(Auto3D.__all__) == expected

    for name in Auto3D.__all__:
        if name == "__version__":
            continue
        assert callable(getattr(Auto3D, name)), name


def test_generate_conformers_is_main():
    import Auto3D
    assert Auto3D.generate_conformers is Auto3D.main


def test_tautomer_functions_public():
    import Auto3D
    assert callable(Auto3D.get_stable_tautomers)
    assert callable(Auto3D.select_tautomers)


def test_module_all_declared():
    for mod_name, expected in (
        ("Auto3D.SPE", {"calc_spe"}),
        ("Auto3D.ASE.geometry", {"opt_geometry"}),
        ("Auto3D.ASE.thermo", {"calc_thermo"}),
        ("Auto3D.tautomer", {"select_tautomers", "get_stable_tautomers"}),
    ):
        mod = importlib.import_module(mod_name)
        assert set(mod.__all__) == expected, mod_name
