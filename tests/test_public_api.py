# tests/test_public_api.py
"""Lock the public API surface: everything in Auto3D.__all__ resolves, the
generate_conformers alias points at main, the tautomer functions are public,
and the property/tautomer modules declare their own __all__."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


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
        "ProgressEvent",
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
        ("Auto3D.entry.SPE", {"calc_spe"}),
        ("Auto3D.entry.ASE.geometry", {"opt_geometry"}),
        ("Auto3D.entry.ASE.thermo", {"calc_thermo"}),
        ("Auto3D.entry.tautomer", {"select_tautomers", "get_stable_tautomers"}),
    ):
        mod = importlib.import_module(mod_name)
        assert set(mod.__all__) == expected, mod_name


CONSUMER_HALF = {
    "forward": (["self", "coords", "species", "charges", "atom_mask"], {"atom_mask": None}),
    "energy": (["self", "coords", "species", "charges", "atom_mask"], {"atom_mask": None}),
    "to_species": (["self", "atomic_numbers"], {}),
}


def _adapter_classes():
    # Hand-listed, not derived from ModelFactory: a fifth in-tree backend
    # adapter must be added to this list by hand, or this pin silently
    # stops covering it.
    from Auto3D.engines.models.adapter import (
        AIMNet2Adapter,
        ANI2xAdapter,
        ANI2xtAdapter,
        CustomModelAdapter,
    )

    return [AIMNet2Adapter, ANI2xAdapter, ANI2xtAdapter, CustomModelAdapter]


@pytest.mark.parametrize("name", sorted(CONSUMER_HALF))
def test_the_consumer_half_of_the_adapter_contract_is_frozen(name):
    """D4: ``forward``, ``energy`` and ``to_species`` are public on every adapter
    ``create_model`` can return, with these parameter names and defaults.
    Checked on the classes, so no model is loaded."""
    names, defaults = CONSUMER_HALF[name]
    for cls in _adapter_classes():
        sig = inspect.signature(getattr(cls, name))
        assert list(sig.parameters) == names, (cls.__name__, name)
        for param, default in defaults.items():
            assert sig.parameters[param].default == default, (cls.__name__, name, param)


def test_every_other_contract_member_is_supplied_by_the_base_class():
    """D4: a new ``ModelAdapter`` member is always supplied by ``BaseModelAdapter``,
    never required of an implementer."""
    from Auto3D.engines.models.adapter import BaseModelAdapter
    from Auto3D.engines.models.contract import ModelAdapter, _protocol_members

    # _protocol_members is hand-written specifically to avoid
    # typing.Protocol.__protocol_attrs__, a CPython implementation detail
    # that is unavailable on Python 3.11 (part of this repo's supported
    # matrix) -- see contract.py's own docstring on _protocol_data_members.
    # Using the same version-independent oracle production code already
    # relies on means this pin does not silently degrade on half the CI matrix.
    members = {name for name in _protocol_members(ModelAdapter) if not name.startswith("_")}
    data = {"coord_pad", "species_pad"}
    init_params = set(inspect.signature(BaseModelAdapter.__init__).parameters)
    for name in members - set(CONSUMER_HALF):
        if name in data:
            assert name in init_params, name
        else:
            assert callable(getattr(BaseModelAdapter, name, None)), name


def test_api_docs_name_the_public_adapter_half():
    text = (Path(__file__).resolve().parents[1] / "docs" / "source" / "api.rst").read_text()
    assert "Auto3D.engines.models.contract.ModelAdapter" in text
    assert "consumer half" in text
    assert "only public name" not in text
