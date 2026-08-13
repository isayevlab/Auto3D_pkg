"""The shared backend registry."""

from __future__ import annotations

import pytest

from Auto3D.exceptions import ConfigurationError
from Auto3D.registry import Registry


def test_register_and_resolve():
    reg: Registry[str] = Registry("widget")
    reg.register("alpha", "A")
    assert reg.resolve("alpha") == "A"
    assert "alpha" in reg
    assert reg.available() == ["alpha"]


def test_aliases_resolve_to_the_same_value_but_are_not_listed():
    """An alias is a second spelling, not a second backend.

    ``available()`` is what the CLI shows and what error messages enumerate, so
    listing aliases there would advertise two engines where one exists.
    """
    reg: Registry[str] = Registry("widget")
    reg.register("alpha", "A", aliases=("a", "first"))
    assert reg.resolve("a") == reg.resolve("first") == "A"
    assert reg.available() == ["alpha"]
    assert "first" in reg


def test_unknown_name_names_the_alternatives():
    """The error has to say what *would* have worked.

    Two hand-written messages did this before, with different wording and only
    one of them quoting the alternatives. One implementation means one message.
    """
    reg: Registry[str] = Registry("optimizing engine")
    reg.register("alpha", "A")
    reg.register("beta", "B")
    with pytest.raises(ConfigurationError) as exc:
        reg.resolve("gamma")
    message = str(exc.value)
    assert "optimizing engine" in message
    assert "'gamma'" in message
    assert "'alpha'" in message and "'beta'" in message


def test_duplicate_registration_raises_instead_of_overwriting():
    """A dict would silently replace, making behavior depend on import order."""
    reg: Registry[str] = Registry("widget")
    reg.register("alpha", "A")
    with pytest.raises(ConfigurationError, match="already registered"):
        reg.register("alpha", "B")
    assert reg.resolve("alpha") == "A"


def test_duplicate_is_caught_across_name_and_alias():
    reg: Registry[str] = Registry("widget")
    reg.register("alpha", "A", aliases=("shared",))
    with pytest.raises(ConfigurationError, match="already registered"):
        reg.register("beta", "B", aliases=("shared",))


def test_case_sensitivity_is_configurable_because_both_kinds_exist():
    """Models resolve case-insensitively; isomer engine types do not.

    Both are current, tested behavior -- ``--engine ani2x`` works, while
    ``rdkit_sdf`` is matched exactly -- so this is configuration rather than a
    policy this module picks.
    """
    folding: Registry[str] = Registry("engine", case_insensitive=True)
    folding.register("ANI2x", "A")
    assert folding.resolve("ani2x") == folding.resolve("ANI2X") == "A"

    exact: Registry[str] = Registry("engine")
    exact.register("rdkit", "R")
    assert exact.resolve("rdkit") == "R"
    with pytest.raises(ConfigurationError):
        exact.resolve("RDKit")


def test_case_folding_also_applies_to_duplicate_detection():
    reg: Registry[str] = Registry("engine", case_insensitive=True)
    reg.register("ANI2x", "A")
    with pytest.raises(ConfigurationError, match="already registered"):
        reg.register("ani2x", "B")


def test_entry_carries_display_metadata():
    """``info`` is why the CLI stops keeping a parallel table.

    ``ENGINE_INFO`` was keyed by engine name and checked against nothing, so a
    backend registered without an entry vanished from ``auto3d models info``
    with no error anywhere.
    """
    reg: Registry[str] = Registry("engine")
    reg.register("alpha", "A", info={"description": "the first one"})
    assert reg.entry("alpha").info == {"description": "the first one"}
    assert reg.entry("alpha").value == "A"


def test_available_is_registration_order():
    """Not sorted: the order backends are declared in is the order shown."""
    reg: Registry[str] = Registry("engine")
    for name in ("gamma", "alpha", "beta"):
        reg.register(name, name.upper())
    assert reg.available() == ["gamma", "alpha", "beta"]
