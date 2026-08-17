# tests/test_element_sets.py
"""Each engine's element set, and the one place it is written down.

The ANI element set appeared five times across three layers -- a numeric
``frozenset`` in ``models/policy.py`` (the gate), the keys of ``ANI2XT_INDEX`` in
``models/species.py`` (the remap), a hand-written symbol string in that module's
error message, and two more copies of that string in the CLI's ``ENGINE_INFO``.
All five agreed, by hand, with nothing connecting them. That is the same shape as
the three parallel engine-name lists the registry collapsed: correct today,
correct only until someone edits one of them.

AIMNet2's sets are different in kind and are handled differently here. The
authoritative source is the model file's own ``implemented_species`` metadata,
which ``AIMNet2Calculator`` enforces at call time -- so Auto3D does not get to
define them, only to quote them correctly. The slow test at the bottom is what
makes "correctly" checkable.
"""

from __future__ import annotations

import pytest

from Auto3D.models.policy import ANI_ELEMENTS
from Auto3D.models.species import ANI2XT_INDEX, format_elements

#: The string every one of these sites has always shown. Written out rather than
#: computed, so that a change to the renderer has something to be wrong against.
ANI_ELEMENT_STRING = "H, C, N, O, F, S, Cl"


def test_format_elements_orders_by_atomic_number():
    """Not sorted by symbol, and not in set-iteration order.

    Atomic number is the order every existing string already used, which is what
    lets this renderer replace all of them without changing a single character of
    user-visible output.
    """
    assert format_elements({8, 1, 6}) == "H, C, O"
    assert format_elements({53, 35, 46, 34}) == "Se, Br, Pd, I"


def test_the_ani_element_string_is_the_one_it_has_always_been():
    """The behavior lock. If this fails, the renderer changed the output."""
    assert format_elements(ANI_ELEMENTS) == ANI_ELEMENT_STRING


def test_ani2xt_index_covers_exactly_the_ani_element_set():
    """The gate and the remap agree -- asserted, not assumed.

    Deliberately an equality check rather than deriving one from the other. They
    are the same seven numbers but not the same fact: ``ANI_ELEMENTS`` is what
    ANI2x and ANI2xt were *trained* on, ``ANI2XT_INDEX`` is one engine's 0-based
    network index order. Defining either in terms of the other would record a
    provenance that is not true, and would quietly stop being a check.
    """
    assert ANI_ELEMENTS == frozenset(ANI2XT_INDEX)


def test_unsupported_element_message_names_the_supported_set(caplog):
    """The remap's rejection message renders from the set it enforces."""
    from Auto3D.models.species import to_ani2xt_species

    with pytest.raises(ValueError) as exc:
        to_ani2xt_species([1, 6, 5])  # boron: in AIMNet2's set, not ANI's

    message = str(exc.value)
    assert "Z=5" in message and "(B)" in message
    assert ANI_ELEMENT_STRING in message


def test_engine_info_ani_entries_render_from_the_element_set():
    """The CLI quotes the gate rather than restating it.

    Two entries carried this string as a literal. A retrained ANI with a
    different element set would have moved the gate and left the CLI advertising
    the old one -- and ``auto3d models info`` is where a user checks precisely
    this before choosing an engine.
    """
    from Auto3D.cli.commands.models import ENGINE_INFO

    for name in ("ANI2X", "ANI2XT"):
        assert ENGINE_INFO[name]["elements"] == format_elements(ANI_ELEMENTS), (
            f"{name}'s advertised element set no longer matches the set "
            f"check_engine_supports_molecules actually enforces"
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("info_key", "registry_name"),
    [
        ("AIMNET", "aimnet2"),
        ("AIMNET2-2025", "aimnet2-2025"),
        ("AIMNET2-NSE", "aimnet2-nse"),
        ("AIMNET2-PD", "aimnet2-pd"),
    ],
)
def test_aimnet_engine_info_elements_match_the_model_metadata(info_key, registry_name):
    """What ``auto3d models info`` advertises is what the model file declares.

    These four strings stay literals in ``ENGINE_INFO`` -- unlike the ANI pair
    above -- because deriving them would mean loading four NNPs at CLI import.
    This test is the alternative: it loads them once, in the slow tier, and pins
    the literals to ground truth. ``AIMNet2Calculator`` reads the same
    ``implemented_species`` to reject out-of-set atomic numbers at call time, so
    a mismatch here is the CLI promising chemistry the engine will refuse.

    **If this goes red, the literal is stale -- do not delete the test.** It fails
    exactly when aimnet ships a model whose element set changed, which is the one
    moment the advertised string needs updating and the one moment nothing else
    would say so.

    CPU-only and constructed directly rather than through ``create_model``: the
    subject is what the aimnet package declares, not how Auto3D wraps it, and
    reaching into an adapter for it would be the abstraction leak this codebase
    just finished removing.
    """
    import torch

    from Auto3D.cli.commands.models import ENGINE_INFO

    aimnet_calculators = pytest.importorskip("aimnet.calculators")

    calc = aimnet_calculators.AIMNet2Calculator(registry_name, device=torch.device("cpu"))
    implemented = (calc.metadata or {}).get("implemented_species")
    assert implemented is not None, (
        f"{registry_name} declares no implemented_species, so aimnet's own "
        f"element validation is a silent no-op for it and this test cannot "
        f"check the advertised set"
    )
    numbers = implemented.tolist() if hasattr(implemented, "tolist") else list(implemented)

    assert ENGINE_INFO[info_key]["elements"] == format_elements(numbers)
