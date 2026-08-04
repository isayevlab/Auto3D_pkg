"""The single owner of atomic-number -> model-species-index conversion.

ANI2xt is constructed with `periodic_table_index=False` everywhere, so it
expects 0-based indices (H=0..Cl=6), not atomic numbers. Before this module
existed the conversion was duplicated in three places and omitted in two more
(audit C3, C4).

The table now lives in ``Auto3D.models.species``, not ``Auto3D.batch_opt.species``:
the species convention is a property of the MODEL, and hosting it under
``batch_opt`` made the optimizer package a shared-utility provider for ``ASE/``,
``cli/`` and ``models/``'s own padder. The canonical way to reach it is
``ModelAdapter.to_species`` -- asking the object that also supplies
``species_pad``, so the two cannot disagree.
"""
from __future__ import annotations

import pytest

from Auto3D.models.species import ANI2XT_INDEX, to_model_species


class TestAni2xtMapping:
    """ANI2xt species are 0-based network indices."""

    def test_methane_maps_to_indices(self):
        """Carbon and four hydrogens become [1, 0, 0, 0, 0]."""
        assert to_model_species([6, 1, 1, 1, 1], "ANI2xt") == [1, 0, 0, 0, 0]

    def test_all_seven_supported_elements(self):
        """H, C, N, O, F, S, Cl map to 0..6 in that order."""
        assert to_model_species([1, 6, 7, 8, 9, 16, 17], "ANI2xt") == [0, 1, 2, 3, 4, 5, 6]

    def test_unsupported_element_names_itself_and_the_model(self):
        """Sodium is outside ANI2xt's set; the error must be actionable."""
        with pytest.raises(ValueError) as exc:
            to_model_species([11], "ANI2xt")
        message = str(exc.value)
        assert "11" in message, "error must name the atomic number"
        assert "Na" in message, "error must name the element symbol"
        assert "ANI2xt" in message, "error must name the model"


class TestCaseInsensitiveMatch:
    """create_model dispatches case-insensitively (model_factory.py: name.upper()
    in cls._adapters), so to_model_species must match that convention -- otherwise
    `auto3d models test ani2xt` loads the correct adapter via create_model and then
    silently evaluates raw atomic numbers through it (a C4-shaped regression hiding
    behind incidental case-matching)."""

    @pytest.mark.parametrize("model_name", ["ani2xt", "ANI2XT", "Ani2xt"])
    def test_ani2xt_matches_regardless_of_case(self, model_name):
        """Any casing of the engine name must still convert to indices."""
        assert to_model_species([6, 1, 1, 1, 1], model_name) == [1, 0, 0, 0, 0]


class TestPassthroughModels:
    """Every other engine consumes atomic numbers unchanged."""

    @pytest.mark.parametrize("model_name", ["AIMNET", "ANI2x", "aimnet2", "userNNP"])
    def test_atomic_numbers_pass_through(self, model_name):
        """Only ANI2xt remaps; everything else is identity."""
        numbers = [1, 6, 7, 8, 11, 26]
        assert to_model_species(numbers, model_name) == numbers

    def test_passthrough_does_not_reject_exotic_elements(self):
        """Iron is meaningless to ANI2xt but fine for a custom model."""
        assert to_model_species([26], "AIMNET") == [26]


class TestIndexMapIsCanonical:
    """ANI2XT_INDEX is the one source of truth."""

    def test_map_contents(self):
        """The map must match ANI2xt's network ordering exactly."""
        assert ANI2XT_INDEX == {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

    def test_no_duplicate_indices(self):
        """Every element gets a distinct network slot."""
        assert len(set(ANI2XT_INDEX.values())) == len(ANI2XT_INDEX)


class TestOnlyAni2xtRemaps:
    """A remap leaking onto any other adapter is a silent, untestable disaster.

    ``CustomModelAdapter`` in particular MUST inherit the identity: a custom NNP
    receives atomic numbers and declares its own ``species_pad``, so remapping
    for it would feed every third-party model different species than its author
    tested against -- on the one path with no in-tree test molecules. This is
    audit C3/C4 inverted.
    """

    def test_base_adapter_to_species_is_the_identity(self):
        from Auto3D.models.adapter import BaseModelAdapter

        assert BaseModelAdapter.to_species(object(), [1, 6, 17]) == [1, 6, 17]

    def test_ani2xt_is_the_only_adapter_that_overrides_it(self):
        from Auto3D.models.adapter import (
            AIMNet2Adapter,
            ANI2xAdapter,
            ANI2xtAdapter,
            BaseModelAdapter,
            CustomModelAdapter,
        )

        overriding = {
            cls.__name__
            for cls in (
                AIMNet2Adapter,
                ANI2xAdapter,
                ANI2xtAdapter,
                CustomModelAdapter,
            )
            if cls.to_species is not BaseModelAdapter.to_species
        }
        assert overriding == {"ANI2xtAdapter"}, (
            f"{overriding - {'ANI2xtAdapter'}} redefine to_species; every engine "
            "but ANI2xt consumes raw atomic numbers"
        )

    def test_ani2xt_adapter_delegates_to_this_module(self, monkeypatch):
        """The adapter method must not carry a second copy of the table.

        Checked by delegation rather than by constructing an ``ANI2xtAdapter``:
        construction loads ``models/ani2xt_no_repulsion.pt`` and needs torchani
        for the AEV computer, neither of which belongs in the fast tier.
        """
        from Auto3D.models import adapter as adapter_mod
        from Auto3D.models.adapter import ANI2xtAdapter

        seen: list = []

        def _spy(atomic_numbers):
            seen.append(list(atomic_numbers))
            return ["sentinel"]

        monkeypatch.setattr(adapter_mod, "to_ani2xt_species", _spy)
        # Unbound call: no adapter instance, so no weights and no torchani.
        result = ANI2xtAdapter.to_species(object(), [6, 1, 1, 1, 1])

        assert seen == [[6, 1, 1, 1, 1]]
        assert result == ["sentinel"]

    def test_the_mapping_itself_is_what_ani2xt_expects(self):
        from Auto3D.models.species import to_ani2xt_species

        assert to_ani2xt_species([6, 1, 1, 1, 1]) == [1, 0, 0, 0, 0]
        assert to_ani2xt_species([1, 6, 7, 8, 9, 16, 17]) == [0, 1, 2, 3, 4, 5, 6]

    def test_out_of_set_element_names_the_element_and_the_model(self):
        from Auto3D.models.species import to_ani2xt_species

        with pytest.raises(ValueError) as exc:
            to_ani2xt_species([11])
        message = str(exc.value)
        assert "11" in message and "Na" in message and "ANI2xt" in message


def test_the_batch_opt_module_is_gone():
    """``batch_opt/species.py`` made the optimizer package a shared-utility host
    for ``ASE/``, ``cli/`` and ``models/``'s own padder. Clean sweep, no alias."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("Auto3D.batch_opt.species")
