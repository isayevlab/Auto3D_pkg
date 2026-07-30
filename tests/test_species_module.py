"""The single owner of atomic-number -> model-species-index conversion.

ANI2xt is constructed with `periodic_table_index=False` everywhere, so it
expects 0-based indices (H=0..Cl=6), not atomic numbers. Before this module
existed the conversion was duplicated in three places and omitted in two more
(audit C3, C4).
"""
from __future__ import annotations

import pytest

from Auto3D.batch_opt.species import ANI2XT_INDEX, to_model_species


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
