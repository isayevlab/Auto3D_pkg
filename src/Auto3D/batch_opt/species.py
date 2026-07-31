"""Atomic-number to model-species-index conversion.

This module is the single owner of that mapping. ANI2xt is constructed with
``periodic_table_index=False`` at every site, so its forward expects 0-based
network indices (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6), not atomic numbers.
Every other engine consumes atomic numbers unchanged.

Before 4.0 this mapping was duplicated in ``utils/chemistry.py`` and
``ASE/thermo.py`` and omitted entirely from the thermo entry points and the
CLI health check, so ANI2xt silently evaluated the wrong species there
(audit findings C3 and C4).
"""
from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem

from Auto3D.constants import MODEL_ANI2XT

# Atomic number -> ANI2xt network index. The order matches the ModuleList in
# batch_opt/ANI2xt_no_rep.py; changing one without the other misroutes elements.
ANI2XT_INDEX: dict[int, int] = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

__all__ = ["ANI2XT_INDEX", "to_model_species"]


def to_model_species(atomic_numbers: Sequence[int], model_name: str) -> list[int]:
    """Convert atomic numbers to the species values a model expects.

    Args:
        atomic_numbers: Atomic numbers, one per atom.
        model_name: Engine name, matched case-insensitively against
            ``"ANI2xt"`` (the same convention ``ModelFactory.create_model``
            uses). Only a match remaps; every other value (AIMNET, any aimnet
            registry name, ANI2x, a custom model path) is passed through
            unchanged.

    Returns:
        Species values in the model's own convention.

    Raises:
        ValueError: If ``model_name`` is ``"ANI2xt"`` and an atomic number is
            outside its supported set. The message names the atomic number,
            the element symbol, and the model.
    """
    # Case-insensitive match, matching ModelFactory.create_model's own
    # normalization (model_factory.py: name.upper() in cls._adapters). Without
    # this, "auto3d models test ani2xt" loads the correct ANI2xt adapter via
    # create_model but this function silently passes atomic numbers through
    # unconverted -- a C4-shaped bug hiding behind incidental case-matching.
    if model_name.upper() != MODEL_ANI2XT.upper():
        return list(atomic_numbers)

    converted: list[int] = []
    for atomic_num in atomic_numbers:
        try:
            converted.append(ANI2XT_INDEX[atomic_num])
        except KeyError:
            symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            raise ValueError(
                f"Element Z={atomic_num} ({symbol}) is not supported by "
                f"ANI2xt (supported: H, C, N, O, F, S, Cl)."
            ) from None
    return converted
