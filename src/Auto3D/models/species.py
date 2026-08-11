# src/Auto3D/models/species.py
"""ANI2xt's species convention, owned by ``models/`` because it is the model's.

ANI2xt is constructed with ``periodic_table_index=False`` at every site, so its
forward expects 0-based network indices (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6), not
atomic numbers. Every other engine consumes atomic numbers unchanged.

This table used to live in ``batch_opt/species.py``, which made ``batch_opt`` a
shared-utility host for ``ASE/``, ``cli/`` and ``models/``'s own padder -- three
layers with no business depending on the optimizer package. The species
convention is a property of *the model*, so the model layer owns it, and the
only way to reach it is
:meth:`Auto3D.models.contract.ModelAdapter.to_species` -- asking the object that
also supplies ``species_pad``, so the remap and the padding sentinel cannot come
from two sources and disagree (audit findings C3/C4).

A name-keyed ``to_model_species(atomic_numbers, model_name)`` lived here until
3.0.0 as a documented residual, for the two ``Auto3D.ASE.thermo`` callers that
held an engine *name* rather than a model. Both now hold an adapter, so it is
deleted: deciding the species convention from a string is what let the remap and
the padding sentinel come from different places, and the only way that cannot
recur is for the function not to exist.

Layering note: rdkit is imported lazily, inside the error branch that needs it
for an element symbol. This module is a pure lookup table plus a remap; it is
imported by the padder, by ``ASE/`` and by ``cli/``, none of which otherwise
need rdkit, and only the error path does. (The original reason was narrower and
no longer applies: ``models/`` used to be reachable from ``import Auto3D.utils``
via a module-scope import in ``utils/validation.py``, which audit M43 deferred
into the two functions that use it.)
"""

from __future__ import annotations

from collections.abc import Sequence

# Atomic number -> ANI2xt network index. The order matches the ModuleList in
# models/ani2xt.py; changing one without the other misroutes elements.
ANI2XT_INDEX: dict[int, int] = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

__all__ = ["ANI2XT_INDEX", "to_ani2xt_species"]


def to_ani2xt_species(atomic_numbers: Sequence[int]) -> list[int]:
    """Convert atomic numbers to ANI2xt's 0-based network indices.

    The single implementation of this remap, reached only through
    ``ANI2xtAdapter.to_species``, which delegates here.

    Args:
        atomic_numbers: Atomic numbers, one per atom.

    Returns:
        ANI2xt network indices in the same order.

    Raises:
        ValueError: An atomic number outside ANI2xt's supported set. The message
            names the atomic number, the element symbol, and the model.
    """
    converted: list[int] = []
    for atomic_num in atomic_numbers:
        try:
            converted.append(ANI2XT_INDEX[int(atomic_num)])
        except KeyError:
            # Deferred so the happy path never imports rdkit (see module
            # docstring); only the message needs the element symbol.
            from rdkit import Chem

            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num))
            raise ValueError(
                f"Element Z={atomic_num} ({symbol}) is not supported by "
                f"ANI2xt (supported: H, C, N, O, F, S, Cl)."
            ) from None
    return converted
