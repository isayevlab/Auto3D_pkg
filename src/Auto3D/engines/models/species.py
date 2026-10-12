"""ANI2xt's species convention, owned by ``models/`` because it is the model's.

ANI2xt's species mapping is fixed to atomic-number order with no
periodic-table option, so its
forward expects 0-based network indices (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6), not
atomic numbers. Every other engine consumes atomic numbers unchanged.

The species
convention is a property of *the model*, so the model layer owns it, and the
only way to reach it is
:meth:`Auto3D.engines.models.contract.ModelAdapter.to_species` -- asking the object that
also supplies ``species_pad``, so the remap and the padding sentinel cannot come
from two sources and disagree.

Layering note: rdkit is imported lazily, inside the error branch that needs it
for an element symbol. This module is a pure lookup table plus a remap; it is
imported by the padder, by ``ASE/`` and by ``cli/``, none of which otherwise
need rdkit, and only the error path does. The table lives in ``engines/models/``
with no cross-layer back-reference; no import-cycle constraint currently
applies to it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# Atomic number -> ANI2xt network index. The order matches the ModuleList in
# models/ani2xt.py; changing one without the other misroutes elements.
ANI2XT_INDEX: dict[int, int] = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

__all__ = ["ANI2XT_INDEX", "format_elements", "to_ani2xt_species"]


def format_elements(atomic_numbers: Iterable[int]) -> str:
    """Render an element set as ``"H, C, N, O, F, S, Cl"``.

    Ordered by **atomic number**, which is the order every hand-written copy of
    every element string in this package already used -- so routing them through
    here changes no user-visible output. Sorting by symbol instead would silently
    rewrite all of them.

    One renderer because there were five hand-maintained copies of the ANI set
    across three layers: the numeric gate in :mod:`Auto3D.engines.models.policy`, the keys
    of :data:`ANI2XT_INDEX` below, the message in :func:`to_ani2xt_species`, and
    two entries in the CLI's ``ENGINE_INFO``. They agreed by hand, which is the
    same arrangement the engine registry replaced for engine *names*.

    Args:
        atomic_numbers: Atomic numbers, in any order and with any duplicates.

    Returns:
        Comma-separated element symbols, ascending by atomic number.
    """
    # Deferred exactly as in to_ani2xt_species below, and for the same reason:
    # this module is a lookup table imported by the padder, by ``ASE/`` and by
    # ``cli/``, none of which otherwise need rdkit. Only rendering does.
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    return ", ".join(table.GetElementSymbol(int(z)) for z in sorted(set(atomic_numbers)))


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
            # The supported set is rendered from the table this loop indexes,
            # not retyped beside it: the message and the check it explains
            # cannot disagree about which elements ANI2xt accepts.
            raise ValueError(
                f"Element Z={atomic_num} ({symbol}) is not supported by "
                f"ANI2xt (supported: {format_elements(ANI2XT_INDEX)})."
            ) from None
    return converted
