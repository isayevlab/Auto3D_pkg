"""The spin state the energy engines and the thermochemistry use for a molecule:
an explicit `multiplicity` property when it is valid, otherwise the high-spin
value from the radical-electron count.

Both callers resolve it per record -- the optimizer when it pads a bucket, the
thermochemistry when it stamps a record -- so the value a spin-aware model is
evaluated at and the value the electronic entropy is built from are the same
number read the same way.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdmolops

from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.molprops import record_name

logger = get_logger(__name__)

__all__ = ["resolve_multiplicity", "electron_count", "drawn_closed_shell_but_open_shell"]

_OPEN_SHELL_DRAWN_CLOSED = ("O=O",)


def drawn_closed_shell_but_open_shell(mol: Chem.Mol) -> bool:
    """True for known species whose closed-shell drawing hides an open shell.

    Public because two callers need it: :func:`resolve_multiplicity` drives its
    own per-record warning from it, and the optimizer asks it about a whole
    bucket so a run over such species is flagged once rather than per record.
    The resolver still returns 1 for them -- nothing in the graph says
    otherwise -- so an explicit 'multiplicity' property is the only way to get
    the ground state.

    Caveat: singlet O2 (a real, if short-lived, excited state) is written with
    the identical closed-shell SMILES/graph as ground-state triplet O2, so
    this predicate cannot tell them apart -- the warning it drives may not
    apply if the input was actually meant to represent singlet O2.
    """
    try:
        canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except (ValueError, RuntimeError):
        return False
    return canonical in {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _OPEN_SHELL_DRAWN_CLOSED}


def electron_count(mol: Chem.Mol) -> int:
    """Total electron count: sum of atomic numbers minus the formal charge.

    Sums over ``Chem.AddHs(mol)`` rather than ``mol`` directly: a mol built
    without explicit hydrogens (e.g. straight from ``MolFromSmiles``, no
    ``AddHs`` call) stores them only as an implicit-H count on each heavy
    atom, not as their own ``Atom`` objects, so summing ``GetAtomicNum()``
    over ``mol.GetAtoms()`` would silently skip every implicit hydrogen and
    undercount electrons. ``Chem.AddHs`` returns a new mol (the input is not
    mutated) and is idempotent when hydrogens are already explicit, so this
    is correct either way.
    """
    return sum(a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms()) - rdmolops.GetFormalCharge(mol)


def resolve_multiplicity(mol: Chem.Mol, *, warn_open_shell: bool = True) -> int:
    """Spin multiplicity (2S+1): the spin state the energy engines and the thermochemistry use.

    The thermochemistry reads it as IdealGasThermo's electronic-degeneracy
    term; the optimizer and the single-point paths hand it to the model.

    ``warn_open_shell=False`` silences the three open-shell warnings (the
    high-spin assumption, the parity check on a derived value, the closed-shell
    drawing of an open-shell species), for a caller that resolves every
    conformer record of a run and reports once itself; the invalid-property
    warning always fires because it names a defect in that record.

    Resolving a record also CLEARS an invalid 'multiplicity' property from it,
    a side effect both callers now share. The optimizer clears the invalid
    property, warns once and writes the record without it, so a
    thermochemistry run on the optimizer's output uses the derived value
    silently; only a caller handing the same in-memory molecules to both paths
    sees the warning twice.

    Uses an explicit integer 'multiplicity' molecule property when present
    and valid. An invalid one (unparseable, out of bounds, or the wrong
    parity for the record's electron count) is dropped from ``mol`` with a
    warning rather than left in place, so the record never advertises a
    multiplicity that was not the one actually used. Otherwise derives it
    from the radical-electron count (multiplicity = unpaired electrons + 1),
    instead of silently assuming a closed-shell singlet -- which would zero
    the electronic entropy term for every radical. This assumes high-spin
    coupling of those radical electrons: an open-shell singlet (the same
    radical count, but paired into S = 0, e.g. some diradicals) needs an
    explicit 'multiplicity' property to override the high-spin default. The
    derived value is returned but not written back onto ``mol``: no energy
    engine reads the property, so recording it would only be a value a
    re-run recomputes anyway. A spin-aware AIMNet2 model receives this value;
    a model without a spin input ignores it and evaluates the molecule as a
    closed shell.

    The property is parsed with plain Python ``int()`` rather than RDKit's
    ``GetUnsignedProp``: the latter parses as an *unsigned* C++ integer, so a
    negative string like "-1" silently wraps around to 4294967295 (2**32 - 1)
    and "0" parses cleanly to 0 -- neither of those failure modes raises, so a
    try/except around ``GetUnsignedProp`` cannot catch them, and both then
    flow into IdealGasThermo's ``R*ln(multiplicity)`` electronic-entropy term
    as nonsense (spin = 2147483647.0 and -0.5, respectively). ``int()`` on the
    same string preserves the sign, so both are correctly rejected as
    multiplicities below the physically valid minimum of 1 (2S+1 for S >= 0).

    The lower bound alone is not sufficient: ``int("4294967295")`` parses
    cleanly (no wraparound -- that only afflicts ``GetUnsignedProp``) to a
    huge but nominally ">= 1" value, which passed the lower-bound check
    unchanged and fed spin = 2147483647.0 into ``R*ln(multiplicity)`` with no
    warning, shifting Gibbs energy by 13.1 kcal/mol at 298.15 K. A
    multiplicity is also bounded above: a molecule with ``n_electrons``
    electrons cannot exceed multiplicity ``n_electrons + 1`` (every electron
    unpaired), and 2S+1 must have parity opposite the electron count --
    integer S (odd multiplicity) for an even-electron species, half-integer S
    (even multiplicity) for an odd-electron one. Both the too-large and the
    wrong-parity cases are rejected the same way as the too-small case:
    warn and fall back to the radical-derived value.
    """
    if mol.HasProp("multiplicity"):
        try:
            value = int(mol.GetProp("multiplicity"))
        except (ValueError, TypeError):
            logger.warning(
                "Molecule %s has an unparseable 'multiplicity' property; "
                "deriving it from the radical-electron count instead and "
                "dropping the property from the output record.",
                record_name(mol),
            )
            mol.ClearProp("multiplicity")
        else:
            n_electrons = electron_count(mol)
            max_multiplicity = n_electrons + 1
            if value < 1:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "multiplicity must be >= 1 (2S+1 for spin S >= 0). "
                    "Deriving it from the radical-electron count instead and "
                    "dropping the property from the output record.",
                    record_name(mol),
                    value,
                )
                mol.ClearProp("multiplicity")
            elif value > max_multiplicity:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "a %d-electron species cannot exceed multiplicity %d "
                    "(2S+1 with every electron unpaired). Deriving it from "
                    "the radical-electron count instead and dropping the "
                    "property from the output record.",
                    record_name(mol),
                    value,
                    n_electrons,
                    max_multiplicity,
                )
                mol.ClearProp("multiplicity")
            elif value % 2 == n_electrons % 2:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "its parity is inconsistent with a %d-electron species "
                    "(2S+1 requires odd multiplicity for an even-electron "
                    "species, even multiplicity for an odd-electron one). "
                    "Deriving it from the radical-electron count instead and "
                    "dropping the property from the output record.",
                    record_name(mol),
                    value,
                    n_electrons,
                )
                mol.ClearProp("multiplicity")
            else:
                return value
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = n_radical + 1
    # The derived value gets the same parity check the supplied one gets.
    # Both the bounds and parity checks above sit inside the `HasProp` branch, so
    # a multiplicity Auto3D derived itself was returned unchecked -- and 2S+1
    # requires odd multiplicity for an even-electron species and even for an
    # odd-electron one, which the radical count can violate when the drawing is
    # wrong (a valence-satisfied structure hiding an open shell). Getting it wrong
    # is worth at least RT ln((m+1)/m) in T*S_elec -- 0.41 kcal/mol at 298 K
    # for a doublet taken for a singlet.
    #
    # Warn-only, and deliberately so: unlike the supplied-value branch, there is
    # no further fallback to take -- this IS the fallback. Silently substituting a
    # parity-consistent guess would replace a wrong number the user can see with a
    # wrong number they cannot.
    #
    # The electron count is computed inside the guard, not above it: it feeds
    # only this warning here (the supplied-value branch above has its own
    # call), and `electron_count` copies the molecule through `AddHs`, which a
    # caller resolving every conformer record of a run quietly should not pay
    # for a message it has silenced.
    if warn_open_shell:
        n_electrons = electron_count(mol)
        if multiplicity % 2 == n_electrons % 2:
            logger.warning(
                "Molecule %s: the multiplicity derived from its radical-electron "
                "count (%d) has a parity inconsistent with a %d-electron species "
                "(2S+1 requires odd multiplicity for an even-electron species, even "
                "for an odd-electron one). The drawing may hide an open shell. Set "
                "the 'multiplicity' property explicitly; the electronic entropy term "
                "is otherwise wrong by at least RT ln((m+1)/m) in T*S, "
                "0.41 kcal/mol at 298 K for a doublet taken for a singlet.",
                record_name(mol),
                multiplicity,
                n_electrons,
            )
    if n_radical > 0:
        if warn_open_shell:
            logger.warning(
                "Molecule %s: open-shell species (%d unpaired electron(s)); "
                "resolved to multiplicity %d, the high-spin value. Set the "
                "'multiplicity' property to override it.",
                record_name(mol),
                n_radical,
                multiplicity,
            )
    elif warn_open_shell and drawn_closed_shell_but_open_shell(mol):
        # O=O draws as a closed-shell double bond and carries zero radical
        # electrons, but its ground state is a triplet. Nothing in the graph
        # distinguishes it, so the electronic entropy term is silently wrong
        # unless the caller sets 'multiplicity' explicitly.
        logger.warning(
            "%s matches a species whose ground state is open-shell but whose "
            "drawing is closed-shell; multiplicity 1 is assumed, a spin-aware "
            "model evaluates the singlet state and the electronic entropy term "
            "will be wrong. Set the 'multiplicity' property explicitly.",
            record_name(mol),
        )
    return multiplicity
