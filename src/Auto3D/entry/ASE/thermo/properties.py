"""Molecular properties the thermochemistry needs, read off the molecule.

Geometry class, symmetry number, spin multiplicity and the display name -- all
pure inspection of an RDKit ``Mol`` or an ASE ``Atoms``, with no model, no
calculator and no ASE thermochemistry involved. They are the inputs
``IdealGasThermo`` is constructed from.
"""

from __future__ import annotations

import ase
import ase.calculators.calculator
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

from Auto3D.foundation.constants import (
    LINEARITY_MAX_PERP_ANGSTROM,
    LINEARITY_MOMENT_RATIO,
)
from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


def _mol_name(mol: Chem.Mol, default: str = "molecule") -> str:
    """The molecule's ``_Name`` property, or ``default`` when it has none.

    Every diagnostic/warning site in this module needs a human-readable
    identifier for a mol that may or may not carry ``_Name``, and used to
    repeat ``mol.GetProp("_Name") if mol.HasProp("_Name") else <default>``
    verbatim at each site (M64). The default itself is NOT hardcoded here to
    a single value: most callers want the generic ``"molecule"``, but
    ``iter_thermo_records`` identifies an unnamed record by its position in
    the file (``f"record {position}"``) instead, so that case is still
    threaded through explicitly.
    """
    return mol.GetProp("_Name") if mol.HasProp("_Name") else default


def _is_collinear(atoms: ase.Atoms) -> bool:
    """True if all atoms lie on a single line.

    Decided by the principal moments of inertia rather than by a rank test on
    raw coordinates. A rank tolerance is an absolute length in Angstrom, so it
    calls a CO2 bent by more than ~1e-3 A nonlinear -- inventing a third
    rotational degree of freedom and discarding a real 667 cm-1 bend, worth
    ~0.95 kcal/mol of zero-point energy before its thermal contribution. The
    moment ratio is dimensionless and scales with the molecule, so it behaves
    the same for a diatomic and for a long polyyne.

    A linear molecule has one vanishing principal moment, so the first test is
    that the smallest moment is negligible against the largest. That test
    alone is not sufficient: the largest moment grows as N^2 (mass x
    length^2, summed over atoms further and further from the center), so for
    a long chain the same absolute bend shrinks the ratio as the molecule gets
    longer -- the ratio becomes a size cutoff, not a shape test. 2,4,6-
    octatriyne (CC#CC#CC#CC) is the case this misses: ratio 5.7e-3, below the
    1e-2 threshold, with every atom sitting 1.02 A off the molecular axis --
    visibly bent, not linear. The second, load-bearing test is therefore an
    absolute one: no atom may sit more than LINEARITY_MAX_PERP_ANGSTROM from
    the principal axis (the eigenvector of the smallest moment), measured
    from the center of mass. A molecule is linear only when both tests agree;
    see LINEARITY_MOMENT_RATIO and LINEARITY_MAX_PERP_ANGSTROM in constants.py
    for the measurements that placed each threshold.
    """
    if len(atoms) <= 2:
        return True
    moments, axes = atoms.get_moments_of_inertia(vectors=True)
    largest = float(np.max(moments))
    if largest <= 0.0:
        # All atoms coincident; degenerate but not meaningfully nonlinear.
        return True
    smallest_idx = int(np.argmin(moments))
    ratio_ok = float(moments[smallest_idx]) / largest < LINEARITY_MOMENT_RATIO

    # axes[i] is the eigenvector belonging to moments[i] (ASE returns the
    # eigenvectors transposed, one full axis per row -- see
    # Atoms.get_moments_of_inertia). The smallest-moment axis is the
    # molecule's long axis for a rod-like structure.
    axis = axes[smallest_idx]
    axis = axis / np.linalg.norm(axis)
    offsets = atoms.get_positions() - atoms.get_center_of_mass()
    perpendicular = offsets - np.outer(offsets @ axis, axis)
    max_perp = float(np.max(np.linalg.norm(perpendicular, axis=1)))
    perp_ok = max_perp < LINEARITY_MAX_PERP_ANGSTROM

    return bool(ratio_ok and perp_ok)


def _detect_geometry(atoms: ase.Atoms) -> str:
    """Classify molecular geometry for IdealGasThermo.

    Returns one of 'monatomic', 'linear', 'nonlinear'.
    """
    n = len(atoms)
    if n == 1:
        return "monatomic"
    if _is_collinear(atoms):
        return "linear"
    return "nonlinear"


_symmetry_default_warned = False
_MAX_SYMMETRY_NUMBER = 60


def _symmetry_number(mol: Chem.Mol) -> int:
    """External rotational symmetry number for IdealGasThermo.

    Read from an optional integer 'symmetry_number' molecule property; defaults
    to 1 when absent. We intentionally do NOT auto-derive sigma from the
    molecular graph: graph automorphisms count internal-rotor and H-permutation
    symmetries that are not part of the external rotational symmetry number, and
    overcount sigma by large factors for flexible molecules (e.g. ethane 12x,
    cyclohexane 128x), biasing Gibbs energy by up to ~3 kcal/mol. sigma=1 is a
    safe default; set the 'symmetry_number' property to the correct value
    (e.g. 2 for water, 12 for benzene, 6 for ethane) when known.

    Defaulting to sigma=1 warns, whatever the reason -- the property is absent,
    unparseable, or outside 1..``_MAX_SYMMETRY_NUMBER`` -- since the bias does not
    cancel between tautomers, isomers or reaction partners the way it does
    between conformers of one species. The defaulting-from-absence warning fires
    once per calc_thermo run, not once per molecule, since every molecule lacking
    the property triggers the identical message; the two invalid-value warnings
    name the offending value and so fire per molecule.
    """
    global _symmetry_default_warned
    if mol.HasProp("symmetry_number"):
        try:
            value = int(mol.GetProp("symmetry_number"))
        except (ValueError, TypeError):
            logger.warning(
                "Molecule %s has an unparseable 'symmetry_number' property "
                "(%r); falling back to sigma=1.",
                _mol_name(mol),
                mol.GetProp("symmetry_number"),
            )
            return 1
        else:
            # A parseable but impossible value used to be clamped by
            # `max(1, ...)` in silence, while every other invalid value in this
            # function warns: symmetry_number="0" and "-3" both became sigma=1
            # with nothing logged. And there was no upper bound at all, so
            # "1000000" was accepted unchecked and shifted Gibbs energy by
            # R*T*ln(1e6) = 8.2 kcal/mol at 298 K -- a silent 8 kcal/mol from one
            # mistyped property. _resolve_multiplicity two functions below already
            # bounds and parity-checks its property; this one did neither.
            #
            # The upper bound is the highest external rotational symmetry number
            # of any real molecule: 60 for the icosahedral point groups (I, Ih --
            # C60, B12H12(2-)). Anything above that is a typo, not a molecule.
            if value < 1 or value > _MAX_SYMMETRY_NUMBER:
                logger.warning(
                    "Molecule %s has an invalid 'symmetry_number' property "
                    "(%d); it must be between 1 and %d (the largest external "
                    "rotational symmetry number of any real molecule, for the "
                    "icosahedral point groups). Falling back to sigma=1.",
                    _mol_name(mol),
                    value,
                    _MAX_SYMMETRY_NUMBER,
                )
                return 1
            return value
    if not _symmetry_default_warned:
        logger.warning(
            "No 'symmetry_number' property on %s; using sigma=1. Gibbs energy is "
            "biased low by RT*ln(sigma) -- 1.47 kcal/mol for benzene at 298 K. "
            "This cancels between conformers of one species but NOT between "
            "tautomers, isomers or reaction partners. Set the 'symmetry_number' "
            "property (2 for water, 6 for ethane, 12 for benzene) when known. "
            "(Logged once per run; later molecules defaulting the same way are "
            "silent.)",
            _mol_name(mol),
        )
        _symmetry_default_warned = True
    return 1


_OPEN_SHELL_DRAWN_CLOSED = ("O=O",)


def _drawn_closed_shell_but_open_shell(mol: Chem.Mol) -> bool:
    """True for known species whose closed-shell drawing hides an open shell.

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


def _electron_count(mol: Chem.Mol) -> int:
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


def _resolve_multiplicity(mol: Chem.Mol) -> int:
    """Spin multiplicity (2S+1) for IdealGasThermo's electronic-degeneracy term.

    Uses an explicit integer 'multiplicity' molecule property when present.
    Otherwise derives it from the radical-electron count
    (multiplicity = unpaired electrons + 1) and records it on the mol, instead of
    silently assuming a closed-shell singlet -- which would zero the electronic
    entropy term for every radical. The NNP *energy* stays closed-shell
    regardless (AIMNet2 takes only coords/species/charge, no spin), so warn for
    open-shell species that the energy is an approximation.

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
                "deriving it from the radical-electron count instead.",
                _mol_name(mol),
            )
        else:
            n_electrons = _electron_count(mol)
            max_multiplicity = n_electrons + 1
            if value < 1:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "multiplicity must be >= 1 (2S+1 for spin S >= 0). "
                    "Deriving it from the radical-electron count instead.",
                    _mol_name(mol),
                    value,
                )
            elif value > max_multiplicity:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "a %d-electron species cannot exceed multiplicity %d "
                    "(2S+1 with every electron unpaired). Deriving it from "
                    "the radical-electron count instead.",
                    _mol_name(mol),
                    value,
                    n_electrons,
                    max_multiplicity,
                )
            elif value % 2 == n_electrons % 2:
                logger.warning(
                    "Molecule %s has an invalid 'multiplicity' property (%d); "
                    "its parity is inconsistent with a %d-electron species "
                    "(2S+1 requires odd multiplicity for an even-electron "
                    "species, even multiplicity for an odd-electron one). "
                    "Deriving it from the radical-electron count instead.",
                    _mol_name(mol),
                    value,
                    n_electrons,
                )
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
    # is worth R*ln 3 = 0.65 kcal/mol in T*S_elec.
    #
    # Warn-only, and deliberately so: unlike the supplied-value branch, there is
    # no further fallback to take -- this IS the fallback. Silently substituting a
    # parity-consistent guess would replace a wrong number the user can see with a
    # wrong number they cannot.
    n_electrons = _electron_count(mol)
    if multiplicity % 2 == n_electrons % 2:
        logger.warning(
            "Molecule %s: the multiplicity derived from its radical-electron "
            "count (%d) has a parity inconsistent with a %d-electron species "
            "(2S+1 requires odd multiplicity for an even-electron species, even "
            "for an odd-electron one). The drawing may hide an open shell. Set "
            "the 'multiplicity' property explicitly; the electronic entropy term "
            "is otherwise wrong by up to R*ln(3) = 0.65 kcal/mol in T*S.",
            _mol_name(mol),
            multiplicity,
            n_electrons,
        )
    mol.SetUnsignedProp("multiplicity", int(multiplicity))
    if n_radical > 0:
        logger.warning(
            "Open-shell species detected (%d unpaired electron(s), "
            "multiplicity %d); the NNP energy is a closed-shell approximation.",
            n_radical,
            multiplicity,
        )
    elif _drawn_closed_shell_but_open_shell(mol):
        # O=O draws as a closed-shell double bond and carries zero radical
        # electrons, but its ground state is a triplet. Nothing in the graph
        # distinguishes it, so the electronic entropy term is silently wrong
        # unless the caller sets 'multiplicity' explicitly.
        logger.warning(
            "%s matches a species whose ground state is open-shell but whose "
            "drawing is closed-shell; multiplicity 1 is assumed and the "
            "electronic entropy term will be wrong. Set the 'multiplicity' "
            "property explicitly.",
            _mol_name(mol),
        )
    return multiplicity
