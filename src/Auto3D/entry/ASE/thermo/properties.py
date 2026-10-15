"""Molecular properties the thermochemistry needs, read off the molecule.

Geometry class, symmetry number, spin multiplicity, the display name and the
mass convention the record was built under -- all pure inspection of an RDKit
``Mol`` or an ASE ``Atoms``, with no model, no calculator and no ASE
thermochemistry involved. All but the last are the inputs ``IdealGasThermo`` is
constructed from; the mass convention annotates the record instead, naming what
``atoms`` actually carries rather than feeding ``IdealGasThermo`` itself.
"""

from __future__ import annotations

import ase
import ase.calculators.calculator
import numpy as np
from ase.data import atomic_masses
from rdkit import Chem

# The resolver lives in the domain layer so the optimizer can share it.
from Auto3D.domain.multiplicity import electron_count as _electron_count  # noqa: F401
from Auto3D.domain.multiplicity import record_name as _mol_name
from Auto3D.domain.multiplicity import resolve_multiplicity as _resolve_multiplicity  # noqa: F401
from Auto3D.foundation.constants import (
    LINEARITY_MAX_PERP_ANGSTROM,
    LINEARITY_MOMENT_RATIO,
)
from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


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

    This decides the rotational partition function only up to the Hessian
    check in ``project_vibrations``: for a bent stationary point inside the
    window that check always drops the phantom near-axis mode (3N-6 modes),
    but it hands ``IdealGasThermo`` the nonlinear rotor only when the smallest
    principal moment is above the classical-rotor floor -- below it the linear
    rotor is kept, which is the quantum limit for Theta_A >> T.
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
    unparseable, or outside 1..``_MAX_SYMMETRY_NUMBER`` -- since the bias cancels
    only between species that share a rotational symmetry number, which two
    conformers of one molecule need not do (chair sigma 6 against twist-boat
    sigma 4 is 0.24 kcal/mol), and which a diastereomer, tautomer, reaction
    partner or point-group-derived Gaussian/ORCA number generally does not.
    The defaulting-from-absence warning fires
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
            # A parseable but impossible value would otherwise be clamped by
            # `max(1, ...)` in silence, while every other invalid value in this
            # function warns: symmetry_number="0" and "-3" would both become
            # sigma=1 with nothing logged. And with no upper bound at all,
            # "1000000" would be accepted unchecked and shift Gibbs energy by
            # R*T*ln(1e6) = 8.2 kcal/mol at 298 K -- a silent 8 kcal/mol from one
            # mistyped property. _resolve_multiplicity two functions below
            # bounds and parity-checks its property the same way.
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
            "The bias cancels between conformers that share a rotational symmetry "
            "number (nearly all conformers of a flexible molecule are C1, sigma 1), "
            "but NOT between conformers of different symmetry (cyclohexane chair, "
            "sigma 6, against twist-boat, sigma 4: 0.24 kcal/mol), between "
            "diastereomers, tautomers or reaction partners, nor against Gaussian or "
            "ORCA, which infer sigma from the point group; enantiomers always share "
            "sigma. Set the 'symmetry_number' property (2 for water, 6 for ethane, "
            "12 for benzene) per record when comparing those. "
            "(Logged once per run; later molecules defaulting the same way are "
            "silent.)",
            _mol_name(mol),
        )
        _symmetry_default_warned = True
    return 1


MOST_ABUNDANT_MASSES = "most-abundant-isotope masses"
ISOTOPE_LABELED_MASSES = "isotope-labeled masses"
STANDARD_ATOMIC_WEIGHTS = "standard atomic weights"
CALLER_SUPPLIED_MASSES = "caller-supplied masses"
#: amu. The tolerance must separate the most-abundant-isotope table from the
#: standard-weight table where they differ: hydrogen's gap is 1.75e-4 (1.007825
#: against 1.008). For mononuclidic and near-mononuclidic elements the two
#: tables nearly coincide (tantalum differs by 4.2e-6, below this tolerance;
#: vanadium by 4.1e-5, manganese by 4.5e-5), so a molecule of such elements
#: built with standard weights reports the most-abundant token, which names the
#: same masses to within a few micro-amu.
_MASS_MATCH_ATOL = 1e-5


def mass_convention(mol: Chem.Mol, atoms: ase.Atoms) -> str:
    """Which masses the thermochemistry used, read from ``atoms`` itself.

    The token describes the ``Atoms`` object the Hessian, the moments of
    inertia and the translational term were built from, not what ``mol2atoms``
    would have produced: a caller-supplied ``Atoms`` carrying other masses is
    reported as what it is. Each stored mass is compared, within
    ``_MASS_MATCH_ATOL``, against the conventions in turn: the isotope labels
    the molecule carries (labeled atoms at their isotope's mass, every other
    atom the most abundant isotope), the most-abundant-isotope table, and the
    standard atomic weights, from RDKit's table or from ASE's
    ``atomic_masses`` (the table an ``Atoms`` built with ASE's defaults
    carries); masses matching none are named caller-supplied. A file mixing
    conventions says different things on different records, as it must,
    because their G are not comparable.
    """
    periodic_table = Chem.GetPeriodicTable()
    atoms_list = list(mol.GetAtoms())
    numbers = np.array([atom.GetAtomicNum() for atom in atoms_list])
    used = np.asarray(atoms.get_masses(), dtype=float)
    if used.shape != numbers.shape:
        return CALLER_SUPPLIED_MASSES
    most_abundant = np.array([periodic_table.GetMostCommonIsotopeMass(int(z)) for z in numbers])
    isotope_mask = np.array([atom.GetIsotope() != 0 for atom in atoms_list])
    label_masses = np.array([atom.GetMass() for atom in atoms_list])
    labeled = np.where(isotope_mask, label_masses, most_abundant)
    has_labels = bool(isotope_mask.any())
    standard_rdkit = np.array([periodic_table.GetAtomicWeight(int(z)) for z in numbers])
    standard_ase = atomic_masses[numbers]
    if has_labels and np.allclose(used, labeled, rtol=0.0, atol=_MASS_MATCH_ATOL):
        return ISOTOPE_LABELED_MASSES
    if np.allclose(used, most_abundant, rtol=0.0, atol=_MASS_MATCH_ATOL):
        return MOST_ABUNDANT_MASSES
    if np.allclose(used, standard_rdkit, rtol=0.0, atol=_MASS_MATCH_ATOL) or np.allclose(
        used, standard_ase, rtol=0.0, atol=_MASS_MATCH_ATOL
    ):
        return STANDARD_ATOMIC_WEIGHTS
    return CALLER_SUPPLIED_MASSES
