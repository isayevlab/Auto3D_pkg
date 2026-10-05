"""Hessians, mode projection, and what comes out of them.

The Hessian is obtained from the adapter -- analytically where the model has a
native second derivative, by autograd otherwise -- and projected onto the
internal degrees of freedom before any frequency is reported.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import ase
import ase.calculators.calculator
import numpy as np
import torch
from ase import units as ase_units
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import VibrationsData
from rdkit import Chem
from rdkit.Chem import rdmolops

from Auto3D.engines.models.contract import ModelAdapter
from Auto3D.entry.ASE.thermo.calculator import mol2atoms
from Auto3D.foundation.constants import (
    EV_PER_WAVENUMBER,
    IMAGINARY_MODE_CUTOFF_CM,
    LINEAR_AXIS_ROTATION_GATE,
    LINEARITY_MARGINAL_RATIO,
    LINEARITY_MAX_PERP_ANGSTROM,
    LOW_FREQUENCY_CUTOFF_CM,
    PROJECTION_RESIDUAL_FRACTION,
)
from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


def vib_hessian(
    mol: Chem.Mol,
    ase_calculator,
    adapter: ModelAdapter,
    device=torch.device("cpu"),
    *,
    positions=None,
):
    """Return a VibrationsData object for one molecule.

    The Hessian source is a CAPABILITY of the adapter, not a type test on it and
    not a branch on an engine name. ``adapter.analytic_hessian(...)`` returns the
    model's own second derivative, or ``None`` meaning "differentiate
    ``adapter.energy``". Only AIMNet2 answers with a Hessian, and it must,
    because its native one runs the FULL energy pipeline including the external
    D3 dispersion and Coulomb modules: differentiating the bare aimnet
    ``nn.Module`` drops those terms (D3 is attractive at bonding range),
    stiffening every bond and shifting C-H stretches up by ~4% (~130 cm-1).
    ANI2xt / ANI2x / userNNP are plain modules with the whole energy in the
    graph, so autograd of ``energy`` is exact for them.

    This replaced ``isinstance(model, AIMNet2Calculator)`` plus a five-branch,
    name-keyed ``aimnet_hessian_helper``. Two consequences worth naming: an
    aimnet registry alias (``aimnet2-2025``, ``aimnet2-nse``, ...) no longer has
    a branch to fall through, and ``energy`` -- not ``forward`` -- is what gets
    differentiated, so the fp64 geometry is not silently answered in fp32 by the
    two adapters whose ``forward`` calls ``coords.float()``.

    Args:
        mol: RDKit molecule; supplies species, formal charge and isotope masses.
        ase_calculator: Attached to the ``Atoms`` object for downstream ASE use.
        adapter: The model, satisfying
            :class:`Auto3D.engines.models.contract.ModelAdapter`. Supplies the species
            convention, the energy and the Hessian capability.
        device: Device the Hessian is built on.
        positions: Geometry to build the Hessian from. Defaults to the mol's
            conformer, which is only correct when no relaxation has happened
            since the conformer was last synced. The caller (do_mol_thermo)
            passes the relaxed geometry explicitly here in addition to
            syncing mol's conformer beforehand, so the Hessian is guaranteed
            to describe the same structure as the energy regardless of sync
            order."""
    # Built through mol2atoms (not a bare Atoms(species, coord) call) so
    # isotope masses are applied here exactly as they are for the other two
    # Atoms constructions (mol2atoms's own default path, calc_thermo's
    # optimization loop) -- otherwise the moments of inertia, VibrationsData's
    # mass weighting, and the rotational partition function silently disagree
    # for isotopically labeled input.
    atoms = mol2atoms(mol, positions=positions)
    # atoms.set_calculator() is deprecated since ase 3.22.1 in favor of the
    # `.calc` attribute (Minor 6); `pyproject.toml` pins no ase upper bound
    # and globally ignores DeprecationWarning, so removal would otherwise
    # land as a silent-until-runtime AttributeError with no advance warning.
    atoms.calc = ase_calculator
    charge = rdmolops.GetFormalCharge(mol)

    # get the Hessian
    coord = torch.tensor(atoms.get_positions()).to(device).unsqueeze(0)
    num_atoms = coord.shape[1]
    # The species convention comes from the adapter, so ANI2xt's 0-based network
    # indices are built here exactly as they are for the optimization batch. This
    # used to be raw atomic numbers, remapped (or not) further down inside a
    # name-keyed helper -- one more place the convention could disagree.
    numbers = torch.tensor([adapter.to_species([a.GetAtomicNum() for a in mol.GetAtoms()])]).to(
        device
    )
    # aimnet's AIMNet2 model requires a 1D charge tensor (one entry per
    # molecule); a 0-dim scalar trips an internal assert.
    #
    # float32, explicitly: `torch.tensor([charge])` on a Python int gives int64,
    # and while most routes recover by casting on arrival,
    # `AIMNet2Adapter.analytic_hessian` deliberately does not -- it passes the
    # charge to the calculator exactly as received. So the *default* engine's
    # analytic Hessian was the one place still receiving an int64 charge after
    # the optimizer path was fixed, leaving CLAUDE.md's "charges reach a model
    # as float32" false on this path.
    charge = torch.tensor([charge], dtype=torch.float32, device=device)

    hess = adapter.analytic_hessian(coord, numbers, charge)
    if hess is None:
        # No native second derivative: differentiate the adapter's energy. Note
        # `energy`, not `forward` -- forward would compute (and discard) forces,
        # and in two adapters would downcast this fp64 geometry to fp32.
        hess = torch.autograd.functional.hessian(
            lambda xyz: adapter.energy(xyz, numbers, charge), coord
        )
    hess = hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()

    # get the VibrationsData object
    vib = VibrationsData(atoms, hess)
    return vib


_EXTERNAL_DOF = {"monatomic": 3, "linear": 5, "nonlinear": 6}
_HESSIAN_ENERGY_CONVERSION = ase_units._hbar * ase_units.m / (ase_units._e * ase_units._amu) ** 0.5
_ASE_HAS_VIB_SELECTION = "vib_selection" in inspect.signature(IdealGasThermo.__init__).parameters


def n_vibrational_modes(n_atoms: int, geometry: str) -> int:
    """Number of genuine vibrational modes: ``3N-6``, ``3N-5``, or 0.

    Args:
        n_atoms: Number of atoms.
        geometry: 'monatomic', 'linear' or 'nonlinear', as classified by
            ``_detect_geometry``.
    """
    try:
        external = _EXTERNAL_DOF[geometry]
    except KeyError:
        raise ValueError(
            f"Unsupported geometry {geometry!r}; expected one of {sorted(_EXTERNAL_DOF)}."
        ) from None
    return max(0, 3 * n_atoms - external)


def _external_mode_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Mass-weighted translation and infinitesimal-rotation vectors, ``3N x 6``.

    Column ``a`` of the first three is the rigid translation along axis ``a``,
    ``T_a[3i+a] = sqrt(m_i)``; column ``a`` of the last three is the
    infinitesimal rotation about axis ``a`` through the center of mass,
    ``R_a[3i:3i+3] = sqrt(m_i) * (e_a x (r_i - r_cm))``. These are the
    Sayvetz/Eckart conditions written as vectors in mass-weighted Cartesian
    space: an exact Hessian at a stationary point annihilates all six.

    For a linear molecule the rotation about the molecular axis is identically
    zero, so the six columns span only five dimensions; the caller keeps the
    leading ``_EXTERNAL_DOF[geometry]`` left singular vectors, which is why the
    count comes from the geometry rather than from the rank of this matrix.
    """
    sqrt_m = np.sqrt(masses)
    center = (masses[:, np.newaxis] * positions).sum(axis=0) / masses.sum()
    offsets = positions - center
    columns = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = sqrt_m
        columns.append(translation.reshape(-1))
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        rotation = np.cross(unit, offsets) * sqrt_m[:, np.newaxis]
        columns.append(rotation.reshape(-1))
    return np.column_stack(columns)


#: ``Thermo_linearity`` for a molecule the linearity window called linear and
#: the Hessian overruled, where the classical nonlinear rotor is still valid:
#: the thermochemistry ran with 3N-6 modes AND the nonlinear rotor.
BENT_RECLASSIFIED = "bent_reclassified_nonlinear"

#: ``Thermo_linearity`` for the same verdict below the classical-rotor floor:
#: the phantom is dropped (3N-6 modes) but the LINEAR rotor is kept, because
#: that is the quantum limit when ``Theta_A >> T``. See
#: ``_CLASSICAL_ROTOR_FLOOR_AMU_A2_K``.
BENT_QUASILINEAR = "bent_quasilinear_linear_rotor"

#: Smallest principal moment, in amu A^2 K, at which the classical rigid-rotor
#: partition function for the near-axis rotation is still valid; divide by the
#: temperature to get a moment in amu A^2 (0.0259 at 298.15 K). This is
#: ``h^2 / (8 pi^3 k)``, the moment at which ``pi T = Theta_A`` with
#: ``Theta_A = h^2 / (8 pi^2 I k)``.
#:
#: Why it bounds the reclassification. ASE's nonlinear rotational entropy is the
#: classical ``R/2 ln(pi T^3 / (Theta_A Theta_B Theta_C))``, which needs
#: ``Theta_A << T``. Below ``floor / T`` the classical ``q_A = sqrt(pi T /
#: Theta_A)`` drops under 1, i.e. under the quantum ground state, and
#: ``dG_rot = -RT/2 ln(pi T / Theta_A)`` grows without bound: for CO2 at 298 K,
#: +1.0 kcal/mol at 179 degrees and +2.4 at 179.9 -- larger than the phantom
#: this module removes. At the floor ``dG_rot = 0``, so switching rotors there
#: leaves G continuous. The quantum limit for ``Theta_A >> T`` is that only
#: ``K = 0`` is populated and ``q_rot(nonlinear) -> q_rot(linear)``, which is
#: why the linear rotor is the right answer below the floor.
_CLASSICAL_ROTOR_FLOOR_AMU_A2_K = ase_units._hplanck**2 / (
    8 * np.pi**3 * ase_units._k * ase_units._amu * 1e-20
)


@dataclass(frozen=True)
class Projection:
    """What ``project_vibrations`` decided and produced.

    Attributes:
        energies: the ``3N-6`` / ``3N-5`` / ``[]`` complex energies in eV, ascending.
        geometry: the ROTOR geometry -- what ``IdealGasThermo`` must be given, so
            that its rotational partition function describes the molecule the
            energies came from. The input value, except that a bent stationary
            point inside the linearity window whose smallest principal moment is
            above the classical-rotor floor comes back as ``"nonlinear"``.
        mode_geometry: the geometry whose external-mode count was projected out,
            i.e. what ``len(energies)`` is ``3N - _EXTERNAL_DOF[...]`` of. This is
            what ``analyze_vibrations`` must be given. It equals ``geometry``
            except in the quasilinear case, where ``3N-6`` modes are paired with
            the linear rotor on purpose (see ``BENT_QUASILINEAR``).
        linearity: the verdict written to ``Thermo_linearity``: ``geometry``
            itself, or ``BENT_RECLASSIFIED`` / ``BENT_QUASILINEAR`` when the input
            said linear and the Hessian said bent.
    """

    energies: list[complex]
    geometry: str
    mode_geometry: str
    linearity: str


def _projected_spectrum(
    mass_weighted: np.ndarray, external: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(kept, discarded)`` eigenvalues of ``P H P`` with ``P = I - V V^T``.

    ``external`` holds ``n_external`` orthonormal columns; the ``n_external``
    smallest-magnitude eigenvalues are the projected-out null space (machine
    zero by construction) and come back as ``discarded``; the rest are the
    vibrations, sorted ascending.
    """
    n_external = external.shape[1]
    projector = np.eye(mass_weighted.shape[0]) - external @ external.T
    eigenvalues = np.linalg.eigvalsh(projector @ mass_weighted @ projector)
    by_magnitude = np.argsort(np.abs(eigenvalues))
    return np.sort(eigenvalues[by_magnitude[n_external:]]), eigenvalues[by_magnitude[:n_external]]


def project_vibrations(
    atoms: ase.Atoms,
    hessian,
    geometry: str,
    *,
    name: str = "molecule",
    temperature_k: float = 298.15,
) -> Projection:
    """Vibrational energies with translation and rotation projected out.

    Returns exactly ``n_vibrational_modes(len(atoms), Projection.mode_geometry)``
    complex energies in eV, ascending in eigenvalue (ASE's own ordering), with
    a negative curvature represented as a purely imaginary energy ``0 + b*i``.
    ``Projection.mode_geometry`` is the ``geometry`` argument except in the one
    case described below, where the Hessian overrules it; ``Projection.geometry``
    is the rotor geometry, which can differ from both (see ``Projection``).

    **Why this exists.** ``VibrationsData.get_energies()`` diagonalizes the raw
    mass-weighted Hessian and returns all ``3N`` eigenvalues; six of them (five
    for a linear molecule) are the translations and rotations, which are exact
    zero modes only at a stationary point in exact arithmetic and in practice
    land at small positive or negative values. Auto3D used to hand that full
    ``3N`` list to ``IdealGasThermo`` and let ASE decide which entries were
    vibrations. That is not a stable interface: ASE 3.23.0-3.27.x sort the list
    by ``np.abs`` and keep the last ``3N-6``; ASE 3.28.0 and later sort by
    ``(f**2).real`` and keep the last ``3N-6`` (``vib_selection='highest'``,
    the default). The two rules disagree whenever any vibrational mode is
    imaginary: under the ``(f**2).real`` key every imaginary mode sorts *below*
    every real one, so the selection discards it and promotes a
    translation/rotation noise mode into the vibrational partition function in
    its place -- worth several kcal/mol of G, silently, with no change to the
    reported mode count.

    Neither rule can be repaired, because both throw away the information
    needed to answer the question. Only the caller has the geometry and the
    eigenvectors; once the eigenvalues are flattened into a list of complex
    numbers, "is this a rotation" is unanswerable except by magnitude, and
    magnitude is exactly the assumption that fails off a stationary point.

    **What this does instead** is the standard vibrational analysis used by
    production quantum-chemistry codes (Gaussian, ORCA; Miller, Handy and
    Adams, J. Chem. Phys. 1980, 72, 99): mass-weight the Hessian, build the
    translation and infinitesimal-rotation vectors, orthonormalize them to
    ``V``, and diagonalize ``P H P`` with ``P = I - V V^T``. The projected-out
    subspace is then a null space *by construction* -- there is no threshold,
    no sorting and no tie-breaking -- and the remaining eigenvalues are the
    vibrations. Where the magnitude heuristic works, this agrees with it
    exactly (measured on MMFF n-butane and n-butanol at a tight stationary
    point: identical to 0.00 cm-1); where it does not, this is the only
    correct answer.

    **A bent stationary point inside the linearity window** (N-M4). ``_detect_geometry``
    calls a molecule linear while every atom sits within ``LINEARITY_MAX_PERP_ANGSTROM``
    of the principal axis, because an optimizer leaves a linear molecule a few degrees
    off axis. But calc_thermo reaches this function only past the stationary-point
    gate, where any residual bend is the model's minimum, and a bent minimum has SIX
    null directions, not five: the sixth, the rotation about the near-axis, then
    survives the five-vector projection as a near-zero "vibration" that the
    quasi-harmonic floor raises to ``LOW_FREQUENCY_CUTOFF_CM`` while the rotational
    term stays linear.

    The Hessian tells the two cases apart where the coordinates cannot. Along that
    sixth direction a linear molecule's curvature is the bend force constant,
    independent of how far off axis the geometry sits (measured on MMFF CO2 displaced
    1e-5 to 0.1 A: 484.0 to 486.7 cm-1 equivalent, ratio 1.000 to 1.021). For a bent
    stationary point the exact identity is

        R_n^T H R_n = sum_i o_perp,i . g_perp,i

    -- perpendicular offsets from the center of mass against the gradient -- so it is
    zero only at ``g = 0`` exactly, and at the 2e-4 eV/A force gate it is bounded by
    ``q_max = fmax * sum|o_perp| / I_n``: 9 cm-1 equivalent for CO2 at 170 degrees,
    28 cm-1 at 179. (The ~0.05 cm-1 figure quoted elsewhere is the synthetic test
    fixture's exact-stationary-point value, not a gate-level residual.)

    When the geometry says linear, N > 2, and the sixth direction is resolved
    (``LINEAR_AXIS_ROTATION_GATE``), its Rayleigh quotient is compared with the
    smallest remaining mode of the would-be nonlinear projection. Both are EIGENVALUES
    of the mass-weighted Hessian, in eV/(A^2 amu) and proportional to omega^2, so
    ``PROJECTION_RESIDUAL_FRACTION = 0.05`` is a frequency ratio of
    ``sqrt(0.05) = 0.2236``, i.e. a 4.47x split. Because the six-vector projection
    removes exactly the near-axis direction, the comparison is bend partner against
    bend partner: a molecule is reclassified only when its near-axis (out-of-plane)
    bend partner is more than 4.47x softer in frequency than the smallest remaining
    mode at a converged geometry. A genuinely linear molecule's pair is degenerate by
    symmetry, so its ratio is at or above 1 whatever the geometric residual. Between
    ``PROJECTION_RESIDUAL_FRACTION`` and ``LINEARITY_MARGINAL_RATIO`` the verdict is
    linear but a warning says the test was marginal, since that band is where
    ``q_max`` above can hide a real bent minimum.

    The comparison is against the NONLINEAR projection on purpose: in the bent case the
    phantom is itself the smallest kept eigenvalue of the linear projection, so a
    comparison against that set can never fire.

    **The rotor is bounded separately** (R27). Dropping the phantom and switching to
    the nonlinear rotor are two decisions, and only the first is always right. ASE's
    nonlinear rotational entropy is the classical
    ``R/2 ln(pi T^3 / (Theta_A Theta_B Theta_C))``, which needs ``Theta_A << T``; a
    reclassified molecule's smallest principal moment can be arbitrarily small, which
    is the ``sqrt(I1 I2 I3)`` divergence ``LINEARITY_MAX_PERP_ANGSTROM`` exists to
    avoid. So the nonlinear rotor is handed to ``IdealGasThermo`` only when
    ``I_min >= _CLASSICAL_ROTOR_FLOOR_AMU_A2_K / temperature_k``
    (``BENT_RECLASSIFIED``); below it the phantom is still dropped but the linear rotor
    is kept (``BENT_QUASILINEAR``), which is the quantum limit for ``Theta_A >> T``.
    ``I_min`` is read off the SVD as ``singular[5] ** 2``, which equals
    ``min(atoms.get_moments_of_inertia())`` to machine precision.

    Args:
        atoms: The atoms the Hessian describes. Supplies both the masses (so
            an isotopic label set by ``mol2atoms`` weights the Hessian the same
            way it weights the moments of inertia) and the positions used to
            build the rotation vectors.
        hessian: The Cartesian Hessian in eV/A^2, shaped ``(3N, 3N)`` or
            ``(N, 3, N, 3)``. This is the unit
            ``ase.vibrations.VibrationsData`` expects and the unit both Hessian
            paths in ``vib_hessian`` produce.
        geometry: 'monatomic', 'linear' or 'nonlinear', from
            ``_detect_geometry``. Fixes how many external degrees of freedom
            are projected out, unless the Hessian check above overrules it.
            Neither ``analyze_vibrations`` nor ``IdealGasThermo`` should be
            given this argument afterwards: pass ``Projection.mode_geometry``
            to the first and ``Projection.geometry`` to the second.
        name: Molecule identifier, for the diagnostic log message only.
        temperature_k: The temperature the thermochemistry will be evaluated
            at, in kelvin. Used only for the classical-rotor floor above, whose
            crossover moment is ``h^2 / (8 pi^3 k T)``; the driver passes its
            own ``T``.

    Returns:
        A :class:`Projection`: the ``3N-6`` (or ``3N-5``, or ``[]``) complex
        energies in eV, the rotor geometry, the geometry whose external-mode
        count was projected out, and the verdict to record in
        ``Thermo_linearity``.
    """
    n_atoms = len(atoms)
    n_vib = n_vibrational_modes(n_atoms, geometry)
    if n_vib <= 0:
        # A monatomic species has no vibrational degrees of freedom at all;
        # nothing to diagonalize and nothing for IdealGasThermo to sum over.
        return Projection(
            energies=[], geometry=geometry, mode_geometry=geometry, linearity=geometry
        )

    masses = np.asarray(atoms.get_masses(), dtype=float)
    if not np.all(masses > 0.0):
        raise ValueError(
            f"{name} has a zero or negative atomic mass; the mass-weighted "
            "Hessian is undefined. Set every mass with Atoms.set_masses()."
        )
    positions = np.asarray(atoms.get_positions(), dtype=float)
    hessian_2d = np.asarray(hessian, dtype=float).reshape(3 * n_atoms, 3 * n_atoms)
    # A Hessian is symmetric; a finite-difference or fp32 analytic one is only
    # nearly so. Symmetrizing before eigh makes the spectrum independent of
    # which triangle LAPACK happens to read.
    hessian_2d = 0.5 * (hessian_2d + hessian_2d.T)

    weights = np.repeat(masses**-0.5, 3)
    mass_weighted = weights[:, np.newaxis] * hessian_2d * weights[np.newaxis, :]

    left_singular, singular, _ = np.linalg.svd(
        _external_mode_basis(positions, masses), full_matrices=False
    )
    n_external = _EXTERNAL_DOF[geometry]
    mode_geometry = linearity = geometry
    kept, discarded = _projected_spectrum(mass_weighted, left_singular[:, :n_external])

    if (
        geometry == "linear"
        # A two-atom molecule has no nonlinear vibrational degrees of freedom at
        # all, so the six-vector projection's `kept` set is empty and the np.min
        # below would raise on a zero-size array. The singular-value conjunct
        # already excludes every diatomic -- the rotation about the internuclear
        # axis is identically zero for any two points, so `singular[5]` is zero
        # -- but this guard states the intent instead of relying on that.
        and n_atoms > 2
        and singular[5] >= LINEAR_AXIS_ROTATION_GATE * singular[0]
    ):
        # The sixth left singular vector is the rotation about the near-axis,
        # orthonormal to the five the linear basis keeps; it is a unit vector,
        # so v @ H @ v is the Rayleigh quotient.
        axis_rotation = left_singular[:, 5]
        curvature = float(axis_rotation @ mass_weighted @ axis_rotation)
        kept_nonlinear, discarded_nonlinear = _projected_spectrum(
            mass_weighted, left_singular[:, :6]
        )
        # min(|.|), not min(): a signed minimum would pick a reaction
        # coordinate's large negative eigenvalue, make the right-hand side of
        # the comparison negative, and leave a BENT SADDLE POINT inside the
        # window unreclassifiable at any curvature -- which is the N-M4 case for
        # a transition state. What the absolute value costs is that a near-zero
        # mode of either sign within ~4.5x of the phantom in frequency blocks
        # the test, conservatively: the molecule then stays linear.
        smallest_nonlinear = float(np.min(np.abs(kept_nonlinear)))
        ratio = abs(curvature) / smallest_nonlinear if smallest_nonlinear else float("inf")
        # The Gram matrix of the basis's rotation columns is the inertia tensor
        # and the translations are orthogonal to them, so the external basis's
        # singular values are sqrt(M) three times and sqrt(I_a): singular[5]**2
        # IS the smallest principal moment, in amu A^2, and equals
        # atoms.get_moments_of_inertia().min() to machine precision.
        i_min = float(singular[5] ** 2)
        classical_floor = _CLASSICAL_ROTOR_FLOOR_AMU_A2_K / temperature_k
        theta_a_over_t = np.pi * _CLASSICAL_ROTOR_FLOOR_AMU_A2_K / (i_min * temperature_k)
        curvature_cm = _HESSIAN_ENERGY_CONVERSION * abs(curvature) ** 0.5 / EV_PER_WAVENUMBER
        smallest_cm = _HESSIAN_ENERGY_CONVERSION * smallest_nonlinear**0.5 / EV_PER_WAVENUMBER
        if abs(curvature) < PROJECTION_RESIDUAL_FRACTION * smallest_nonlinear:
            # Either way the phantom goes: the spectrum is the 3N-6 one.
            mode_geometry = "nonlinear"
            kept, discarded = kept_nonlinear, discarded_nonlinear
            n_vib = len(kept)
            if i_min >= classical_floor:
                geometry, linearity = "nonlinear", BENT_RECLASSIFIED
                logger.warning(
                    "%s: classified linear by its geometry (every atom within %.2f A "
                    "of the principal axis), but its Hessian has a sixth near-zero "
                    "direction: the curvature along the rotation about the near-axis "
                    "is %.3e (%.1f cm-1 equivalent) against a smallest nonlinear "
                    "vibration of %.3e (%.1f cm-1), a mass-weighted-Hessian "
                    "eigenvalue ratio of %.1e, below %.2f. That is a bent stationary "
                    "point inside the linearity window, not a thermally bent linear "
                    "molecule (past the stationary-point gate any residual bend is "
                    "the model's minimum), so it is treated as nonlinear: 3N-6 modes "
                    "and the nonlinear rotational partition function, whose classical "
                    "form is still valid here (smallest principal moment %.3e amu "
                    "A^2, above the %.3e floor at %.2f K; Theta_A/T = %.2f). "
                    "Thermo_linearity records the reclassification. RRHO is marginal "
                    "for a quasi-linear species either way.",
                    name,
                    LINEARITY_MAX_PERP_ANGSTROM,
                    abs(curvature),
                    curvature_cm,
                    smallest_nonlinear,
                    smallest_cm,
                    ratio,
                    PROJECTION_RESIDUAL_FRACTION,
                    i_min,
                    classical_floor,
                    temperature_k,
                    theta_a_over_t,
                )
            else:
                linearity = BENT_QUASILINEAR
                logger.warning(
                    "%s: its Hessian says bent (near-axis curvature %.3e, %.1f cm-1 "
                    "equivalent, against a smallest nonlinear vibration of %.3e, "
                    "%.1f cm-1, eigenvalue ratio %.1e), but its smallest principal "
                    "moment is %.3e amu A^2, below the %.3e classical-rotor floor at "
                    "%.2f K (Theta_A/T = %.2f). The phantom near-axis mode is "
                    "therefore dropped -- 3N-6 modes -- but the linear rotor is kept: "
                    "for Theta_A >> T only K = 0 is populated and the nonlinear "
                    "rotational partition function tends to the linear one, while "
                    "ASE's classical nonlinear form would put q_A below the quantum "
                    "ground state and cost spurious -T*S_rot without bound. "
                    "Thermo_linearity records this as %s. RRHO is marginal for a "
                    "quasi-linear species either way.",
                    name,
                    abs(curvature),
                    curvature_cm,
                    smallest_nonlinear,
                    smallest_cm,
                    ratio,
                    i_min,
                    classical_floor,
                    temperature_k,
                    theta_a_over_t,
                    BENT_QUASILINEAR,
                )
        elif ratio < LINEARITY_MARGINAL_RATIO:
            logger.warning(
                "%s: the linearity test ran and did not fire, but only marginally. "
                "The curvature along the rotation about the near-axis is %.3e (%.1f "
                "cm-1 equivalent) against a smallest nonlinear vibration of %.3e "
                "(%.1f cm-1), a mass-weighted-Hessian eigenvalue ratio of %.2f -- "
                "above the %.2f that would reclassify it, but below %.2f, where a "
                "genuinely linear molecule's degenerate bend pair puts the ratio at "
                "or above 1. A bent stationary point does not give exactly zero "
                "curvature here away from an exact stationary point (the identity is "
                "R^T H R = sum_i o_perp,i . g_perp,i, bounded by fmax * sum|o_perp| / "
                "I_n at the force gate), so this molecule is kept linear with 3N-5 "
                "modes, and a phantom of about %.1f cm-1 may be sitting in that list "
                "-- below the quasi-harmonic floor, hence raised to it. Treat its "
                "rotational and low-frequency thermochemistry as uncertain.",
                name,
                abs(curvature),
                curvature_cm,
                smallest_nonlinear,
                smallest_cm,
                ratio,
                PROJECTION_RESIDUAL_FRACTION,
                LINEARITY_MARGINAL_RATIO,
                curvature_cm,
            )

    largest_discarded = float(np.max(np.abs(discarded)))
    smallest_kept = float(np.min(np.abs(kept)))
    if largest_discarded >= PROJECTION_RESIDUAL_FRACTION * smallest_kept:
        # Projection puts n_external eigenvalues at machine zero by
        # construction, so this fires only when a genuine vibration has become
        # numerically indistinguishable from that null space -- a dissociating
        # fragment, or a Hessian conditioned badly enough that the separation
        # is gone. Reported rather than raised: the spectrum is still the best
        # available one, and this assumption is precisely what the magnitude
        # heuristic made silently and never checked.
        logger.warning(
            "%s: the translation/rotation subspace is not cleanly separated "
            "from the vibrations. Largest projected-out eigenvalue %.3e vs "
            "smallest retained %.3e (ratio %.2f, expected below %.2f). The "
            "%d retained modes may include a rotation or omit a very soft "
            "vibration.",
            name,
            largest_discarded,
            smallest_kept,
            largest_discarded / smallest_kept if smallest_kept else float("inf"),
            PROJECTION_RESIDUAL_FRACTION,
            n_vib,
        )

    energies = _HESSIAN_ENERGY_CONVERSION * kept.astype(complex) ** 0.5
    return Projection(
        energies=[complex(value) for value in energies],
        geometry=geometry,
        mode_geometry=mode_geometry,
        linearity=linearity,
    )


def projected_vibrations(
    atoms: ase.Atoms,
    hessian,
    geometry: str,
    *,
    name: str = "molecule",
) -> list[complex]:
    """``project_vibrations(...).energies``: the list alone, for callers whose geometry is settled.

    The driver uses :func:`project_vibrations` because the effective geometry
    must reach ``analyze_vibrations`` and ``IdealGasThermo`` too.
    """
    return project_vibrations(atoms, hessian, geometry, name=name).energies


@dataclass
class VibrationAnalysis:
    """Verdict on a vibrational spectrum, computed without touching a model.

    Attributes:
        energies: The untouched input -- exactly the ``3N-6`` / ``3N-5``
            projected vibrational modes, in input order. Every diagnostic
            below is computed from this, before any correction.
        corrected_energies: The list actually handed to ``IdealGasThermo``,
            after (1) inverting every sub-cutoff imaginary mode to ``|nu|``,
            (2) removing every remaining imaginary mode, i.e. a genuine
            reaction coordinate, and (3) applying the quasi-harmonic floor to
            the real modes. Its length is ``len(energies) - n_removed``, which
            is ``3N-6`` for a minimum and ``3N-7`` for a first-order saddle
            point.
        n_imag: Imaginary modes in ``energies``.
        n_inverted: How many of those were below ``imag_cutoff_cm`` and were
            therefore kept at ``|nu|``.
        n_removed: How many were at or above it and were therefore dropped --
            the reaction coordinate(s) of a saddle point.
        n_raised: How many modes in ``corrected_energies`` were below
            ``low_freq_cutoff_cm`` and were evaluated at the floor instead.
            Counts inverted artifacts too, since after inversion they are
            ordinary soft real modes.
        max_imag_cm: Largest imaginary wavenumber in ``energies``.
        imag_cutoff_cm: Magnitude at or above which an imaginary mode is a
            reaction coordinate rather than a numerical artifact.
        low_freq_cutoff_cm: The quasi-harmonic floor, in cm^-1; 0.0 means
            plain RRHO with no floor.
        linearity: what ``Thermo_linearity`` records: the geometry the
            thermochemistry used, or ``BENT_RECLASSIFIED`` /
            ``BENT_QUASILINEAR`` when the Hessian overruled the window.
    """

    energies: list[complex]
    n_imag: int
    max_imag_cm: float
    imag_cutoff_cm: float
    corrected_energies: list[complex]
    n_inverted: int
    n_removed: int
    n_raised: int
    low_freq_cutoff_cm: float
    linearity: str

    @property
    def is_transition_state(self) -> bool:
        """True when an imaginary mode is too large to be numerical noise."""
        return self.max_imag_cm >= self.imag_cutoff_cm

    @property
    def convention(self) -> str:
        """The thermochemical convention that produced ``corrected_energies``.

        Written to every record's ``Thermo_convention`` SD property, because
        the quasi-harmonic floor is a modeling choice rather than a bug fix:
        two Auto3D runs with different floors are not comparable, and neither
        is a floored Auto3D number and a plain-RRHO Gaussian/ORCA one.
        """
        if self.low_freq_cutoff_cm > 0.0:
            return f"RRHO+quasiharmonic({self.low_freq_cutoff_cm:g}cm-1)"
        return "RRHO"


def analyze_vibrations(
    vib_energies,
    n_atoms: int,
    geometry: str,
    *,
    imag_cutoff_cm: float = IMAGINARY_MODE_CUTOFF_CM,
    low_freq_cutoff_cm: float = LOW_FREQUENCY_CUTOFF_CM,
    linearity: str | None = None,
) -> VibrationAnalysis:
    """Classify a vibrational spectrum and build the list ASE is given.

    Takes exactly the projected vibrational modes -- ``3N-6`` for a nonlinear
    molecule, ``3N-5`` for a linear one, none for a monatomic -- as
    ``projected_vibrations`` returns them, and raises ``ValueError`` on any
    other length. It does **not** select modes: translation and rotation were
    already removed by projection, so there is nothing here to cut, and the
    magnitude-sorted slice this function used to perform (to mirror what ASE
    would do internally) is gone. Mirroring was never safe -- ASE changed that
    rule in 3.28.0, after which Auto3D's reported ``N_imaginary_modes`` and
    ``Is_transition_state`` described a different mode set from the one that
    produced ``G_hartree``.

    The three diagnostics -- ``n_imag``, ``max_imag_cm`` and
    ``is_transition_state`` -- are computed first, on the untouched input.
    That ordering is load-bearing: they are meaningless once modes have been
    inverted, removed or raised.

    Then three corrections are applied, in this order:

    1. **Invert** every imaginary mode below ``imag_cutoff_cm``, keeping it at
       ``|nu|``. This is the Gaussian/ORCA convention for a numerical
       artifact, and the reason is mode counting rather than the size of any
       one number: a nonlinear molecule has exactly ``3N-6`` vibrational
       degrees of freedom, so deleting an artifact would give a species with
       one artifact a ``3N-7``-mode partition function and a species with none
       a ``3N-6``-mode one. Those two free energies are not the same
       thermodynamic quantity, and the difference does not cancel in the
       comparison a user runs thermochemistry to make.
    2. **Remove** every imaginary mode at or above ``imag_cutoff_cm``. That is
       a genuine reaction coordinate: the rigid-rotor/harmonic partition
       function has no expression for it, and the standard treatment is to
       omit it and report ``3N-7``. Removing it here, rather than leaving it
       to ``ignore_imag_modes``, is what makes the count deliberate -- and it
       is the case ASE >= 3.28's selection got wrong, dropping the reaction
       coordinate at the *selection* stage and pulling a ~1.6 cm-1 rotation
       into the partition function to fill the quota.
    3. **Raise** every remaining real mode below ``low_freq_cutoff_cm`` to the
       floor (Truhlar's quasi-harmonic prescription). The harmonic entropy of
       a mode diverges as ``-R*ln(h*nu/kT)`` as ``nu -> 0``, so G is most
       sensitive to exactly the modes an fp32 NNP Hessian resolves worst; the
       floor makes ``dG/dnu`` zero below the cutoff. It is applied to the
       zero-point and enthalpy sums as well as the entropy, which is what
       handing a single floored list to ``IdealGasThermo`` does. That
       simplification is measured, not assumed: at 298 K a mode below the
       floor carries ``ZPE + dH_vib`` of 0.594 kcal/mol at 30 cm-1 and 0.604
       at 100 cm-1 -- the zero-point rise is cancelled by the thermal-enthalpy
       fall -- so raising everywhere differs from raising inside the entropy
       only by 0.010-0.012 kcal/mol per mode.

    ``imag_cutoff_cm`` and ``low_freq_cutoff_cm`` answer different questions
    and are deliberately not merged. The first is a classification threshold:
    is this geometry a saddle point? The second is a thermodynamic floor: how
    far do we trust a soft mode's frequency? A useful consequence is that once
    the floor is in force the exact artifact cutoff stops mattering for G --
    an artifact at 10i, 20i, 30i or 49i all invert to a sub-floor real mode
    and are all evaluated at the floor, contributing identically.

    One classification detail deliberately does NOT match ASE: this function
    calls a mode imaginary when ``imag(v) != 0``, while ``_clean_vib_energies``
    keeps a mode only when ``real(v) > 0``, i.e. it treats ``real(v) <= 0`` as
    imaginary. The two agree everywhere except for an exactly-zero mode
    (``complex(0, 0)``), which this function calls real. With the
    quasi-harmonic floor in force such a mode is raised to the floor and the
    difference is moot; with the floor disabled it is passed through as a zero
    energy, which is a genuinely singular mode (a dissociated fragment with no
    restoring force) and is reported rather than silently deleted.

    Args:
        vib_energies: Complex vibrational energies in eV -- exactly the
            projected ``3N-6`` / ``3N-5`` set, translation and rotation
            already removed. An imaginary mode has a nonzero imaginary part.
        n_atoms: Number of atoms, for the mode-count check.
        geometry: 'monatomic', 'linear' or 'nonlinear', as classified by
            ``_detect_geometry``, for the mode-count check.
        imag_cutoff_cm: Magnitude at or above which an imaginary mode means
            the structure is a saddle point, not a noisy minimum.
        low_freq_cutoff_cm: Quasi-harmonic floor in cm^-1; 0.0 disables
            raising and gives plain RRHO.
        linearity: The verdict to carry into ``Thermo_linearity``, i.e.
            ``Projection.linearity`` (one of 'monatomic', 'linear',
            'nonlinear', ``BENT_RECLASSIFIED`` or ``BENT_QUASILINEAR``).
            ``None`` means "the geometry itself", which is right for every
            caller whose geometry the Hessian did not overrule; the driver
            passes ``projection.linearity``. Note that in the quasilinear case
            ``geometry`` above is ``Projection.mode_geometry`` ('nonlinear',
            which the mode count must match) while ``linearity`` records that
            the rotor stayed linear.

    Returns:
        A :class:`VibrationAnalysis`.

    Raises:
        ValueError: if ``vib_energies`` does not hold exactly the number of
            vibrational modes ``n_atoms`` and ``geometry`` imply.
    """
    energies = [complex(e) for e in vib_energies]
    expected = n_vibrational_modes(n_atoms, geometry)
    if len(energies) != expected:
        raise ValueError(
            f"analyze_vibrations expects exactly the {expected} vibrational "
            f"mode(s) of a {geometry} {n_atoms}-atom molecule, got "
            f"{len(energies)}. Translation and rotation must already be "
            "removed -- build the list with projected_vibrations."
        )

    # Diagnostics first, on the untouched spectrum: n_imag, max_imag_cm and
    # is_transition_state are meaningless on an inverted or raised list.
    n_imag = 0
    max_imag_cm = 0.0
    for value in energies:
        if abs(value.imag) > 0.0:
            n_imag += 1
            max_imag_cm = max(max_imag_cm, abs(value.imag) / EV_PER_WAVENUMBER)

    floor_ev = max(0.0, low_freq_cutoff_cm) * EV_PER_WAVENUMBER
    corrected: list[complex] = []
    n_inverted = 0
    n_removed = 0
    n_raised = 0
    for value in energies:
        if abs(value.imag) > 0.0:
            if abs(value.imag) / EV_PER_WAVENUMBER < imag_cutoff_cm:
                # Numerical artifact of a soft mode: keep it at |nu| so the
                # mode count is conserved.
                value = complex(abs(value), 0.0)
                n_inverted += 1
            else:
                # Reaction coordinate: no harmonic expression exists for it.
                n_removed += 1
                continue
        if floor_ev > 0.0 and value.real < floor_ev:
            value = complex(floor_ev, 0.0)
            n_raised += 1
        corrected.append(value)

    return VibrationAnalysis(
        energies=energies,
        n_imag=n_imag,
        max_imag_cm=max_imag_cm,
        imag_cutoff_cm=imag_cutoff_cm,
        corrected_energies=corrected,
        n_inverted=n_inverted,
        n_removed=n_removed,
        n_raised=n_raised,
        low_freq_cutoff_cm=max(0.0, low_freq_cutoff_cm),
        linearity=linearity if linearity is not None else geometry,
    )


def _verbatim_mode_kwargs(n_passed: int, n_expected: int) -> dict:
    """``IdealGasThermo`` kwargs that make it consume the mode list verbatim.

    Auto3D builds the vibrational list itself -- projected, inverted, trimmed
    and floored -- so ASE's own selection must not run on top of it. Two
    mechanisms exist across the supported range, and both were read out of the
    installed sources rather than assumed:

    * ASE >= 3.28.0 has ``vib_selection``. ``'exact'`` consumes the list
      unchanged *and* asserts it has the ``3N-6`` / ``3N-5`` length the
      geometry implies, which is a free independent check for the ordinary
      minimum path. A confirmed transition state deliberately supplies
      ``3N-7``, so it uses ``'all'``, which disables both the selection and
      the length check.
    * ASE 3.23.0-3.27.x has no ``vib_selection``; its cut is guarded by
      ``if natoms:``, so passing ``natoms=0`` skips it. ``self.natoms`` is
      assigned there and read nowhere else in the module, and in 3.28+ it is
      not even stored, so this is inert beyond disabling the cut. (In 3.28+
      the same ``natoms=0`` would also work -- ``if natoms and ...`` -- but
      ``vib_selection`` is the documented mechanism, so it is preferred where
      it exists.)

    ASE 3.22.1 is not supported: its ``IdealGasThermo`` has no
    ``ignore_imag_modes`` parameter at all, and it does not sort before
    slicing. ``pyproject.toml`` pins ``ase>=3.23.0`` accordingly.
    """
    if _ASE_HAS_VIB_SELECTION:
        return {"vib_selection": "exact" if n_passed == n_expected else "all"}
    return {"natoms": 0}
