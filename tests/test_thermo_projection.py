"""Auto3D must decide which modes are the vibrations, not ASE.

``VibrationsData.get_energies()`` diagonalizes the raw mass-weighted Hessian
and hands back all ``3N`` eigenvalues, translations and rotations included.
Auto3D used to pass that whole list to ``IdealGasThermo`` and let ASE choose
the ``3N-6``. That is not a stable interface, and it is not even a correct one:

* ASE 3.23.0-3.27.x sort by ``np.abs`` and keep the last ``3N-6``;
* ASE 3.28.0 (2026-03-17) and later sort by ``(f**2).real`` instead, under
  which every imaginary mode ranks below every real one -- so a genuine
  imaginary mode is dropped by the *selection* and a ~1.6 cm-1 rotation is
  promoted into the vibrational partition function to fill the quota.

Both rules rest on an assumption nothing checks: that every translation and
rotation eigenvalue is smaller in magnitude than every vibrational one. That
holds at a converged stationary point and fails off it, and no selection rule
can recover the information -- once the eigenvalues are a flat list of complex
numbers, "is this a rotation" is unanswerable except by magnitude.

``projected_vibrations`` removes translation and rotation by Eckart/Sayvetz
projection instead, so the count is fixed by the geometry and the null space is
exact by construction. These tests build Hessians directly (synthetic ones with
a prescribed spectrum, and real MMFF ones); no neural network potential is
loaded anywhere.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.vibrations import VibrationsData

from Auto3D.entry.ASE.thermo.properties import _detect_geometry
from Auto3D.entry.ASE.thermo.vibrations import (
    _CLASSICAL_ROTOR_FLOOR_AMU_A2_K,
    BENT_QUASILINEAR,
    BENT_RECLASSIFIED,
    _external_mode_basis,
    _projected_spectrum,
    n_vibrational_modes,
    project_vibrations,
    projected_vibrations,
)
from Auto3D.foundation.constants import (
    LINEARITY_MARGINAL_RATIO,
    PROJECTION_RESIDUAL_FRACTION,
)
from tests.helpers_vibrations import (
    ASE_SELECTION_RULES,
    atoms_for,
    co2_atoms,
    hessian_with_spectrum,
    mmff_hessian,
    n_vib_expected,
    probe_mol,
    wavenumbers,
)

#: A realistic 21-mode organic spectrum for a 9-atom molecule (ethanol).
REAL_MODES = (
    250,
    300,
    420,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1450,
    1470,
    2900,
    2950,
    3000,
    3010,
    3020,
    3050,
    3600,
    3700,
)
#: Translation/rotation eigenvalues at the magnitudes a converged NNP Hessian
#: actually produces -- mixed real and imaginary, a few cm-1 either way. The
#: analysis behind this work measured 1.6-4.1 cm-1 on MMFF n-decane at Auto3D's
#: own thermo convergence gate (2e-4 eV/A), an order of magnitude below the
#: lowest genuine vibration (36 cm-1).
TRANS_ROT_NOISE = (1.6, -3.2, 3.4, -3.5, 3.7, -4.1)


def _ethanol():
    mol = probe_mol("CCO")
    return mol, atoms_for(mol)


class TestTheModeCountComesFromTheGeometry:
    def test_a_nonlinear_molecule_yields_exactly_3n_minus_6(self):
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        energies = projected_vibrations(atoms, hessian, "nonlinear")
        assert len(energies) == 3 * len(atoms) - 6 == 21

    def test_a_linear_molecule_yields_exactly_3n_minus_5(self):
        atoms = Atoms("OCO", [[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
        assert _detect_geometry(atoms) == "linear", "test premise"
        hessian = hessian_with_spectrum(
            atoms, [667, 667, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1], "linear"
        )
        energies = projected_vibrations(atoms, hessian, "linear")
        assert len(energies) == 3 * 3 - 5 == 4
        assert sorted(wavenumbers(energies)) == pytest.approx([667, 667, 1333, 2349], abs=1e-6)

    def test_a_monatomic_species_has_no_vibrations(self):
        atoms = Atoms("Ar", [[0.0, 0.0, 0.0]])
        assert projected_vibrations(atoms, np.zeros((3, 3)), "monatomic") == []

    def test_an_unknown_geometry_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported geometry"):
            n_vibrational_modes(3, "planar")


class TestTheRotationCountIsNotAnSvdRankTest:
    """``_detect_geometry`` decides, and it must, because the two disagree.

    ``_is_collinear`` deliberately calls a molecule linear up to
    ``LINEARITY_MAX_PERP_ANGSTROM = 0.25 A`` of bend, because CO2's real
    bending mode is thermally populated to several degrees at room temperature
    and an optimizer leaves residual curvature there. An SVD rank test on the
    six translation/rotation vectors flips to "nonlinear" as soon as the third
    rotation vector is numerically nonzero -- around 1e-6 A. If the projection
    took its count from the rank while ``IdealGasThermo`` took its rotational
    partition function from ``_detect_geometry``, the two halves of G would
    describe different molecules and the error would be a whole low-frequency
    mode.
    """

    @staticmethod
    def _bent_co2(perpendicular_angstrom: float) -> Atoms:
        return Atoms(
            "OCO",
            [[-1.16, 0.0, 0.0], [0.0, perpendicular_angstrom, 0.0], [1.16, 0.0, 0.0]],
        )

    def test_a_thermally_bent_co2_keeps_all_five_external_modes(self):
        atoms = self._bent_co2(0.074)
        assert _detect_geometry(atoms) == "linear"

        basis = _external_mode_basis(
            np.asarray(atoms.get_positions(), float),
            np.asarray(atoms.get_masses(), float),
        )
        singular = np.linalg.svd(basis, compute_uv=False)
        rank = int(np.sum(singular > 1e-8 * singular[0]))
        assert rank == 6, (
            "test premise: an SVD rank test sees six independent external "
            f"vectors here (singular values {np.round(singular, 4)}), so it "
            "would call this molecule nonlinear"
        )
        assert singular[-1] == pytest.approx(0.219, abs=0.01), (
            "test premise: the third rotation vector is far from negligible"
        )

        hessian = hessian_with_spectrum(
            atoms, [667, 667, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1], "linear"
        )
        energies = projected_vibrations(atoms, hessian, "linear")
        assert len(energies) == 4, (
            "the projection took its external-mode count from the SVD rank "
            "(6) instead of from _detect_geometry (5), so a genuine bending "
            "mode was discarded"
        )

    def test_a_genuinely_bent_triatomic_is_nonlinear_and_keeps_three(self):
        """Non-vacuity: the linear branch is not simply always taken."""
        atoms = self._bent_co2(0.3)
        assert _detect_geometry(atoms) == "nonlinear"
        hessian = hessian_with_spectrum(
            atoms, [667, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1, 0.05], "nonlinear"
        )
        assert len(projected_vibrations(atoms, hessian, "nonlinear")) == 3


class TestTheFixtureIsWhatAseWouldSee:
    """Non-vacuity for every synthetic-Hessian test in this repo.

    If ``hessian_with_spectrum`` did not really produce the spectrum it claims,
    every projection assertion below would be comparing one bug against
    another. This checks it through ASE's own, independent diagonalization.
    """

    def test_ase_reads_back_exactly_the_prescribed_3n_spectrum(self):
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [-20, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        n_atoms = len(atoms)
        raw = VibrationsData(atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)).get_energies()
        assert sorted(wavenumbers(raw)) == pytest.approx(
            sorted([-20, *REAL_MODES, *TRANS_ROT_NOISE]), abs=1e-6
        )


class TestTranslationAndRotationNeverEnterTheSpectrum:
    def test_the_noise_modes_are_gone_and_every_vibration_survives(self):
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [-20, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        energies = projected_vibrations(atoms, hessian, "nonlinear")
        assert sorted(wavenumbers(energies)) == pytest.approx(sorted([-20, *REAL_MODES]), abs=1e-6)

    def test_a_noise_mode_larger_than_a_real_vibration_is_still_removed(self):
        """The heuristic's one assumption, violated on purpose.

        Sorting by magnitude only works while every translation/rotation
        eigenvalue is smaller than every vibrational one. Here a rotation sits
        at 120 cm-1 and a genuine torsion at 35 cm-1, so both ASE rules keep
        the rotation and throw the torsion away. Projection does not care:
        it removes the external subspace by construction, not by size.
        """
        _, atoms = _ethanol()
        vibrations = [35, *REAL_MODES]
        external = (120.0, -3.2, 3.4, -3.5, 3.7, -4.1)
        hessian = hessian_with_spectrum(atoms, vibrations, external, "nonlinear")
        energies = projected_vibrations(atoms, hessian, "nonlinear")
        assert sorted(wavenumbers(energies)) == pytest.approx(sorted(vibrations), abs=1e-6)

        n_atoms = len(atoms)
        raw = VibrationsData(atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)).get_energies()
        for label, rule in ASE_SELECTION_RULES.items():
            picked = sorted(wavenumbers(rule(raw, 3 * n_atoms - 6)))
            assert picked[0] == pytest.approx(120.0, abs=1e-6), (
                f"test premise: the {label} rule is supposed to fail here"
            )
            assert not any(abs(w - 35.0) < 1e-6 for w in picked), (
                f"test premise: the {label} rule is supposed to discard the "
                "genuine 35 cm-1 torsion here"
            )


class TestAgainstARealForceFieldHessian:
    """The non-synthetic anchor: an MMFF Hessian, not one built to order."""

    def test_at_a_tight_stationary_point_projection_matches_both_ase_rules(self):
        """Projection costs nothing where the heuristic works.

        At a converged minimum the six external eigenvalues really are the
        smallest, so both selection rules pick the right modes -- and the
        projected frequencies must be identical to them, not merely close,
        because the vibrational eigenvectors carry no rotational contamination
        when the gradient vanishes.
        """
        atoms, hessian = mmff_hessian("CCCC")
        n_atoms = len(atoms)
        n_vib = 3 * n_atoms - 6
        projected = sorted(wavenumbers(projected_vibrations(atoms, hessian, "nonlinear")))
        raw = VibrationsData(atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)).get_energies()

        for label, rule in ASE_SELECTION_RULES.items():
            heuristic = sorted(wavenumbers(rule(raw, n_vib)))
            assert projected == pytest.approx(heuristic, abs=5e-3), (
                f"projection disagrees with the {label} rule at a stationary "
                "point, where they must agree"
            )
        # Not vacuous: this is a real spectrum, with no imaginary modes and a
        # genuine low-frequency torsion well clear of the noise floor.
        assert min(projected) == pytest.approx(122.9, abs=0.5)
        assert max(projected) == pytest.approx(3010, abs=60)

    def test_off_the_stationary_point_the_square_sort_rule_loses_the_imaginary_mode(
        self,
    ):
        """And this is the bug that shipped.

        Displaced off the minimum, MMFF n-butane has genuine imaginary modes.
        The ``(f**2).real`` key ASE adopted in 3.28.0 sorts those below every
        real mode, so the selection drops them and substitutes
        near-zero rotations -- reporting a structure with a large reaction
        coordinate as if it had none.
        """
        rng = np.random.default_rng(7)
        base, _ = mmff_hessian("CCCC")
        displacement = rng.normal(0.0, 0.05, 3 * len(base))
        displacement = displacement / np.abs(displacement).max() * 0.15
        atoms, hessian = mmff_hessian("CCCC", displacement=displacement)
        n_atoms = len(atoms)
        n_vib = 3 * n_atoms - 6
        projected = sorted(wavenumbers(projected_vibrations(atoms, hessian, "nonlinear")))
        raw = VibrationsData(atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)).get_energies()

        assert sum(1 for w in projected if w < 0) == 2, (
            f"test premise: the displaced structure has two genuine imaginary "
            f"modes, got {[w for w in projected if w < 0]}"
        )

        square_sorted = sorted(
            wavenumbers(ase_rule := ASE_SELECTION_RULES["square-sort (ASE >=3.28)"](raw, n_vib))
        )
        assert ase_rule is not None
        assert all(w > 0 for w in square_sorted), (
            "test premise: the >=3.28 rule is supposed to discard every "
            f"imaginary mode here, got {square_sorted[:4]}"
        )
        assert min(square_sorted) < 1.0, (
            "the >=3.28 rule promoted a near-zero rotation into the "
            f"vibrational set, expected here; got {min(square_sorted)}"
        )

        abs_sorted = sorted(
            wavenumbers(ASE_SELECTION_RULES["abs-sort (ASE 3.23-3.27)"](raw, n_vib))
        )
        assert abs_sorted != pytest.approx(projected, abs=1.0), (
            "the abs-sort rule is supposed to differ from the projected "
            "spectrum off a stationary point"
        )


class TestTheProjectionReportsWhatItAssumed:
    def test_a_collapsed_separation_is_logged(self, caplog):
        """The assumption the magnitude heuristic made silently, now checked.

        Projection puts the external eigenvalues at machine zero, so this can
        only fire when a genuine vibration has itself collapsed to numerical
        zero -- a dissociating fragment, or a Hessian conditioned badly enough
        that the separation is gone.
        """
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [0.0, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            energies = projected_vibrations(atoms, hessian, "nonlinear", name="floppy")
        assert len(energies) == 21, "the mode was dropped instead of reported"
        assert any("not cleanly separated" in record.getMessage() for record in caplog.records), (
            f"no warning for a collapsed separation: {[r.getMessage() for r in caplog.records]}"
        )

    def test_a_healthy_spectrum_is_silent(self, caplog):
        """Non-vacuity: the warning must discriminate."""
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [35, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            projected_vibrations(atoms, hessian, "nonlinear", name="healthy")
        assert not any("not cleanly separated" in record.getMessage() for record in caplog.records)
        # The threshold really is a ratio against the smallest kept mode, not
        # an absolute number: 35 cm-1 is small, and it stays silent.
        assert PROJECTION_RESIDUAL_FRACTION == 0.05

    def test_a_zero_mass_is_refused(self):
        _, atoms = _ethanol()
        masses = np.asarray(atoms.get_masses(), dtype=float)
        masses[0] = 0.0
        atoms.set_masses(masses)
        with pytest.raises(ValueError, match="mass"):
            projected_vibrations(atoms, np.zeros((3 * len(atoms), 3 * len(atoms))), "nonlinear")


class TestHessianShapeAndMasses:
    def test_the_four_index_hessian_ase_uses_is_accepted(self):
        _, atoms = _ethanol()
        n_atoms = len(atoms)
        hessian = hessian_with_spectrum(atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        flat = projected_vibrations(atoms, hessian, "nonlinear")
        nested = projected_vibrations(atoms, hessian.reshape(n_atoms, 3, n_atoms, 3), "nonlinear")
        assert wavenumbers(flat) == pytest.approx(wavenumbers(nested), abs=1e-9)

    def test_an_asymmetric_hessian_is_symmetrized_before_diagonalizing(self):
        """A Hessian is symmetric; a finite-difference or fp32 one is only nearly so.

        ``numpy.linalg.eigvalsh`` reads a single triangle, so without an
        explicit symmetrization the spectrum silently depends on which
        triangle LAPACK happens to use -- i.e. on a detail of the caller's
        Hessian layout rather than on the physics.
        """
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 0.05, hessian.shape)
        antisymmetric = noise - noise.T
        clean = sorted(wavenumbers(projected_vibrations(atoms, hessian, "nonlinear")))
        perturbed = sorted(
            wavenumbers(projected_vibrations(atoms, hessian + antisymmetric, "nonlinear"))
        )
        assert perturbed == pytest.approx(clean, abs=1e-6), (
            "an antisymmetric perturbation changed the spectrum, so only one "
            "triangle of the Hessian is being read"
        )
        # Non-vacuity: the perturbation is large enough to matter if read.
        assert np.abs(antisymmetric).max() > 0.05

    def test_isotope_masses_reach_the_mass_weighting(self):
        """Deuteration must move the spectrum, or masses are being ignored."""
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
        light = sorted(wavenumbers(projected_vibrations(atoms, hessian, "nonlinear")))

        heavy_atoms = atoms.copy()
        masses = np.asarray(heavy_atoms.get_masses(), dtype=float)
        masses[[a.index for a in heavy_atoms if a.symbol == "H"]] = 2.0141
        heavy_atoms.set_masses(masses)
        heavy = sorted(wavenumbers(projected_vibrations(heavy_atoms, hessian, "nonlinear")))
        assert max(heavy) < max(light) * 0.85, (
            "deuterating every hydrogen did not lower the highest stretch; the "
            "masses are not reaching the mass weighting"
        )


#: A bent stationary point: three vibrations and six external noise modes. At
#: 170 degrees it sits inside the linearity window, which is the N-M4 case.
BENT_CO2 = ([667, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1, 0.05], "nonlinear")
#: A linear molecule: the degenerate bend pair, two stretches, five external noise modes.
LINEAR_CO2 = ([667, 667, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1], "linear")


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


class TestABentStationaryPointInsideTheWindowIsReclassified:
    """N-M4: the linearity window cannot tell a thermally bent linear molecule
    from a bent minimum, but the Hessian can. A bent minimum annihilates the
    rotation about its near-axis; a linear molecule's Hessian puts the bend's
    curvature there. ``project_vibrations`` asks, and the mode count follows
    the answer rather than the window."""

    def test_a_bent_stationary_point_is_projected_as_nonlinear_and_warned(self, caplog):
        atoms = co2_atoms(170.0)
        assert _detect_geometry(atoms) == "linear", "test premise: 170 deg is inside the window"
        hessian = hessian_with_spectrum(atoms, *BENT_CO2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            projection = project_vibrations(atoms, hessian, "linear", name="bent")
        assert projection.geometry == "nonlinear"
        assert projection.linearity == BENT_RECLASSIFIED
        assert sorted(wavenumbers(projection.energies)) == pytest.approx(
            [667, 1333, 2349], abs=1e-6
        )
        messages = _warnings(caplog)
        assert len(messages) == 1 and "bent stationary point" in messages[0], messages

    def test_an_exactly_linear_molecule_keeps_3n_minus_5_and_is_silent(self, caplog):
        atoms = co2_atoms(180.0)
        hessian = hessian_with_spectrum(atoms, *LINEAR_CO2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            projection = project_vibrations(atoms, hessian, "linear", name="linear")
        assert (projection.geometry, projection.linearity) == ("linear", "linear")
        assert sorted(wavenumbers(projection.energies)) == pytest.approx(
            [667, 667, 1333, 2349], abs=1e-6
        )
        assert _warnings(caplog) == []

    def test_a_thermally_bent_linear_molecule_is_not_reclassified(self, caplog):
        """Same 170-degree geometry as the bent case; only the Hessian differs."""
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(atoms, *LINEAR_CO2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            projection = project_vibrations(atoms, hessian, "linear", name="thermal")
        assert (projection.geometry, projection.linearity) == ("linear", "linear")
        assert len(projection.energies) == 4
        assert _warnings(caplog) == []

    @pytest.mark.parametrize(
        "vibrations, external, built_as, reclassified",
        [
            ([40, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1, 0.05], "nonlinear", True),
            ([40, 40, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1], "linear", False),
        ],
        ids=["soft-bend-of-a-bent-minimum", "soft-degenerate-bend-of-a-linear-molecule"],
    )
    def test_a_soft_real_mode_is_not_mistaken_for_the_phantom(
        self, vibrations, external, built_as, reclassified
    ):
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(atoms, vibrations, external, built_as)
        projection = project_vibrations(atoms, hessian, "linear")
        assert (projection.linearity == BENT_RECLASSIFIED) is reclassified
        assert len(projection.energies) == len(vibrations)

    def test_a_diatomic_is_never_tested(self):
        """A diatomic stays linear, and the singular-value gate is why.

        Two points have no sixth external direction at all, so ``s6/s1`` is
        identically 0 and ``LINEAR_AXIS_ROTATION_GATE`` already excludes every
        diatomic before the ``n_atoms > 2`` guard is consulted. This test
        therefore pins the OUTCOME -- 3N-5 = 1 mode and the linear rotor, never
        a reclassification -- not the guard; what the guard is for is stated at
        its own site in ``vibrations.py``.
        """
        atoms = Atoms("NN", [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        hessian = hessian_with_spectrum(atoms, [2330], [0.5, -0.4, 0.3, -0.2, 0.1], "linear")
        projection = project_vibrations(atoms, hessian, "linear")
        assert (projection.geometry, projection.linearity, len(projection.energies)) == (
            "linear",
            "linear",
            1,
        )

    def test_projected_vibrations_is_the_energies_of_project_vibrations(self):
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(atoms, *BENT_CO2)
        assert (
            projected_vibrations(atoms, hessian, "linear")
            == project_vibrations(atoms, hessian, "linear").energies
        )


@pytest.mark.parametrize(
    "vibrations, reclassified",
    [
        ([149, 667, 1333, 2349], True),
        ([300, 667, 1333, 2349], False),
    ],
    ids=[
        "near-axis-partner-4.47x-softer-reclassified",
        "near-axis-partner-2.2x-softer-stays-linear",
    ],
)
def test_the_stays_linear_boundary_is_a_bend_pair_split(vibrations, reclassified):
    """The gate's boundary, pinned on the linear side with a NON-degenerate pair.

    ``_projected_spectrum(mass_weighted, left_singular[:, :6])`` removes exactly
    the near-axis direction, so the comparison is bend partner against bend
    partner: the gate fires when the near-axis partner is more than
    ``1/sqrt(0.05)`` = 4.47x softer in frequency than the smallest remaining
    mode. Every other stays-linear test in this file uses a degenerate pair
    (ratio 1.0) and so cannot see this boundary at all. Both rows below are
    built as LINEAR Hessians, i.e. no bent stationary point is involved -- the
    bend-pair split alone decides.
    """
    atoms = co2_atoms(170.0)
    hessian = hessian_with_spectrum(atoms, vibrations, [0.5, -0.4, 0.3, -0.2, 0.1], "linear")
    projection = project_vibrations(atoms, hessian, "linear")
    assert (projection.linearity == BENT_RECLASSIFIED) is reclassified
    assert len(projection.energies) == (3 if reclassified else 4)


class TestAMarginalLinearityTestIsReported:
    """Chem Major 2: the "bent curvature is zero" claim holds only at ``g = 0``.

    The exact identity is ``R^T H R = sum_i o_perp,i . g_perp,i``, so at Auto3D's
    2e-4 eV/A stationary-point gate the near-axis curvature of a bent minimum is
    bounded by ``fmax * sum|o_perp| / I_n`` -- 9 cm-1 equivalent for CO2 at 170
    degrees, 28 at 179 -- not zero. A bent molecule whose bend is soft enough can
    therefore land above ``PROJECTION_RESIDUAL_FRACTION`` and keep its phantom in
    the 3N-5 list, where the quasi-harmonic floor raises it silently. The band
    ``PROJECTION_RESIDUAL_FRACTION <= ratio < LINEARITY_MARGINAL_RATIO`` is where
    that is plausible, and it is warned about rather than acted on.
    """

    def test_a_marginal_ratio_stays_linear_and_is_warned(self, caplog):
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(
            atoms, [300, 667, 1333, 2349], [0.5, -0.4, 0.3, -0.2, 0.1], "linear"
        )
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            projection = project_vibrations(atoms, hessian, "linear", name="marginal")
        assert (projection.geometry, projection.linearity) == ("linear", "linear")
        assert len(projection.energies) == 4, "a marginal ratio must not reclassify"
        messages = _warnings(caplog)
        assert len(messages) == 1, messages
        assert "marginal" in messages[0], messages[0]

    def test_a_degenerate_bend_pair_is_silent(self, caplog):
        """Non-vacuity: ratio 1.0, which is where every genuinely linear molecule sits."""
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(atoms, *LINEAR_CO2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            project_vibrations(atoms, hessian, "linear", name="degenerate")
        assert _warnings(caplog) == []

    def test_the_marginal_band_sits_above_the_reclassification_threshold(self):
        assert PROJECTION_RESIDUAL_FRACTION < LINEARITY_MARGINAL_RATIO


class TestTheClassicalRotorFloorKeepsTheLinearRotor:
    """Chem Critical 1 / R27: a reclassified molecule gets the nonlinear rotor
    only where the classical rigid-rotor formula is valid.

    ASE's nonlinear rotational entropy is the classical
    ``R/2 ln(pi T^3 / (Theta_A Theta_B Theta_C))``, which needs
    ``Theta_A << T``. Inside the linearity window that fails: at 178 degrees CO2
    has ``Theta_A / T = 22.7``, the classical ``q_A = sqrt(pi T / Theta_A)``
    drops below 1 -- under the quantum ground state -- and ``dG_rot`` grows
    without bound (+1.0 kcal/mol at 179 degrees, +2.4 at 179.9). The quantum
    limit for ``Theta_A >> T`` is that only ``K = 0`` is populated and
    ``q_rot(nonlinear) -> q_rot(linear)``, so below the floor the phantom is
    dropped (3N-6 modes) but the LINEAR rotor is kept.
    """

    def test_the_floor_constant_is_the_quantum_crossover(self):
        """``h^2 / (8 pi^3 k)`` in amu A^2 K, and the 298.15 K moment it implies."""
        assert 7.6 < _CLASSICAL_ROTOR_FLOOR_AMU_A2_K < 7.8
        assert 0.0255 < _CLASSICAL_ROTOR_FLOOR_AMU_A2_K / 298.15 < 0.0263

    def test_i_min_is_the_sixth_singular_value_squared(self):
        """The guard reads ``singular[5] ** 2``; that IS the smallest moment.

        The Gram matrix of ``_external_mode_basis``'s three rotation columns is
        the inertia tensor and the translations are orthogonal to them, so the
        external basis's singular values are ``sqrt(M)`` three times and
        ``sqrt(I_a)``. Reading the moment off the SVD the gate already computed
        costs nothing; this pins that it is the same number ASE would report.
        """
        atoms = co2_atoms(170.0)
        basis = _external_mode_basis(
            np.asarray(atoms.get_positions(), float),
            np.asarray(atoms.get_masses(), float),
        )
        singular = np.linalg.svd(basis, compute_uv=False)
        assert singular[5] ** 2 == pytest.approx(
            float(min(atoms.get_moments_of_inertia())), rel=1e-9
        )

    def test_a_bent_stationary_point_below_the_floor_keeps_the_linear_rotor(self, caplog):
        """178 degrees: I_min 0.00358 amu A^2, below the 0.0259 floor at 298.15 K."""
        atoms = co2_atoms(178.0)
        assert _detect_geometry(atoms) == "linear", "test premise: inside the window"
        hessian = hessian_with_spectrum(atoms, *BENT_CO2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            projection = project_vibrations(atoms, hessian, "linear", name="quasilinear")
        assert (projection.geometry, projection.mode_geometry, projection.linearity) == (
            "linear",
            "nonlinear",
            BENT_QUASILINEAR,
        )
        assert len(projection.energies) == 3, "the phantom is still dropped"
        assert sorted(wavenumbers(projection.energies)) == pytest.approx(
            [667, 1333, 2349], abs=1e-6
        )
        messages = _warnings(caplog)
        assert len(messages) == 1 and "linear rotor is kept" in messages[0], messages

    @pytest.mark.parametrize(
        "temperature_k, expected_quasilinear",
        [(298.15, True), (1000.0, False)],
        ids=["298K-below-the-floor", "1000K-above-the-floor"],
    )
    def test_the_rotor_switch_follows_the_temperature(self, temperature_k, expected_quasilinear):
        """177 degrees: I_min 0.00805, floor 0.0259 at 298.15 K and 0.00772 at 1000 K.

        The crossover moment is ``h^2 / (8 pi^3 k T)``, so a fixed 298 K constant
        would switch rotors at the wrong angle at any other temperature. The
        driver passes its own ``T``.
        """
        atoms = co2_atoms(177.0)
        hessian = hessian_with_spectrum(atoms, *BENT_CO2)
        projection = project_vibrations(atoms, hessian, "linear", temperature_k=temperature_k)
        assert projection.linearity == (
            BENT_QUASILINEAR if expected_quasilinear else BENT_RECLASSIFIED
        )
        assert projection.geometry == ("linear" if expected_quasilinear else "nonlinear")
        assert projection.mode_geometry == "nonlinear", "3N-6 modes either way"
        assert len(projection.energies) == 3

    def test_above_the_floor_the_nonlinear_rotor_is_used(self):
        """Non-vacuity: 170 degrees has I_min 0.0893, well above the floor."""
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(atoms, *BENT_CO2)
        projection = project_vibrations(atoms, hessian, "linear")
        assert (projection.geometry, projection.mode_geometry, projection.linearity) == (
            "nonlinear",
            "nonlinear",
            BENT_RECLASSIFIED,
        )

    def test_mode_geometry_equals_geometry_when_the_test_does_not_fire(self):
        atoms = co2_atoms(170.0)
        hessian = hessian_with_spectrum(atoms, *LINEAR_CO2)
        projection = project_vibrations(atoms, hessian, "linear")
        assert projection.mode_geometry == projection.geometry == projection.linearity == "linear"


class TestARealForceFieldHessianOnALinearMolecule:
    """Chem Minor 5: the synthetic "linear" fixture encodes the answer.

    ``hessian_with_spectrum(..., "linear")`` assigns ``vibrations_cm[0]`` to the
    sixth singular vector -- the very direction the code tests -- so every
    synthetic stays-linear case passes because the fixture put the bend there,
    not because a linear molecule's physics puts it there. An MMFF Hessian does
    not: it is built from energy second differences with no knowledge of the
    projection, so where the bend curvature lands is physics.
    """

    @staticmethod
    def _near_axis_ratio(atoms, hessian):
        """``|q| / min|3N-6 eigenvalue|``, the quantity the gate compares."""
        masses = np.asarray(atoms.get_masses(), float)
        symmetric = 0.5 * (hessian + hessian.T)
        weights = np.repeat(masses**-0.5, 3)
        mass_weighted = weights[:, None] * symmetric * weights[None, :]
        left, _, _ = np.linalg.svd(
            _external_mode_basis(np.asarray(atoms.get_positions(), float), masses),
            full_matrices=False,
        )
        axis_rotation = left[:, 5]
        curvature = float(axis_rotation @ mass_weighted @ axis_rotation)
        kept, _ = _projected_spectrum(mass_weighted, left[:, :6])
        return abs(curvature) / float(np.min(np.abs(kept)))

    def test_mmff_co2_at_its_minimum_stays_linear(self):
        atoms, hessian = mmff_hessian("O=C=O")
        assert _detect_geometry(atoms) == "linear", "test premise"
        # The bend curvature really is on the near-axis direction: ratio ~1, not
        # the ~1e-9 a bent stationary point gives. Measured 1.000.
        assert self._near_axis_ratio(atoms, hessian) == pytest.approx(1.0, abs=0.05)
        projection = project_vibrations(atoms, hessian, "linear", name="mmff-co2")
        assert projection.linearity == "linear"
        assert len(projection.energies) == 4

    def test_mmff_co2_bent_inside_the_window_stays_linear(self):
        """The carbon moved 0.05 A off axis -- still inside the 0.25 A window."""
        reference, _ = mmff_hessian("O=C=O")
        moments, axes = reference.get_moments_of_inertia(vectors=True)
        axis = axes[int(np.argmin(moments))]
        perpendicular = np.cross(axis, [0.0, 0.0, 1.0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        displacement = np.zeros(3 * len(reference))
        displacement[3:6] = 0.05 * perpendicular

        atoms, hessian = mmff_hessian("O=C=O", displacement=displacement)
        assert _detect_geometry(atoms) == "linear", "test premise: inside the window"
        # Measured 1.005: the bend partner, not a noise eigenvalue.
        assert self._near_axis_ratio(atoms, hessian) == pytest.approx(1.0, abs=0.05)
        projection = project_vibrations(atoms, hessian, "linear", name="mmff-co2-bent")
        assert projection.linearity == "linear", (
            "a real bent-but-linear CO2 was reclassified; the Hessian test is "
            "reading noise rather than the bend on the near-axis direction"
        )
        assert len(projection.energies) == 4
