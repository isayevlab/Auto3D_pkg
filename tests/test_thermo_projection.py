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

from Auto3D.ASE.thermo import (
    _detect_geometry,
    _external_mode_basis,
    n_vibrational_modes,
    projected_vibrations,
)
from Auto3D.constants import PROJECTION_RESIDUAL_FRACTION
from tests.helpers_vibrations import (
    ASE_SELECTION_RULES,
    atoms_for,
    hessian_with_spectrum,
    mmff_hessian,
    n_vib_expected,
    probe_mol,
    wavenumbers,
)

#: A realistic 21-mode organic spectrum for a 9-atom molecule (ethanol).
REAL_MODES = (
    250, 300, 420, 800, 900, 1000, 1100, 1200, 1300, 1400,
    1450, 1470, 2900, 2950, 3000, 3010, 3020, 3050, 3600, 3700,
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
        hessian = hessian_with_spectrum(
            atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
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
        assert sorted(wavenumbers(energies)) == pytest.approx(
            [667, 667, 1333, 2349], abs=1e-6
        )

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
        hessian = hessian_with_spectrum(
            atoms, [-20, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        n_atoms = len(atoms)
        raw = VibrationsData(
            atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)
        ).get_energies()
        assert sorted(wavenumbers(raw)) == pytest.approx(
            sorted([-20, *REAL_MODES, *TRANS_ROT_NOISE]), abs=1e-6
        )


class TestTranslationAndRotationNeverEnterTheSpectrum:
    def test_the_noise_modes_are_gone_and_every_vibration_survives(self):
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(
            atoms, [-20, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        energies = projected_vibrations(atoms, hessian, "nonlinear")
        assert sorted(wavenumbers(energies)) == pytest.approx(
            sorted([-20, *REAL_MODES]), abs=1e-6
        )

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
        assert sorted(wavenumbers(energies)) == pytest.approx(
            sorted(vibrations), abs=1e-6
        )

        n_atoms = len(atoms)
        raw = VibrationsData(
            atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)
        ).get_energies()
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
        raw = VibrationsData(
            atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)
        ).get_energies()

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
        raw = VibrationsData(
            atoms, hessian.reshape(n_atoms, 3, n_atoms, 3)
        ).get_energies()

        assert sum(1 for w in projected if w < 0) == 2, (
            f"test premise: the displaced structure has two genuine imaginary "
            f"modes, got {[w for w in projected if w < 0]}"
        )

        square_sorted = sorted(wavenumbers(ase_rule := ASE_SELECTION_RULES[
            "square-sort (ASE >=3.28)"
        ](raw, n_vib)))
        assert ase_rule is not None
        assert all(w > 0 for w in square_sorted), (
            "test premise: the >=3.28 rule is supposed to discard every "
            f"imaginary mode here, got {square_sorted[:4]}"
        )
        assert min(square_sorted) < 1.0, (
            "the >=3.28 rule promoted a near-zero rotation into the "
            f"vibrational set, expected here; got {min(square_sorted)}"
        )

        abs_sorted = sorted(wavenumbers(ASE_SELECTION_RULES[
            "abs-sort (ASE 3.23-3.27)"
        ](raw, n_vib)))
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
        hessian = hessian_with_spectrum(
            atoms, [0.0, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            energies = projected_vibrations(atoms, hessian, "nonlinear", name="floppy")
        assert len(energies) == 21, "the mode was dropped instead of reported"
        assert any(
            "not cleanly separated" in record.getMessage() for record in caplog.records
        ), f"no warning for a collapsed separation: {[r.getMessage() for r in caplog.records]}"

    def test_a_healthy_spectrum_is_silent(self, caplog):
        """Non-vacuity: the warning must discriminate."""
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(
            atoms, [35, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            projected_vibrations(atoms, hessian, "nonlinear", name="healthy")
        assert not any(
            "not cleanly separated" in record.getMessage() for record in caplog.records
        )
        # The threshold really is a ratio against the smallest kept mode, not
        # an absolute number: 35 cm-1 is small, and it stays silent.
        assert PROJECTION_RESIDUAL_FRACTION == 0.05

    def test_a_zero_mass_is_refused(self):
        _, atoms = _ethanol()
        masses = np.asarray(atoms.get_masses(), dtype=float)
        masses[0] = 0.0
        atoms.set_masses(masses)
        with pytest.raises(ValueError, match="mass"):
            projected_vibrations(
                atoms, np.zeros((3 * len(atoms), 3 * len(atoms))), "nonlinear"
            )


class TestHessianShapeAndMasses:
    def test_the_four_index_hessian_ase_uses_is_accepted(self):
        _, atoms = _ethanol()
        n_atoms = len(atoms)
        hessian = hessian_with_spectrum(
            atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        flat = projected_vibrations(atoms, hessian, "nonlinear")
        nested = projected_vibrations(
            atoms, hessian.reshape(n_atoms, 3, n_atoms, 3), "nonlinear"
        )
        assert wavenumbers(flat) == pytest.approx(wavenumbers(nested), abs=1e-9)

    def test_an_asymmetric_hessian_is_symmetrized_before_diagonalizing(self):
        """A Hessian is symmetric; a finite-difference or fp32 one is only nearly so.

        ``numpy.linalg.eigvalsh`` reads a single triangle, so without an
        explicit symmetrization the spectrum silently depends on which
        triangle LAPACK happens to use -- i.e. on a detail of the caller's
        Hessian layout rather than on the physics.
        """
        _, atoms = _ethanol()
        hessian = hessian_with_spectrum(
            atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 0.05, hessian.shape)
        antisymmetric = noise - noise.T
        clean = sorted(wavenumbers(projected_vibrations(atoms, hessian, "nonlinear")))
        perturbed = sorted(
            wavenumbers(
                projected_vibrations(atoms, hessian + antisymmetric, "nonlinear")
            )
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
        hessian = hessian_with_spectrum(
            atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear"
        )
        light = sorted(wavenumbers(projected_vibrations(atoms, hessian, "nonlinear")))

        heavy_atoms = atoms.copy()
        masses = np.asarray(heavy_atoms.get_masses(), dtype=float)
        masses[[a.index for a in heavy_atoms if a.symbol == "H"]] = 2.0141
        heavy_atoms.set_masses(masses)
        heavy = sorted(
            wavenumbers(projected_vibrations(heavy_atoms, hessian, "nonlinear"))
        )
        assert max(heavy) < max(light) * 0.85, (
            "deuterating every hydrogen did not lower the highest stretch; the "
            "masses are not reaching the mass weighting"
        )
