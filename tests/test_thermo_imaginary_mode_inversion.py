"""What Auto3D does with an imaginary mode, and why the answer no longer
depends on which ASE is installed.

Three separate decisions, tested here:

1. **A sub-cutoff imaginary mode is inverted, not deleted.** A nonlinear
   molecule has exactly ``3N-6`` vibrational degrees of freedom. Deleting an
   artifact would give a species with one artifact a ``3N-7``-mode partition
   function and a species with none a ``3N-6``-mode one; those two free
   energies are not the same thermodynamic quantity and the difference does
   not cancel in the comparison a user runs thermochemistry to make. Keeping
   the mode at ``|nu|`` is the Gaussian/ORCA convention.
2. **A genuine reaction coordinate is removed by Auto3D, deliberately**, so a
   confirmed saddle point passes ``3N-7`` and the count is a decision rather
   than a side effect of ``ignore_imag_modes``.
3. **A quasi-harmonic floor is applied to the remaining real modes.** The
   harmonic entropy diverges as ``-R*ln(h*nu/kT)``, so G is most sensitive to
   exactly the modes an fp32 NNP Hessian resolves worst. With the floor in
   force, an artifact at 10i, 20i, 30i or 49i all contribute identically.

And the property that was broken: Auto3D hands ``IdealGasThermo`` exactly the
modes it means, so **G is the same under both of ASE's mode-selection rules**.
Before this, the full ``3N`` list was passed and ASE chose; ASE changed that
choice in 3.28.0, and the same input then gave a different Gibbs energy.

Everything here is driven from synthetic Hessians; no neural network potential
is loaded and no model is downloaded.
"""

from __future__ import annotations

import math

import pytest
from ase import units
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import VibrationsData

import Auto3D.entry.ASE.thermo.driver as thermo_mod
from Auto3D.entry.ASE.thermo import vibrations as _vibrations
from Auto3D.entry.ASE.thermo.vibrations import analyze_vibrations, projected_vibrations
from Auto3D.foundation.constants import (
    EV_PER_WAVENUMBER,
    EV_TO_HARTREE,
    IMAGINARY_MODE_CUTOFF_CM,
    LOW_FREQUENCY_CUTOFF_CM,
)
from Auto3D.foundation.utils.energy import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL
from tests.helpers_vibrations import (
    ASE_SELECTION_RULES,
    atoms_for,
    energies_ev,
    fake_vib_for,
    hessian_with_spectrum,
    probe_mol,
    wavenumbers,
)

T_REFERENCE = 298.15
PRESSURE_PA = 101325

#: 9 atoms (ethanol), nonlinear -> 27 Hessian eigenvalues, of which 21 are
#: genuine vibrations. One slot is left for the mode under test.
REAL_VIBRATIONS = (
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
#: Translation/rotation eigenvalues at realistic magnitudes, mixed real and
#: imaginary. A 1.6 cm-1 mode of these is worth about -2.9 kcal/mol of G on its
#: own if it ever reaches the partition function -- which is exactly what ASE
#: >= 3.28's selection did with it.
TRANS_ROT_NOISE = (1.6, -3.2, 3.4, -3.5, 3.7, -4.1)

N_ATOMS = 9
N_VIB = 3 * N_ATOMS - 6


def _spectrum(mode_cm: float):
    """The 21 vibrations, with ``mode_cm`` in the free slot."""
    return [mode_cm, *REAL_VIBRATIONS]


def _harmonic_gibbs_kcal(wavenumber_cm: float, temperature: float = T_REFERENCE):
    """One harmonic mode's contribution to G, in kcal/mol.

    ``G_vib = hv/2 + kB*T*ln(1 - exp(-hv/kB*T))``: the thermal-enthalpy term
    ``hv/(exp(x)-1)`` in H cancels exactly against the ``x/(exp(x)-1)`` term in
    ``-T*S``, leaving the zero-point energy plus the (negative) configurational
    term. Written out here rather than taken from ASE so the expected shift is
    an independent closed form, not the code under test rearranged.
    """
    energy_ev = wavenumber_cm * EV_PER_WAVENUMBER
    x = energy_ev / (units.kB * temperature)
    return (
        0.5 * energy_ev + units.kB * temperature * math.log(1.0 - math.exp(-x))
    ) * EV_TO_KCAL_PER_MOL


def _selection_disabled_kwargs() -> dict:
    """IdealGasThermo kwargs that make it consume a list verbatim, any version."""
    if _vibrations._ASE_HAS_VIB_SELECTION:
        return {"vib_selection": "all"}
    return {"natoms": 0}


def _gibbs_kcal(atoms, modes, potential_energy=0.0) -> float:
    """G in kcal/mol for a given mode list, with ASE's own selection disabled."""
    thermo = IdealGasThermo(
        vib_energies=list(modes),
        potentialenergy=potential_energy,
        atoms=atoms,
        geometry="nonlinear",
        symmetrynumber=1,
        spin=0.0,
        ignore_imag_modes=True,
        **_selection_disabled_kwargs(),
    )
    return (
        thermo.get_gibbs_energy(temperature=T_REFERENCE, pressure=PRESSURE_PA, verbose=False)
        * EV_TO_HARTREE
        * HARTREE_TO_KCAL_PER_MOL
    )


def _run(
    mode_cm: float, *, low_freq_cutoff_cm=LOW_FREQUENCY_CUTOFF_CM, name="probe", monkeypatch=None
):
    """``do_mol_thermo`` on a synthetic Hessian; returns (mol, atoms, modes given to ASE)."""
    mol = probe_mol("CCO", name)
    atoms = atoms_for(mol, potential_energy=0.0)
    vib = fake_vib_for(atoms, _spectrum(mode_cm), TRANS_ROT_NOISE, "nonlinear")
    captured: list[list[complex]] = []
    real_thermo = thermo_mod.IdealGasThermo

    def recording(*args, **kwargs):
        captured.append([complex(v) for v in kwargs["vib_energies"]])
        return real_thermo(*args, **kwargs)

    monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: vib)
    monkeypatch.setattr(thermo_mod, "IdealGasThermo", recording)
    produced = thermo_mod.do_mol_thermo(
        mol,
        atoms,
        adapter=None,
        T=T_REFERENCE,
        low_freq_cutoff_cm=low_freq_cutoff_cm,
    )
    assert len(captured) == 1, "do_mol_thermo built more than one IdealGasThermo"
    return produced, atoms, captured[0]


class TestTheModeCountIsConserved:
    def test_a_sub_cutoff_artifact_is_kept_at_its_absolute_value(self):
        analysis = analyze_vibrations(
            energies_ev(*_spectrum(-20)),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
            low_freq_cutoff_cm=0.0,
        )
        assert analysis.n_imag == 1
        assert analysis.n_inverted == 1
        assert analysis.n_removed == 0
        assert len(analysis.corrected_energies) == N_VIB, (
            "inverting must conserve the 3N-6 mode count"
        )
        assert sorted(wavenumbers(analysis.corrected_energies)) == pytest.approx(
            sorted([20.0, *REAL_VIBRATIONS]), abs=1e-6
        )

    def test_a_reaction_coordinate_is_removed_not_inverted(self):
        analysis = analyze_vibrations(
            energies_ev(*_spectrum(-400)),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
            low_freq_cutoff_cm=0.0,
        )
        assert (analysis.n_imag, analysis.n_inverted, analysis.n_removed) == (1, 0, 1)
        assert analysis.is_transition_state
        assert len(analysis.corrected_energies) == N_VIB - 1, (
            "a confirmed saddle point must pass 3N-7 modes, deliberately"
        )
        assert all(v.imag == 0.0 for v in analysis.corrected_energies), (
            "the reaction coordinate is still in the list ASE is given"
        )

    def test_the_cutoff_boundary_is_read_not_hardcoded(self):
        just_under = analyze_vibrations(
            energies_ev(*_spectrum(-(IMAGINARY_MODE_CUTOFF_CM - 1))),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
        )
        just_over = analyze_vibrations(
            energies_ev(*_spectrum(-(IMAGINARY_MODE_CUTOFF_CM + 1))),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
        )
        assert (just_under.n_inverted, just_under.n_removed) == (1, 0)
        assert (just_over.n_inverted, just_over.n_removed) == (0, 1)

    def test_a_wrong_length_mode_list_is_refused(self):
        """The contract is 'exactly the vibrations', and it is enforced.

        Handing the raw 3N spectrum here is the old behavior; it must not be
        silently re-sliced, because that is the delegation this work removed.
        """
        with pytest.raises(ValueError, match="projected_vibrations"):
            analyze_vibrations(
                energies_ev(*TRANS_ROT_NOISE, *_spectrum(-20)),
                n_atoms=N_ATOMS,
                geometry="nonlinear",
            )


class TestTheQuasiHarmonicFloor:
    def test_every_tolerated_artifact_lands_on_the_floor(self):
        """10i, 20i, 30i and 49i are all evaluated at 100 cm-1.

        Once the floor is in force the exact artifact cutoff stops mattering
        for G, which is the point: the mode's frequency has just been declared
        untrustworthy, so G must not depend on it.
        """
        floored = {}
        for artifact in (10, 20, 30, 49):
            analysis = analyze_vibrations(
                energies_ev(*_spectrum(-artifact)),
                n_atoms=N_ATOMS,
                geometry="nonlinear",
            )
            assert analysis.n_inverted == 1
            assert analysis.n_raised == 1
            floored[artifact] = sorted(wavenumbers(analysis.corrected_energies))
        for artifact, modes in floored.items():
            assert min(modes) == pytest.approx(LOW_FREQUENCY_CUTOFF_CM, abs=1e-6), (
                f"the {artifact}i artifact was not raised to the floor"
            )
        assert len({tuple(round(w, 9) for w in m) for m in floored.values()}) == 1

    def test_a_genuine_soft_real_mode_is_raised_too(self):
        """The floor is not an imaginary-mode special case.

        A real 30 cm-1 torsion is exactly as unreliable as an inverted 30i
        artifact -- ``dG/dnu`` is +0.020 kcal/mol per cm-1 there under plain
        RRHO -- so both get the same treatment.
        """
        analysis = analyze_vibrations(
            energies_ev(*_spectrum(30)),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
        )
        assert (analysis.n_imag, analysis.n_inverted) == (0, 0)
        assert analysis.n_raised == 1
        assert min(wavenumbers(analysis.corrected_energies)) == pytest.approx(
            LOW_FREQUENCY_CUTOFF_CM, abs=1e-6
        )

    def test_the_floor_can_be_switched_off(self):
        analysis = analyze_vibrations(
            energies_ev(*_spectrum(30)),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
            low_freq_cutoff_cm=0.0,
        )
        assert analysis.n_raised == 0
        assert min(wavenumbers(analysis.corrected_energies)) == pytest.approx(30.0, abs=1e-6)
        assert analysis.convention == "RRHO"

    def test_the_convention_names_the_cutoff(self):
        analysis = analyze_vibrations(
            energies_ev(*_spectrum(30)),
            n_atoms=N_ATOMS,
            geometry="nonlinear",
        )
        assert analysis.convention == "RRHO+quasiharmonic(100cm-1)"

    @pytest.mark.parametrize("artifact_cm", [10, 20, 30, 49])
    def test_with_the_floor_off_inverting_lowers_g_by_one_modes_free_energy(
        self, monkeypatch, artifact_cm
    ):
        """The plain-RRHO statement of the inversion, checked against a closed form.

        Deleting the mode removes its whole contribution to G; keeping it at
        ``|nu|`` restores exactly the harmonic free energy of one mode at
        ``|nu|``, which is negative because ``-T*S_vib`` dominates.
        """
        produced, atoms, given = _run(-artifact_cm, low_freq_cutoff_cm=0.0, monkeypatch=monkeypatch)
        g_inverted = float(produced.GetProp("G_hartree")) * HARTREE_TO_KCAL_PER_MOL
        deleted = [v for v in given if abs(v.real - artifact_cm * EV_PER_WAVENUMBER) > 1e-12]
        assert len(deleted) == N_VIB - 1, "test premise: exactly one mode removed"
        g_deleted = _gibbs_kcal(atoms, deleted)

        delta = g_inverted - g_deleted
        assert delta < 0.0, (
            "keeping a low-frequency mode must LOWER G (the -T*S_vib term "
            f"dominates), but G moved by {delta:+.4f} kcal/mol"
        )
        assert delta == pytest.approx(_harmonic_gibbs_kcal(artifact_cm), abs=1e-6)
        assert 0.8 < abs(delta) < 1.9

    @pytest.mark.parametrize("artifact_cm", [10, 20, 30, 49])
    def test_with_the_floor_on_every_artifact_gives_the_same_g(self, monkeypatch, artifact_cm):
        produced, _atoms, _given = _run(-artifact_cm, monkeypatch=monkeypatch)
        g = float(produced.GetProp("G_hartree")) * HARTREE_TO_KCAL_PER_MOL
        reference, _, _ = _run(
            -20,
            name="reference",
            monkeypatch=monkeypatch,
        )
        g_reference = float(reference.GetProp("G_hartree")) * HARTREE_TO_KCAL_PER_MOL
        assert g == pytest.approx(g_reference, abs=1e-9), (
            "G still depends on the exact wavenumber of a mode whose "
            "frequency was declared untrustworthy"
        )


class TestGibbsEnergyDoesNotDependOnAsesSelectionRule:
    """The property the ASE 3.28.0 change broke.

    Both of ASE's selection rules are applied here as pure functions to the
    exact list ``do_mol_thermo`` hands ``IdealGasThermo``, so this runs on
    whichever single ASE version happens to be installed and still covers both.
    Each transcription is quoted from the corresponding ASE source in
    ``tests/helpers_vibrations.py``.
    """

    @pytest.mark.parametrize(
        "mode_cm",
        [120, -10, -20, -30, -49, -400],
        ids=["clean", "10i", "20i", "30i", "49i", "saddle"],
    )
    def test_both_rules_select_the_same_modes_and_the_same_g(self, monkeypatch, mode_cm):
        produced, atoms, given = _run(mode_cm, monkeypatch=monkeypatch)
        g_reported = float(produced.GetProp("G_hartree")) * HARTREE_TO_KCAL_PER_MOL

        for label, rule in ASE_SELECTION_RULES.items():
            selected = rule(given, len(given))
            assert sorted(wavenumbers(selected)) == pytest.approx(
                sorted(wavenumbers(given)), abs=1e-9
            ), f"the {label} rule changed the mode set Auto3D handed ASE"
            assert _gibbs_kcal(atoms, selected) == pytest.approx(g_reported, abs=1e-9), (
                f"G differs under the {label} rule"
            )

    @pytest.mark.parametrize(
        "mode_cm, expect_spread",
        [(120, False), (-20, True), (-400, True)],
        ids=["clean", "artifact", "saddle"],
    )
    def test_the_old_contract_really_did_depend_on_the_rule(self, mode_cm, expect_spread):
        """Non-vacuity: the test above would pass on a list that cannot discriminate.

        Handing ASE the raw ``3N`` spectrum -- what Auto3D used to do -- makes
        the two rules disagree by kcal/mol as soon as any mode is imaginary,
        and agree exactly when none is. Same molecule, same Hessian.
        """
        mol = probe_mol("CCO")
        atoms = atoms_for(mol, potential_energy=0.0)
        hessian = hessian_with_spectrum(atoms, _spectrum(mode_cm), TRANS_ROT_NOISE, "nonlinear")
        raw = VibrationsData(atoms, hessian.reshape(N_ATOMS, 3, N_ATOMS, 3)).get_energies()
        gibbs = {
            label: _gibbs_kcal(atoms, rule(raw, N_VIB))
            for label, rule in ASE_SELECTION_RULES.items()
        }
        spread = max(gibbs.values()) - min(gibbs.values())
        if expect_spread:
            assert spread > 0.5, (
                "test premise: the two rules are supposed to disagree on the "
                f"raw 3N list here, got {gibbs}"
            )
        else:
            assert spread == pytest.approx(0.0, abs=1e-9), (
                f"test premise: no imaginary mode, so the rules agree: {gibbs}"
            )

    def test_a_saddle_points_reaction_coordinate_never_reaches_the_partition_function(
        self, monkeypatch
    ):
        produced, atoms, given = _run(-400, monkeypatch=monkeypatch)
        assert produced.GetProp("Is_transition_state") == "True"
        assert len(given) == N_VIB - 1, "the saddle point was not handed 3N-7 modes"
        assert all(v.imag == 0.0 for v in given)
        assert not any(abs(w - 400.0) < 1.0 for w in wavenumbers(given)), (
            "the reaction coordinate was inverted into the partition function"
        )
        assert produced.GetProp("Thermo_vib_modes") == str(N_VIB - 1)


class TestTheRecordSaysWhatWasDone:
    def test_every_correction_is_counted_in_the_record(self, monkeypatch):
        produced, _atoms, given = _run(-20, monkeypatch=monkeypatch)
        assert produced.GetProp("N_imaginary_modes") == "1"
        assert produced.GetProp("N_inverted_imaginary_modes") == "1"
        assert produced.GetProp("N_raised_modes") == "1"
        assert produced.GetProp("Thermo_vib_modes") == str(len(given)) == str(N_VIB)
        assert produced.GetProp("Thermo_convention") == "RRHO+quasiharmonic(100cm-1)"

    def test_the_convention_property_follows_the_opt_out(self, monkeypatch):
        produced, _atoms, _given = _run(-20, low_freq_cutoff_cm=0.0, monkeypatch=monkeypatch)
        assert produced.GetProp("Thermo_convention") == "RRHO"
        assert produced.GetProp("N_raised_modes") == "0"

    def test_the_log_says_the_mode_was_kept_not_dropped(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            _run(-20, monkeypatch=monkeypatch)
        messages = [record.getMessage() for record in caplog.records]
        assert any("kept at |nu|" in message for message in messages), (
            f"the log does not say the artifact was inverted: {messages}"
        )
        assert not any("removed from the thermochemistry" in m for m in messages), (
            f"the log still claims the artifact was dropped: {messages}"
        )

    def test_the_log_says_a_reaction_coordinate_was_removed(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
            _run(-400, monkeypatch=monkeypatch)
        messages = [record.getMessage() for record in caplog.records]
        assert any("removed from the thermochemistry" in m for m in messages), messages


class TestProjectionFeedsTheAnalysis:
    """The two halves must agree on the mode count, or nothing else holds."""

    def test_the_projected_spectrum_is_exactly_what_analyze_vibrations_wants(self):
        mol = probe_mol("CCO")
        atoms = atoms_for(mol)
        hessian = hessian_with_spectrum(atoms, _spectrum(-20), TRANS_ROT_NOISE, "nonlinear")
        energies = projected_vibrations(atoms, hessian, "nonlinear")
        analysis = analyze_vibrations(
            energies, n_atoms=N_ATOMS, geometry="nonlinear", low_freq_cutoff_cm=0.0
        )
        assert len(analysis.energies) == N_VIB
        assert sorted(wavenumbers(analysis.energies)) == pytest.approx(
            sorted(_spectrum(-20)), abs=1e-6
        )


class TestAseSelectionIsDisabledAtTheCallSite:
    """The kwargs that stop ASE re-selecting, pinned on whichever ASE is here.

    Every other test in this file works because the list Auto3D hands ASE is
    already exactly the right length, which makes both selection rules
    no-ops -- so none of them can tell whether the selection was actually
    disabled. These can: they hand ASE a list that is deliberately NOT
    ``3N-6`` long and check it comes back untouched.
    """

    def test_the_mechanism_named_exists_in_the_installed_ase(self):
        import inspect

        parameters = inspect.signature(IdealGasThermo.__init__).parameters
        kwargs = _vibrations._verbatim_mode_kwargs(N_VIB, N_VIB)
        assert set(kwargs) <= set(parameters), (
            f"_verbatim_mode_kwargs returned {kwargs}, which this ASE "
            f"({sorted(parameters)}) does not accept"
        )
        if _vibrations._ASE_HAS_VIB_SELECTION:
            assert kwargs == {"vib_selection": "exact"}
            assert _vibrations._verbatim_mode_kwargs(N_VIB - 1, N_VIB) == {
                "vib_selection": "all"
            }, "a saddle point's 3N-7 list must not be checked against 3N-6"
        else:
            assert kwargs == {"natoms": 0}
            assert _vibrations._verbatim_mode_kwargs(N_VIB - 1, N_VIB) == {"natoms": 0}

    def test_a_longer_than_3n_minus_6_list_is_consumed_verbatim(self):
        """With the selection disabled, ASE keeps every mode it is given.

        A ``3N-5``-long list is the discriminator: ASE's default selection
        would either trim it to ``3N-6`` (<= 3.27) or refuse it outright
        (>= 3.28), so anything that leaves the selection enabled fails here.
        """
        mol = probe_mol("CCO")
        atoms = atoms_for(mol, potential_energy=0.0)
        too_many = energies_ev(*range(200, 200 + 50 * (N_VIB + 1), 50))
        assert len(too_many) == N_VIB + 1, "test premise"

        thermo = IdealGasThermo(
            vib_energies=too_many,
            potentialenergy=0.0,
            atoms=atoms,
            geometry="nonlinear",
            symmetrynumber=1,
            spin=0.0,
            ignore_imag_modes=True,
            **_vibrations._verbatim_mode_kwargs(len(too_many), N_VIB),
        )
        assert len(thermo.vib_energies) == N_VIB + 1

        # Non-vacuity: ASE's own default really would have changed this list.
        with_default_selection = None
        try:
            with_default_selection = len(
                IdealGasThermo(
                    vib_energies=too_many,
                    potentialenergy=0.0,
                    atoms=atoms,
                    geometry="nonlinear",
                    symmetrynumber=1,
                    spin=0.0,
                    ignore_imag_modes=True,
                ).vib_energies
            )
        except ValueError:
            pass  # >= 3.28 with vib_selection='exact' refuses the wrong length
        assert with_default_selection != N_VIB + 1, (
            "this ASE's default selection is a no-op on a wrong-length list, "
            "so this test cannot discriminate on it"
        )
