"""A sub-cutoff imaginary mode must be kept at |nu|, not deleted.

``IMAGINARY_MODE_CUTOFF_CM = 50`` declares everything below 50 cm-1 a tolerable
numerical artifact of a low-frequency vibration. ASE's ``ignore_imag_modes``
then **removes** those modes from the mode list, which deletes their entire
vibrational partition-function contribution. The standard treatments for a
numerical artifact are to keep the mode at ``|nu|`` (the Gaussian/ORCA
convention) or to damp it quasi-harmonically; deleting it is the one choice
that changes G the most, and the log said only "treat the result as
approximate".

The shift is dominated by the lost ``-T*S_vib`` term, which is large and
negative for a low-frequency mode because ``S_vib`` diverges as ``1/nu``. So
keeping the mode moves G **down**; deleting it moves G **up**, by 0.85 kcal/mol
for a 49 cm-1 artifact and 1.80 kcal/mol for a 10 cm-1 one at 298.15 K. That
bias does not cancel between two species carrying different artifact counts --
which is exactly the comparison a user runs thermochemistry to make.

Everything here is driven from a synthetic mode list; no Hessian is built and
no neural network potential is loaded.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from ase import units
from ase.thermochemistry import IdealGasThermo
from rdkit import Chem
from rdkit.Chem import AllChem

import Auto3D.ASE.thermo as thermo_mod
from Auto3D.ASE.thermo import analyze_vibrations, ev2hatree
from Auto3D.constants import EV_PER_WAVENUMBER, IMAGINARY_MODE_CUTOFF_CM
from Auto3D.utils.chemistry import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL

T_REFERENCE = 298.15
PRESSURE_PA = 101325


def _ev(*wavenumbers_cm):
    """Vibrational energies in eV; a negative wavenumber is an imaginary mode."""
    return [
        complex(0.0, abs(w) * EV_PER_WAVENUMBER)
        if w < 0
        else complex(w * EV_PER_WAVENUMBER, 0.0)
        for w in wavenumbers_cm
    ]


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


#: 9 atoms, nonlinear -> 27 modes, of which 21 are genuine vibrations. The
#: trans/rot six are tiny (some spuriously imaginary, as a real NNP Hessian
#: produces) and are removed by the 3N-6 cut on both sides of every comparison.
_TRANS_ROT = (-8, -5, -3, 2, 4, 6)
_REAL_VIBRATIONS = (
    250, 300, 420, 800, 900, 1000, 1100, 1200, 1300, 1400,
    1450, 1470, 2900, 2950, 3000, 3010, 3020, 3050, 3600, 3700,
)


def _spectrum(artifact_cm: float):
    """3N modes with one imaginary mode at ``artifact_cm`` among the vibrations."""
    modes = _ev(*_TRANS_ROT, -abs(artifact_cm), *_REAL_VIBRATIONS)
    assert len(modes) == 27, "test premise: a full 3N mode set for 9 atoms"
    return modes


def _ethanol(name="probe"):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", name)
    return mol


def _atoms_for(mol, potential_energy=-1234.5):
    from ase import Atoms

    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions)
    atoms.get_calculator = lambda: None
    atoms.get_potential_energy = lambda: potential_energy
    return atoms


class _FakeVib:
    def __init__(self, energies):
        self._energies = energies

    def get_energies(self):
        return list(self._energies)


class TestAnalyzeVibrationsInvertsArtifacts:
    def test_a_sub_cutoff_imaginary_mode_comes_back_at_its_absolute_value(self):
        modes = _spectrum(20)
        analysis = analyze_vibrations(modes, n_atoms=9, geometry="nonlinear")

        assert analysis.n_imag == 1
        assert analysis.n_inverted == 1
        assert len(analysis.corrected_energies) == len(analysis.energies)

        artifact_index = modes.index(_ev(-20)[0])
        original = analysis.energies[artifact_index]
        corrected = analysis.corrected_energies[artifact_index]
        assert original.imag != 0.0, "test premise: the input mode is imaginary"
        assert corrected.imag == 0.0, "the artifact was not made real"
        assert corrected.real == pytest.approx(abs(original)), (
            "the artifact was not kept at |nu|"
        )
        # Inversion preserves the magnitude, so IdealGasThermo's own
        # sort-by-magnitude 3N-6 slice selects exactly the same modes.
        assert [abs(v) for v in analysis.corrected_energies] == pytest.approx(
            [abs(v) for v in analysis.energies]
        )

    def test_a_genuine_reaction_coordinate_is_left_imaginary(self):
        """Above the cutoff it is not an artifact and must not be inverted."""
        modes = _ev(*_TRANS_ROT, -400, *_REAL_VIBRATIONS)
        analysis = analyze_vibrations(modes, n_atoms=9, geometry="nonlinear")

        assert analysis.n_imag == 1
        assert analysis.n_inverted == 0
        assert analysis.corrected_energies == analysis.energies

    def test_the_cutoff_boundary_is_read_not_hardcoded(self):
        """A mode just under the cutoff inverts; one just over it does not."""
        just_under = analyze_vibrations(
            _spectrum(IMAGINARY_MODE_CUTOFF_CM - 1), n_atoms=9, geometry="nonlinear"
        )
        just_over = analyze_vibrations(
            _spectrum(IMAGINARY_MODE_CUTOFF_CM + 1), n_atoms=9, geometry="nonlinear"
        )
        assert (just_under.n_inverted, just_over.n_inverted) == (1, 0)

    def test_a_clean_spectrum_is_returned_unchanged(self):
        modes = _ev(*_TRANS_ROT, 120, *_REAL_VIBRATIONS)
        analysis = analyze_vibrations(modes, n_atoms=9, geometry="nonlinear")
        assert analysis.n_inverted == 0
        assert analysis.corrected_energies == analysis.energies


class TestGibbsEnergyMovesInTheDirectionPhysicsRequires:
    """G with the artifact inverted vs G with it deleted -- the whole point.

    Deleting the mode removes its ``-T*S_vib`` contribution entirely; inverting
    keeps it at ``|nu|``. The two must differ, and specifically the inverted
    value must be **lower**, by exactly the harmonic free energy of one mode at
    ``|nu|``.
    """

    def _gibbs_hartree(self, mol, atoms, energies):
        """G the way ``do_mol_thermo`` computes it, for a given mode list."""
        thermo = IdealGasThermo(
            vib_energies=energies,
            potentialenergy=atoms.get_potential_energy(),
            atoms=atoms,
            geometry="nonlinear",
            symmetrynumber=1,
            spin=0.0,
            ignore_imag_modes=True,
        )
        return (
            thermo.get_gibbs_energy(
                temperature=T_REFERENCE, pressure=PRESSURE_PA, verbose=False
            )
            * ev2hatree
        )

    @pytest.mark.parametrize("artifact_cm", [10, 20, 30, 49])
    def test_inverting_lowers_g_by_one_modes_harmonic_free_energy(
        self, monkeypatch, artifact_cm
    ):
        modes = _spectrum(artifact_cm)
        mol = _ethanol()
        atoms = _atoms_for(mol)
        monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: _FakeVib(modes))

        produced = thermo_mod.do_mol_thermo(
            mol, atoms, model=None, model_name="AIMNET", T=T_REFERENCE
        )
        g_inverted = float(produced.GetProp("G_hartree"))
        # The pre-fix behavior: hand IdealGasThermo the raw list, whose
        # imaginary mode ignore_imag_modes then deletes.
        g_deleted = self._gibbs_hartree(mol, atoms, modes)

        delta_kcal = (g_inverted - g_deleted) * HARTREE_TO_KCAL_PER_MOL
        assert delta_kcal < 0.0, (
            "keeping a low-frequency mode must LOWER G (the -T*S_vib term "
            f"dominates), but G moved by {delta_kcal:+.4f} kcal/mol"
        )
        expected = _harmonic_gibbs_kcal(artifact_cm)
        assert delta_kcal == pytest.approx(expected, abs=1e-6), (
            f"G shifted by {delta_kcal:.4f} kcal/mol, but one harmonic mode at "
            f"{artifact_cm} cm-1 is worth {expected:.4f} kcal/mol"
        )
        # Magnitude, stated as the report states it: this is not a rounding
        # error at any wavenumber the cutoff tolerates.
        assert 0.8 < abs(delta_kcal) < 1.9

    def test_the_inverted_mode_is_actually_kept_in_the_partition_function(self):
        """One more mode survives the clean-up than before, not one fewer."""
        modes = _spectrum(20)
        analysis = analyze_vibrations(modes, n_atoms=9, geometry="nonlinear")
        mol = _ethanol()
        atoms = _atoms_for(mol)

        def n_modes_used(energies):
            thermo = IdealGasThermo(
                vib_energies=energies, potentialenergy=0.0, atoms=atoms,
                geometry="nonlinear", symmetrynumber=1, spin=0.0,
                ignore_imag_modes=True,
            )
            return len(thermo.vib_energies)

        deleted = n_modes_used(analysis.energies)
        inverted = n_modes_used(analysis.corrected_energies)
        assert deleted == 3 * 9 - 6 - 1, (
            f"test premise: the artifact used to be deleted, got {deleted} modes"
        )
        assert inverted == deleted + 1

    def test_a_transition_state_still_loses_its_reaction_coordinate(
        self, monkeypatch
    ):
        """Non-vacuity: the fix must not resurrect a genuine imaginary mode.

        Inverting a -400 cm-1 reaction coordinate would silently turn a saddle
        point into a "minimum" with an extra 400 cm-1 vibration, so G must be
        unchanged from the deleted-mode value here.
        """
        modes = _ev(*_TRANS_ROT, -400, *_REAL_VIBRATIONS)
        mol = _ethanol("saddle")
        atoms = _atoms_for(mol)
        monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: _FakeVib(modes))

        produced = thermo_mod.do_mol_thermo(
            mol, atoms, model=None, model_name="AIMNET", T=T_REFERENCE
        )
        g_produced = float(produced.GetProp("G_hartree"))
        g_deleted = self._gibbs_hartree(mol, atoms, modes)
        assert (g_produced - g_deleted) * HARTREE_TO_KCAL_PER_MOL == pytest.approx(
            0.0, abs=1e-9
        )
        assert produced.GetProp("N_inverted_imaginary_modes") == "0"


class TestTheRecordAndTheLogSayWhatWasDone:
    def test_the_record_counts_the_inverted_modes(self, monkeypatch):
        mol = _ethanol()
        monkeypatch.setattr(
            thermo_mod, "vib_hessian", lambda *a, **k: _FakeVib(_spectrum(20))
        )
        produced = thermo_mod.do_mol_thermo(
            mol, _atoms_for(mol), model=None, model_name="AIMNET"
        )
        assert produced.GetProp("N_imaginary_modes") == "1"
        assert produced.GetProp("N_inverted_imaginary_modes") == "1"

    def test_the_log_says_the_mode_was_kept_not_dropped(self, monkeypatch, caplog):
        import logging

        mol = _ethanol()
        monkeypatch.setattr(
            thermo_mod, "vib_hessian", lambda *a, **k: _FakeVib(_spectrum(20))
        )
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            thermo_mod.do_mol_thermo(
                mol, _atoms_for(mol), model=None, model_name="AIMNET"
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any("kept at |nu|" in message for message in messages), (
            f"the log does not say the artifact was inverted: {messages}"
        )
        assert not any(
            "they are dropped from the thermochemistry" in message
            for message in messages
        ), f"the log still claims the artifact was dropped: {messages}"
