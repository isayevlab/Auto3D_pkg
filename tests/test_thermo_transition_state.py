"""A transition state must not pass the documented thermochemistry success filter.

``analyze_vibrations`` already identifies a first-order saddle point and records
``Is_transition_state``, but the record still received ``G_hartree``, was
appended to ``out_mols``, and was stamped ``Thermo_failed = ""`` -- the property
CHANGELOG.md and docs/source/migration-4.0.rst document as *the* success filter.
A saddle point was therefore indistinguishable from a minimum to every
documented way of reading the output.

The rigid-rotor/harmonic partition function assumes a **minimum**: at a saddle
point one mode has no restoring force, the reaction coordinate is deleted
outright, and the resulting "free energy" is a different quantity from every
other record's. The numbers are still written -- a deliberate transition-state
calculation wants them -- but the record now carries
``Thermo_failed = "transition_state"``, so the documented filter rejects it.

Nothing here loads a neural network potential: ``vib_hessian`` is replaced by a
fake returning a canned spectrum, and the model/calculator construction is
monkeypatched exactly as ``test_thermo_helpers`` already does.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import Auto3D.ASE.thermo as thermo_mod
from Auto3D.ASE.thermo import THERMO_FAILED_PROP, TRANSITION_STATE_FAILURE

EV_PER_CM = 1.0 / 8065.54429


def _ev(*wavenumbers_cm):
    """Vibrational energies in eV; a negative wavenumber is an imaginary mode."""
    return [
        complex(0.0, abs(w) * EV_PER_CM) if w < 0 else complex(w * EV_PER_CM, 0.0)
        for w in wavenumbers_cm
    ]


#: Water: 3 atoms, nonlinear, so 9 modes of which 3 are genuine vibrations.
#: The trans/rot six are tiny (and deliberately include the small spurious
#: imaginary values a real NNP Hessian produces), so the 3N-6 cut keeps exactly
#: the last three named here.
_TRANS_ROT = (-8, -5, -3, 2, 4, 6)
SADDLE_SPECTRUM = _ev(*_TRANS_ROT, -400, 1600, 3700)
MINIMUM_SPECTRUM = _ev(*_TRANS_ROT, 1595, 3657, 3756)


def _water(name):
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
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


class TestDoMolThermoMarksASaddlePoint:
    """The verdict is written where it is decided, not left to the caller."""

    def test_a_saddle_point_is_marked_as_a_failure(self, monkeypatch):
        mol = _water("saddle")
        monkeypatch.setattr(
            thermo_mod, "vib_hessian",
            lambda *a, **k: _FakeVib(SADDLE_SPECTRUM),
        )

        result = thermo_mod.do_mol_thermo(
            mol, _atoms_for(mol), model=None, model_name="AIMNET"
        )

        assert result.GetProp("Is_transition_state") == "True"
        assert result.GetProp(THERMO_FAILED_PROP) == TRANSITION_STATE_FAILURE
        # Marked, not silently emptied: a deliberate TS calculation still has
        # its numbers, and the mode that makes it a saddle point is named.
        assert result.HasProp("G_hartree")
        assert result.GetProp("N_imaginary_modes") == "1"
        assert float(result.GetProp("Max_imaginary_mode_cm-1")) == pytest.approx(
            400.0, abs=1.0
        )

    def test_a_minimum_is_marked_as_a_success(self, monkeypatch):
        """Non-vacuity: the marker must discriminate, not fail everything."""
        mol = _water("minimum")
        monkeypatch.setattr(
            thermo_mod, "vib_hessian",
            lambda *a, **k: _FakeVib(MINIMUM_SPECTRUM),
        )

        result = thermo_mod.do_mol_thermo(
            mol, _atoms_for(mol), model=None, model_name="AIMNET"
        )

        assert result.GetProp("Is_transition_state") == "False"
        assert result.GetProp(THERMO_FAILED_PROP) == ""
        assert result.HasProp("G_hartree")


class TestTheWriterCannotEraseTheVerdict:
    """Even mis-routed, a saddle point must not read as a success.

    ``_write_thermo_output`` used to stamp ``Thermo_failed = ""`` over every
    ``out_mols`` record unconditionally, so the guarantee depended entirely on
    ``calc_thermo`` routing correctly. Feeding the saddle point in as a
    "success" here pins the guarantee independently of the routing branch.
    """

    def _written_records(self, tmp_path, out_mols, mols_failed):
        outpath = tmp_path / "out.sdf"
        thermo_mod._write_thermo_output(
            outpath, out_mols=out_mols, mols_failed=mols_failed
        )
        records = [m for m in Chem.SDMolSupplier(str(outpath)) if m is not None]
        assert len(records) == len(out_mols) + len(mols_failed), (
            "a record was lost on the SDF round trip"
        )
        return records

    def test_a_saddle_point_routed_as_a_success_still_fails_the_filter(
        self, monkeypatch, tmp_path
    ):
        mol = _water("saddle")
        monkeypatch.setattr(
            thermo_mod, "vib_hessian",
            lambda *a, **k: _FakeVib(SADDLE_SPECTRUM),
        )
        saddle = thermo_mod.do_mol_thermo(
            mol, _atoms_for(mol), model=None, model_name="AIMNET"
        )

        records = self._written_records(tmp_path, [saddle], [])
        kept = [m for m in records if m.GetProp(THERMO_FAILED_PROP) == ""]
        assert kept == [], (
            "the documented success filter accepted a transition state: "
            f"{[m.GetProp(THERMO_FAILED_PROP) for m in records]}"
        )
        assert records[0].GetProp(THERMO_FAILED_PROP) == TRANSITION_STATE_FAILURE

    def test_a_record_with_no_marker_still_gets_the_success_marker(self, tmp_path):
        """The historical behavior for anything that never set the property."""
        records = self._written_records(tmp_path, [_water("plain")], [])
        assert records[0].GetProp(THERMO_FAILED_PROP) == ""


class TestCalcThermoRoutesASaddlePointToTheFailures:
    """The whole call, one saddle point and one minimum, no NNP.

    ``create_model`` and ``_load_hessian_model`` are monkeypatched the way
    ``test_thermo_helpers.TestCalculatorDeviceAndDtypeFollowTheCaller`` does, so
    ``calc_thermo`` runs end to end on the CPU with a stub that returns zero
    forces (the fmax pre-check therefore passes and the record goes straight to
    ``do_mol_thermo``).
    """

    @staticmethod
    def _zero_force_model():
        import torch
        from torch import nn

        class _StubNNP(nn.Module):
            def forward(self, coords, species, charges, atom_mask=None):
                energy = torch.zeros(coords.shape[0], dtype=coords.dtype)
                return energy, torch.zeros_like(coords).detach()

        return _StubNNP()

    def test_the_documented_filter_selects_the_minimum_and_not_the_saddle(
        self, monkeypatch, tmp_path
    ):
        spectra = {
            "saddle": SADDLE_SPECTRUM,
            "minimum": MINIMUM_SPECTRUM,
        }

        def fake_vib_hessian(mol, *a, **k):
            return _FakeVib(spectra[mol.GetProp("_Name")])

        monkeypatch.setattr(
            thermo_mod, "create_model", lambda *a, **k: self._zero_force_model()
        )
        monkeypatch.setattr(
            thermo_mod, "_load_hessian_model", lambda *a, **k: object()
        )
        monkeypatch.setattr(thermo_mod, "vib_hessian", fake_vib_hessian)

        sdf = tmp_path / "in.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            for name in ("saddle", "minimum"):
                writer.write(_water(name))
        out = tmp_path / "out.sdf"

        thermo_mod.calc_thermo(
            str(sdf), "AIMNET", use_gpu=False, out_path=str(out)
        )

        records = [m for m in Chem.SDMolSupplier(str(out)) if m is not None]
        assert {m.GetProp("_Name") for m in records} == {"saddle", "minimum"}, (
            "a record was dropped instead of being marked"
        )
        kept = [m for m in records if m.GetProp(THERMO_FAILED_PROP) == ""]
        assert [m.GetProp("_Name") for m in kept] == ["minimum"], (
            "the documented success filter did not separate the saddle point "
            "from the minimum: "
            f"{[(m.GetProp('_Name'), m.GetProp(THERMO_FAILED_PROP)) for m in records]}"
        )
        by_name = {m.GetProp("_Name"): m for m in records}
        assert (
            by_name["saddle"].GetProp(THERMO_FAILED_PROP)
            == TRANSITION_STATE_FAILURE
        )
        assert by_name["saddle"].GetProp("Is_transition_state") == "True"
        assert by_name["minimum"].GetProp("Is_transition_state") == "False"
