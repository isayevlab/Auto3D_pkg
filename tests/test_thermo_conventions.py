"""Every thermo record states the conventions that produced its G (N-M5, m17, D7).

sigma, the standard state and the mass convention are modeling choices that do
not cancel between species, so a record must carry them; ``Thermo_convention``
names all three in one string and ``Symmetry_number`` / ``Thermo_standard_state``
carry the two that a consumer filters on. Driven through ``do_mol_thermo`` on
synthetic Hessians; no model is loaded.
"""

from __future__ import annotations

from pathlib import Path

from Auto3D.entry.ASE.thermo.calculator import mol2atoms
from Auto3D.entry.ASE.thermo.properties import (
    ISOTOPE_LABELED_MASSES,
    MOST_ABUNDANT_MASSES,
    mass_convention,
)
from Auto3D.foundation.constants import STANDARD_PRESSURE, STANDARD_STATE_LABEL
from tests.helpers_vibrations import atoms_for, drive_do_mol_thermo, fake_vib_for, probe_mol
from tests.test_thermo_projection import REAL_MODES, TRANS_ROT_NOISE

FULL_CONVENTION = "RRHO+quasiharmonic(100cm-1); 1 atm; most-abundant-isotope masses"


def _ethanol_run(monkeypatch, mol=None, **kwargs):
    mol = mol if mol is not None else probe_mol("CCO")
    atoms = atoms_for(mol, potential_energy=0.0)
    vib = fake_vib_for(atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
    return drive_do_mol_thermo(mol, atoms, vib, monkeypatch, **kwargs)


def test_every_thermo_record_states_its_conventions(monkeypatch):
    produced, calls = _ethanol_run(monkeypatch)
    assert produced.GetProp("Symmetry_number") == "1"
    assert produced.GetProp("Thermo_standard_state") == "1 atm"
    assert produced.GetProp("Thermo_convention") == FULL_CONVENTION
    assert produced.GetProp("Thermo_linearity") == "nonlinear"
    assert calls[0]["symmetrynumber"] == 1


def test_the_convention_follows_the_floor_opt_out(monkeypatch):
    produced, _ = _ethanol_run(monkeypatch, low_freq_cutoff_cm=0.0)
    assert produced.GetProp("Thermo_convention") == "RRHO; 1 atm; most-abundant-isotope masses"


def test_a_supplied_symmetry_number_is_echoed_as_the_value_used(monkeypatch):
    mol = probe_mol("CCO")
    mol.SetProp("symmetry_number", "12")
    produced, calls = _ethanol_run(monkeypatch, mol=mol)
    assert produced.GetProp("Symmetry_number") == "12"
    assert calls[0]["symmetrynumber"] == 12


def test_an_invalid_symmetry_number_echoes_the_fallback_not_the_request(monkeypatch):
    mol = probe_mol("CCO")
    mol.SetProp("symmetry_number", "0")
    produced, calls = _ethanol_run(monkeypatch, mol=mol)
    assert produced.GetProp("Symmetry_number") == "1"
    assert calls[0]["symmetrynumber"] == 1


def test_an_isotope_label_changes_the_mass_token_and_the_masses_agree(monkeypatch):
    mol = probe_mol("CCO")
    plain_masses = mol2atoms(mol).get_masses().copy()
    mol.GetAtomWithIdx(0).SetIsotope(13)
    assert mass_convention(mol) == ISOTOPE_LABELED_MASSES
    labeled_masses = mol2atoms(mol).get_masses()
    assert labeled_masses[0] != plain_masses[0], "test premise: mol2atoms honors the label"
    assert (labeled_masses[1:] == plain_masses[1:]).all()
    produced, _ = _ethanol_run(monkeypatch, mol=mol)
    assert produced.GetProp("Thermo_convention").endswith(f"; {ISOTOPE_LABELED_MASSES}")


def test_an_unlabeled_molecule_uses_most_abundant_masses():
    assert mass_convention(probe_mol("CCO")) == MOST_ABUNDANT_MASSES


def test_the_standard_state_label_matches_the_pressure_constant():
    assert STANDARD_PRESSURE == 101325
    assert STANDARD_STATE_LABEL == "1 atm"


def test_usage_rst_documents_the_thermo_conventions():
    """The usage guide must carry a bullet per property, not just the words.

    No skip guard: every CI job checks the repo out and runs pytest from its
    root, and the conda package's test phase never runs pytest at all, so a
    missing-docs skip could only ever hide a real deletion. The needles are the
    bold bullet labels rather than bare substrings for the same reason -- the
    bare names appear in the sigma paragraph, so a test on those would survive
    the deletion of the whole output list.
    """
    text = (Path(__file__).resolve().parents[1] / "docs" / "source" / "usage.rst").read_text()
    for needle in (
        "**Thermo_convention**",
        "**Thermo_standard_state**",
        "**Symmetry_number**",
        "**Thermo_linearity**",
        "**multiplicity**",
        "``symmetry_number`` SD property",
        "RT ln sigma",
    ):
        assert needle in text, needle
