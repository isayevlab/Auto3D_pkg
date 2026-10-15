"""Every thermo record states the conventions that produced its G (N-M5, m17, D7).

sigma, the standard state and the mass convention are modeling choices that do
not cancel between species, so a record must carry them; ``Thermo_convention``
names all three in one string and ``Symmetry_number`` / ``Thermo_standard_state``
carry the two that a consumer filters on. Driven through ``do_mol_thermo`` on
synthetic Hessians; no model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import ase
import pytest
from rdkit import Chem

from Auto3D.entry.ASE.thermo.calculator import mol2atoms
from Auto3D.entry.ASE.thermo.properties import (
    CALLER_SUPPLIED_MASSES,
    ISOTOPE_LABELED_MASSES,
    MOST_ABUNDANT_MASSES,
    STANDARD_ATOMIC_WEIGHTS,
    mass_convention,
)
from Auto3D.foundation.constants import EV_PER_WAVENUMBER, STANDARD_PRESSURE, STANDARD_STATE_LABEL
from tests.helpers_vibrations import atoms_for, drive_do_mol_thermo, fake_vib_for, probe_mol
from tests.test_thermo_projection import REAL_MODES, TRANS_ROT_NOISE

FULL_CONVENTION = "RRHO+quasiharmonic(100cm-1); 1 atm; most-abundant-isotope masses"


def _ethanol_run(monkeypatch, mol=None, atoms=None, **kwargs):
    mol = mol if mol is not None else probe_mol("CCO")
    if atoms is None:
        atoms = atoms_for(mol, potential_energy=0.0)
    else:
        # atoms_for stubs get_potential_energy itself; a caller-supplied
        # (e.g. bare mol2atoms(mol)) Atoms carries no calculator at all, so
        # IdealGasThermo's potentialenergy=atoms.get_potential_energy() would
        # otherwise raise, unrelated to whatever this test is checking.
        atoms.get_potential_energy = lambda: 0.0
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
    assert mass_convention(mol, mol2atoms(mol)) == ISOTOPE_LABELED_MASSES
    labeled_masses = mol2atoms(mol).get_masses()
    assert labeled_masses[0] != plain_masses[0], "test premise: mol2atoms honors the label"
    assert (labeled_masses[1:] == plain_masses[1:]).all()
    produced, _ = _ethanol_run(monkeypatch, mol=mol)
    assert produced.GetProp("Thermo_convention").endswith(f"; {ISOTOPE_LABELED_MASSES}")


def test_an_unlabeled_molecule_uses_most_abundant_masses():
    assert mass_convention(probe_mol("CCO"), mol2atoms(probe_mol("CCO"))) == MOST_ABUNDANT_MASSES


def _standard_weight_atoms_from_rdkit(mol: Chem.Mol) -> ase.Atoms:
    """Masses copied from RDKit's three-decimal ``GetAtomicWeight`` table."""
    atoms = mol2atoms(mol)
    periodic_table = Chem.GetPeriodicTable()
    atoms.set_masses([periodic_table.GetAtomicWeight(int(z)) for z in atoms.numbers])
    return atoms


def _standard_weight_atoms_from_ase_defaults(mol: Chem.Mol) -> ase.Atoms:
    """A bare ``ase.Atoms(symbols, positions)``, never touched by ``mol2atoms``.

    This is what a caller outside ``calc_thermo`` actually gets -- from
    ``ase.io.read`` or from building ``Atoms`` by hand -- and it carries
    ASE's own per-element table (``ase.data.atomic_masses``), not RDKit's.
    """
    return ase.Atoms(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )


@pytest.mark.parametrize(
    "build_atoms",
    [_standard_weight_atoms_from_rdkit, _standard_weight_atoms_from_ase_defaults],
    ids=["rdkit-table", "ase-defaults"],
)
def test_standard_atomic_weights_are_recognized_from_either_table(monkeypatch, build_atoms):
    """Chlorine's standard weight differs between the two tables (RDKit
    35.453, ASE 35.45) by more than the match tolerance, so a molecule built
    from either one must still be named ``standard atomic weights`` -- not
    just the table ``mass_convention`` happened to be checked against.

    Ethanol has no chlorine, so this uses a chloroethane probe instead of
    ``_ethanol_run``/``REAL_MODES`` (sized for ethanol's 9 atoms and 21
    modes): chloroethane has 8 atoms (nonlinear, 3N-6 = 18 modes), so its own
    mode list is built here rather than forcing a mismatched atom count
    through the ethanol-shaped fixture.
    """
    mol = probe_mol("CCCl")
    atoms = build_atoms(mol)
    atoms.get_potential_energy = lambda: 0.0
    assert mass_convention(mol, atoms) == STANDARD_ATOMIC_WEIGHTS
    modes = (120, *REAL_MODES[:17])  # 18 modes: 3*8 - 6 for this 8-atom probe
    vib = fake_vib_for(atoms, modes, TRANS_ROT_NOISE, "nonlinear")
    produced, _ = drive_do_mol_thermo(mol, atoms, vib, monkeypatch)
    assert produced.GetProp("Thermo_convention").endswith(f"; {STANDARD_ATOMIC_WEIGHTS}")


def test_masses_matching_no_convention_are_named_caller_supplied():
    mol = probe_mol("CCO")
    atoms = mol2atoms(mol)
    masses = atoms.get_masses()
    masses[0] += 0.5
    atoms.set_masses(masses)
    assert mass_convention(mol, atoms) == CALLER_SUPPLIED_MASSES


def test_the_record_carries_the_zero_point_energy_ase_used(monkeypatch):
    from ase.thermochemistry import IdealGasThermo

    from Auto3D.foundation.constants import EV_TO_HARTREE

    produced, calls = _ethanol_run(monkeypatch)
    rebuilt = IdealGasThermo(**calls[0])
    assert float(produced.GetProp("ZPE_hartree")) == pytest.approx(
        rebuilt.get_ZPE_correction() * EV_TO_HARTREE, rel=1e-12
    )
    # Independent of IdealGasThermo entirely: ZPE = 1/2 sum(h*c*nu) over the
    # fake's own 21-mode list ([120, *REAL_MODES]), converted by hand. This
    # catches a wrong conversion direction that the rebuild above cannot,
    # since that rebuild shares whatever conversion the production code used.
    modes_cm = (120, *REAL_MODES)
    assert float(produced.GetProp("ZPE_hartree")) == pytest.approx(
        0.5 * sum(modes_cm) * EV_PER_WAVENUMBER * EV_TO_HARTREE, rel=1e-9
    )


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
        "**ZPE_hartree**",
        "**Thermo_convention**",
        "**Thermo_standard_state**",
        "**Symmetry_number**",
        "**Thermo_linearity**",
        "**multiplicity**",
        "``symmetry_number`` SD property",
        "RT ln sigma",
    ):
        assert needle in text, needle
