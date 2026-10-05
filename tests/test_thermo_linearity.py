"""The thermochemistry uses the geometry the Hessian supports, and says which.

N-M4: a bent stationary point whose O-C-O angle sits inside the linearity window
used to be projected as linear. The sixth external direction survived as a
~0.05 cm-1 phantom vibration, the quasi-harmonic floor raised it to 100 cm-1
(``N_raised_modes=1``), and the rotational partition function was the linear
one. These tests drive ``do_mol_thermo`` on synthetic Hessians through the
same seams ``tests/test_thermo_imaginary_mode_inversion.py`` uses; no model is
loaded.
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from Auto3D.entry.ASE.thermo.properties import _detect_geometry
from Auto3D.entry.ASE.thermo.vibrations import BENT_RECLASSIFIED
from tests.helpers_vibrations import (
    atoms_for,
    co2_atoms,
    drive_do_mol_thermo,
    fake_vib_for,
    probe_mol,
)
from tests.test_thermo_projection import BENT_CO2, LINEAR_CO2, REAL_MODES, TRANS_ROT_NOISE


def _co2_mol(angle_deg: float) -> Chem.Mol:
    """O=C=O (atom order O, C, O, matching ``co2_atoms``) at the given angle."""
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C=O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conformer = mol.GetConformer()
    for index, xyz in enumerate(co2_atoms(angle_deg).get_positions()):
        conformer.SetAtomPosition(index, Point3D(*map(float, xyz)))
    mol.SetProp("_Name", f"co2_{angle_deg:g}")
    return mol


def test_a_bent_stationary_point_is_reported_as_nonlinear_thermochemistry(monkeypatch, caplog):
    mol = _co2_mol(170.0)
    atoms = atoms_for(mol, potential_energy=0.0)
    assert _detect_geometry(atoms) == "linear", "test premise: inside the window"
    vib = fake_vib_for(atoms, *BENT_CO2)
    with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
        produced, calls = drive_do_mol_thermo(mol, atoms, vib, monkeypatch)
    assert produced.GetProp("Thermo_linearity") == BENT_RECLASSIFIED
    assert produced.GetProp("Thermo_vib_modes") == "3"
    assert produced.GetProp("N_raised_modes") == "0", (
        "M4's symptom: the phantom raised to the floor"
    )
    assert produced.GetProp("Thermo_failed") == ""
    assert calls[0]["geometry"] == "nonlinear"
    assert any("bent stationary point" in r.getMessage() for r in caplog.records)


def test_a_linear_molecule_is_reported_linear(monkeypatch):
    mol = _co2_mol(180.0)
    atoms = atoms_for(mol, potential_energy=0.0)
    vib = fake_vib_for(atoms, *LINEAR_CO2)
    produced, calls = drive_do_mol_thermo(mol, atoms, vib, monkeypatch)
    assert produced.GetProp("Thermo_linearity") == "linear"
    assert produced.GetProp("Thermo_vib_modes") == "4"
    assert calls[0]["geometry"] == "linear"


def test_a_nonlinear_molecule_is_reported_nonlinear(monkeypatch):
    mol = probe_mol("CCO")
    atoms = atoms_for(mol, potential_energy=0.0)
    vib = fake_vib_for(atoms, [120, *REAL_MODES], TRANS_ROT_NOISE, "nonlinear")
    produced, calls = drive_do_mol_thermo(mol, atoms, vib, monkeypatch)
    assert produced.GetProp("Thermo_linearity") == "nonlinear"
    assert calls[0]["geometry"] == "nonlinear"
