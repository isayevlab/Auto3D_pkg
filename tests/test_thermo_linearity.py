"""The thermochemistry uses the geometry the Hessian supports, and says which.

N-M4: a bent stationary point whose O-C-O angle sits inside the linearity window
used to be projected as linear. The sixth external direction survived as a
near-zero phantom vibration (0.05 cm-1 in these exact synthetic fixtures; up to
``fmax * sum|o_perp| / I_n``, about 9-28 cm-1, at Auto3D's real force gate), the
quasi-harmonic floor raised it to 100 cm-1 (``N_raised_modes=1``), and the
rotational partition function was the linear one. R27 then bounds the other half
of the fix: the nonlinear ROTOR is used only above the classical-rotor floor,
since below it ASE's classical form is invalid and the linear rotor is the
quantum limit. These tests drive ``do_mol_thermo`` on synthetic Hessians through
the same seams ``tests/test_thermo_imaginary_mode_inversion.py`` uses; no model
is loaded except in the one test marked slow.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from Auto3D.entry.ASE.thermo import calc_thermo
from Auto3D.entry.ASE.thermo.properties import _detect_geometry
from Auto3D.entry.ASE.thermo.vibrations import BENT_QUASILINEAR, BENT_RECLASSIFIED
from tests.helpers_vibrations import (
    FakeVib,
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


def test_a_quasilinear_stationary_point_keeps_the_linear_rotor_on_the_record(monkeypatch, caplog):
    """R27 at the record level: 178 degrees, I_min below the classical floor.

    The phantom is dropped (3 modes, so ``N_raised_modes`` stays 0) but the
    rotational partition function stays linear, because the classical nonlinear
    rotor is invalid at ``Theta_A / T = 22.7`` -- it would put ``q_A`` below the
    quantum ground state and cost +0.6 kcal/mol of spurious ``-T*S_rot``. The
    3N-6 list against a linear geometry is a deliberate count mismatch, so
    ``_verbatim_mode_kwargs`` disables ASE's own selection exactly as it does
    for a 3N-7 saddle point.
    """
    mol = _co2_mol(178.0)
    atoms = atoms_for(mol, potential_energy=0.0)
    assert _detect_geometry(atoms) == "linear", "test premise: inside the window"
    vib = fake_vib_for(atoms, *BENT_CO2)
    with caplog.at_level(logging.WARNING, logger="Auto3D.entry.ASE.thermo"):
        produced, calls = drive_do_mol_thermo(mol, atoms, vib, monkeypatch)
    assert produced.GetProp("Thermo_linearity") == BENT_QUASILINEAR
    assert produced.GetProp("Thermo_vib_modes") == "3"
    assert produced.GetProp("N_raised_modes") == "0"
    assert produced.GetProp("Thermo_failed") == ""
    assert calls[0]["geometry"] == "linear", "the rotor must stay linear below the floor"
    assert calls[0].get("vib_selection", "all") == "all", (
        "a 3N-6 list against a linear geometry must disable ASE's selection"
    )
    assert any("linear rotor is kept" in r.getMessage() for r in caplog.records)


def test_a_monatomic_species_is_reported_monatomic(monkeypatch):
    """The fourth ``Thermo_linearity`` value, pinned on a record.

    A single atom has no vibrational degrees of freedom, so ``project_vibrations``
    returns early and the label comes straight from ``_detect_geometry``.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("[Ar]"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", "argon")
    atoms = atoms_for(mol, potential_energy=0.0)
    assert _detect_geometry(atoms) == "monatomic", "test premise"
    vib = FakeVib(np.zeros((3, 3)))
    produced, calls = drive_do_mol_thermo(mol, atoms, vib, monkeypatch)
    assert produced.GetProp("Thermo_linearity") == "monatomic"
    assert produced.GetProp("Thermo_vib_modes") == "0"
    assert calls[0]["geometry"] == "monatomic"


@pytest.mark.slow
def test_a_real_model_calls_co2_and_hcn_linear(tmp_path):
    """The one guard against a false-positive reclassification in production.

    Every other test here uses a synthetic fp64 Hessian whose noise floor is
    ~1e-16; a production AIMNet2 analytic Hessian's is ~1e-7. If a real Hessian
    at a converged near-linear geometry put noise rather than bend curvature on
    the near-axis direction, the ratio would collapse and a genuinely linear
    molecule would be handed the nonlinear rotor. Nothing else in either tier
    drives a real NNP Hessian through a linear molecule. ``opt_tol`` is left
    at its default, Auto3D's 2e-4 eV/A force gate, so the real-model run
    converges on the same stationary-point criterion production uses.
    """
    for smiles, name in (("O=C=O", "co2"), ("C#N", "hcn")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", name)
        path = str(tmp_path / f"{name}.sdf")
        with Chem.SDWriter(path) as writer:
            writer.write(mol)

        out = calc_thermo(path, "AIMNET", use_gpu=False)
        produced = next(Chem.SDMolSupplier(out, removeHs=False))
        assert produced.GetProp("Thermo_linearity") == "linear", (
            f"{name} was not treated as linear; the Hessian test is reading "
            "noise on the near-axis direction"
        )
        assert produced.GetProp("Thermo_vib_modes") == "4", (
            f"{name} is a linear triatomic: 3N-5 = 4 modes"
        )
