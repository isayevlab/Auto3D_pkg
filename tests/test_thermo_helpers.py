"""Fast unit tests for thermochemistry helper functions.

These cover the pure-Python helpers in Auto3D.ASE.thermo and do not run any
neural-network potential or thermodynamic calculation, so they stay in the
fast test suite (the main tests/test_thermo.py module is marked slow).

The two AIMNET Hessian-model checks below are marked ``slow``: each requires a
real ~9s NNP load (a separate model from the conftest ``aimnet_model`` adapter,
since ``_load_hessian_model`` returns the bare AIMNet2Calculator). They share a
module-scoped ``aimnet_hessian_model`` fixture so that, in the slow suite, the
model still loads only once instead of twice.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def aimnet_hessian_model():
    """Load the AIMNET Hessian evaluator once for this module's NNP checks."""
    import torch
    from Auto3D.ASE.thermo import _load_hessian_model
    return _load_hessian_model("AIMNET", torch.device("cpu"))


def test_detect_geometry_linear_vs_nonlinear():
    from ase import Atoms
    from Auto3D.ASE.thermo import _detect_geometry
    co2 = Atoms("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    water = Atoms("OH2", [[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]])
    assert _detect_geometry(co2) == "linear"
    assert _detect_geometry(water) == "nonlinear"


def test_symmetry_number_defaults_to_one():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    m = Chem.MolFromSmiles("CCO")
    assert _symmetry_number(m) == 1  # no property -> default 1


def test_symmetry_number_reads_property():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    m = Chem.MolFromSmiles("c1ccccc1")
    m.SetProp("symmetry_number", "12")
    assert _symmetry_number(m) == 12


def test_symmetry_number_invalid_property_falls_back():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    m = Chem.MolFromSmiles("CCO")
    m.SetProp("symmetry_number", "not_a_number")
    assert _symmetry_number(m) == 1


def test_resolve_multiplicity_closed_shell_is_singlet():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _resolve_multiplicity
    m = Chem.MolFromSmiles("CCO")
    assert _resolve_multiplicity(m) == 1
    # Derived multiplicity is recorded on the mol.
    assert m.GetUnsignedProp("multiplicity") == 1


def test_resolve_multiplicity_radical_is_doublet():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _resolve_multiplicity
    m = Chem.MolFromSmiles("[CH3]")  # methyl radical, 1 unpaired electron
    assert _resolve_multiplicity(m) == 2
    assert m.GetUnsignedProp("multiplicity") == 2


def test_resolve_multiplicity_respects_explicit_property():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _resolve_multiplicity
    m = Chem.MolFromSmiles("[CH3]")  # would derive 2 ...
    m.SetUnsignedProp("multiplicity", 4)  # ... but an explicit value wins
    assert _resolve_multiplicity(m) == 4


def test_do_mol_thermo_default_temperature_is_298_15():
    """Reference temperature must be the thermochemistry standard 298.15 K."""
    import inspect
    from Auto3D.ASE.thermo import do_mol_thermo
    assert inspect.signature(do_mol_thermo).parameters["T"].default == 298.15


@pytest.mark.slow
def test_load_hessian_model_aimnet(aimnet_hessian_model):
    m = aimnet_hessian_model
    # An AIMNet2Calculator from the aimnet registry (not a bundled .jpt);
    # vib_hessian routes it through the calculator's full-pipeline analytic Hessian.
    assert m is not None
    assert hasattr(m, "model")  # the calculator wraps the underlying nn.Module


@pytest.mark.slow
def test_load_hessian_model_aimnet_is_fp32(aimnet_hessian_model):
    import torch
    # The underlying aimnet module stays fp32 (no whole-graph fp64 upcast).
    p = next(aimnet_hessian_model.model.parameters())
    assert p.dtype == torch.float32
