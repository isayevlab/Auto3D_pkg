"""A non-positive temperature is refused with a clear error, not a ZeroDivisionError
from the classical-rotor floor (WS5 carry-over)."""

import pytest

from Auto3D.foundation.exceptions import ConfigurationError


def test_do_mol_thermo_refuses_a_non_positive_temperature():
    from rdkit import Chem

    from Auto3D.entry.ASE.thermo.driver import do_mol_thermo

    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    with pytest.raises(ConfigurationError, match="temperature"):
        do_mol_thermo(mol, atoms=None, adapter=None, T=0.0)


def test_project_vibrations_refuses_a_non_positive_temperature():
    import numpy as np
    from ase import Atoms

    from Auto3D.entry.ASE.thermo.vibrations import project_vibrations

    atoms = Atoms("CO2", positions=[[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])
    with pytest.raises(ConfigurationError, match="temperature"):
        project_vibrations(atoms, np.eye(9), "linear", temperature_k=-5.0)


def test_calc_thermo_marks_a_non_positive_mol_info_func_temperature_as_failed(
    tmp_path, monkeypatch
):
    """``calc_thermo``'s own docstring: a ``mol_info_func`` returning a
    non-positive ``T`` is "checked before the relaxation is spent, not
    raised" -- the record is marked ``Thermo_failed`` and continues, rather
    than ``calc_thermo`` (or the whole run) raising ``ConfigurationError``.

    Same stub-NNP-plus-monkeypatched-``create_model`` pattern
    ``tests.test_thermo_record_assertions.test_calc_thermo_marks_implicit_hydrogen_records_as_failed``
    uses: the bad ``T`` is caught at the ``mol_info_func`` unpack, before the
    forward pass, so the stub never needs to behave realistically and no
    model is loaded.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch import nn

    import Auto3D.entry.ASE.thermo.driver as thermo_mod
    from Auto3D.entry.ASE.thermo import calculator as _calculator
    from tests.helpers_adapter import AdapterModuleMixin

    class _StubNNP(AdapterModuleMixin, nn.Module):
        def forward(self, coords, species, charges, atom_mask=None):
            import torch

            energy = torch.zeros(coords.shape[0], dtype=coords.dtype)
            forces = torch.zeros_like(coords)
            return energy, forces

    stub = _StubNNP()
    monkeypatch.setattr(thermo_mod, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(_calculator, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(thermo_mod, "_load_hessian_model", lambda *a, **k: object())

    # The pre-check must mark the record before do_mol_thermo is reached; if the
    # pre-check were removed, do_mol_thermo's own ConfigurationError would be
    # caught by the per-record handler and the output would look the same, so
    # this spy is what makes the test tell the two paths apart.
    def _never_called(*args, **kwargs):
        raise AssertionError("do_mol_thermo must not run for a non-positive T")

    monkeypatch.setattr(thermo_mod, "do_mol_thermo", _never_called)

    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol.SetProp("_Name", "m1")
    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)
    out = tmp_path / "out.sdf"

    thermo_mod.calc_thermo(
        str(sdf),
        "AIMNET",
        mol_info_func=lambda m: ("m1", 0.0),
        use_gpu=False,
        out_path=str(out),
    )

    results = list(Chem.SDMolSupplier(str(out), removeHs=False))
    assert len(results) == 1, "the record must be present in the output, not dropped"
    assert results[0].GetProp("Thermo_failed") == "ConfigurationError"
    assert not results[0].HasProp("G_hartree")
