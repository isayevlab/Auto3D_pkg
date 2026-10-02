"""The slow tier's thermochemistry assertions, checked without a potential.

``tests/test_thermo.assert_thermo_record`` is what the NNP thermochemistry tests
compare their output against. It runs only in the slow tier -- a CI-only job of
several minutes -- so nothing would notice if it stopped constraining anything:
loosen its ``abs=1e-9`` to a relative tolerance, or drop the entropy comparison,
and every slow test keeps passing while the guarantee quietly disappears.

The assertions are pure arithmetic over four SD properties, so they can be
exercised here in milliseconds against records built by hand. No model is loaded
and no geometry is optimized; what is under test is the checking, not the
chemistry.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from tests.test_thermo import (
    REFERENCE_G_HARTREE,
    REFERENCE_H_HARTREE,
    REFERENCE_S_HARTREE_PER_K,
    REFERENCE_T_K,
    assert_thermo_record,
)

_H = REFERENCE_H_HARTREE
_S = REFERENCE_S_HARTREE_PER_K
_T = REFERENCE_T_K


def _record(G: float, H: float, S: float, T: float = _T) -> Chem.Mol:
    """A stand-in for one record of a calc_thermo output SDF."""
    mol = Chem.MolFromSmiles("C")
    for name, value in (
        ("G_hartree", G),
        ("H_hartree", H),
        ("S_hartree_per_K", S),
        ("T_K", T),
    ):
        mol.SetProp(name, str(value))
    return mol


def _check(mol: Chem.Mol) -> None:
    assert_thermo_record(mol, reference_G=REFERENCE_G_HARTREE, reference_H=REFERENCE_H_HARTREE)


def test_a_consistent_record_is_accepted():
    """Without this, every case below could pass by rejecting everything."""
    _check(_record(_H - _T * _S, _H, _S))


def test_a_sigma_convention_difference_is_tolerated():
    """The entropy band is deliberately 10%, not tight.

    Auto3D uses sigma=1 for a molecule with no ``symmetry_number`` property. If
    the reference calculation used cyclooctane's rotational symmetry number
    instead, R*ln(8) alone is 4.8% of S -- so a tight band would fail on a
    convention difference rather than on a defect.
    """
    _check(_record(_H - _T * _S * 0.92, _H, _S * 0.92))


@pytest.mark.parametrize(
    "label, mol",
    [
        # The entropy term is 25.5 kcal/mol for cyclooctane against a
        # 12.5 kcal/mol window on each of G and H, so the old G/H-only pair
        # bounded S to roughly +-50% and no better.
        ("entropy zeroed", _record(_H, _H, 0.0)),
        ("entropy halved", _record(_H - _T * _S / 2, _H, _S / 2)),
        ("entropy negative", _record(_H + _T * _S, _H, -_S)),
        # The failure do_mol_thermo's own comment warns about: S carries eV/K
        # from ASE and is written as Hartree/K, so a reader that treats the
        # property as an energy is off by a factor of T. Nothing checked it.
        ("S written in Hartree, not Hartree/K", _record(_H - _T * _S, _H, _S * _T)),
        ("G not equal to H - T*S", _record(_H - 0.001, _H, _S)),
        ("temperature not the documented 298.15 K", _record(_H - 310.0 * _S, _H, _S, T=310.0)),
    ],
)
def test_a_record_that_contradicts_itself_is_rejected(label, mol):
    with pytest.raises(AssertionError):
        _check(mol)


def test_calc_thermo_marks_implicit_hydrogen_records_as_failed(tmp_path, caplog, monkeypatch):
    """calc_thermo must mark an implicit-H record `Thermo_failed`, not score
    its bare heavy-atom skeleton and not let it silently vanish from the
    output the way a None/conformerless record does (N-C1).

    Runs `calc_thermo` end to end with a param-less stub NNP -- the same
    double-and-monkeypatch pattern
    `tests.test_thermo_helpers.TestCalculatorDeviceAndDtypeFollowTheCaller`
    uses to exercise `calc_thermo` without a real NNP in the fast tier --
    rather than the slow, real-model integration tests in test_thermo.py.
    Since the implicit-H record is caught before the fmax pre-check/
    relaxation/Hessian stages, the stub's `forward` need not even behave
    realistically; it exists only so `create_model`/`_load_hessian_model`
    never try to download or load a real model.
    """
    import logging

    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch import nn

    import Auto3D.entry.ASE.thermo.driver as thermo_mod
    from Auto3D.entry.ASE.thermo import calculator as _calculator
    from tests.helpers_adapter import AdapterModuleMixin

    class _StubNNP(AdapterModuleMixin, nn.Module):
        def forward(self, coords, species, charges, atom_mask=None):
            energy = torch.zeros(coords.shape[0], dtype=coords.dtype)
            forces = torch.zeros_like(coords)
            return energy, forces

    stub = _StubNNP()
    monkeypatch.setattr(thermo_mod, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(_calculator, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(thermo_mod, "_load_hessian_model", lambda *a, **k: object())

    mol = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol.SetProp("_Name", "noH")
    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)
    out = tmp_path / "out.sdf"

    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        thermo_mod.calc_thermo(str(sdf), "AIMNET", use_gpu=False, out_path=str(out))

    results = list(Chem.SDMolSupplier(str(out), removeHs=False))
    assert len(results) == 1, "the implicit-H record must be present in the output, not dropped"
    assert results[0].GetProp("Thermo_failed") == "implicit_hydrogens"
    implicit_h_warnings = [r for r in caplog.records if "implicit hydrogen" in r.message]
    assert len(implicit_h_warnings) == 1, (
        "the implicit-H record must be named exactly once -- calc_thermo reads "
        "`path` a single time now, so a second, contradictory 'Skipping ...' "
        f"line from sdf_io must not appear; got {[r.message for r in caplog.records]}"
    )


def test_calc_thermo_marks_dummy_atom_records_as_failed(tmp_path, caplog, monkeypatch):
    """A dummy-atom record gets the same treatment as an implicit-H one (N-M3).

    Mirrors ``test_calc_thermo_marks_implicit_hydrogen_records_as_failed``
    above, down to the param-less stub NNP: the record is caught before the
    fmax pre-check/relaxation/Hessian stages, so the stub exists only so
    ``create_model``/``_load_hessian_model`` never load a real model.

    Two things are pinned. ``_THERMO_SKIP_MESSAGES`` must carry an entry for
    the new reason -- it is looked up with ``[reason]``, so a missing entry
    raises ``KeyError`` on the first dummy-atom record and takes the whole run
    down. And the record must survive into the output carrying
    ``Thermo_failed="dummy_atoms"``, because every record of a calc_thermo
    output carries ``Thermo_failed``; dropping it would make the record vanish
    instead.
    """
    import logging

    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch import nn

    import Auto3D.entry.ASE.thermo.driver as thermo_mod
    from Auto3D.entry.ASE.thermo import calculator as _calculator
    from tests.helpers_adapter import AdapterModuleMixin

    class _StubNNP(AdapterModuleMixin, nn.Module):
        def forward(self, coords, species, charges, atom_mask=None):
            energy = torch.zeros(coords.shape[0], dtype=coords.dtype)
            forces = torch.zeros_like(coords)
            return energy, forces

    stub = _StubNNP()
    monkeypatch.setattr(thermo_mod, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(_calculator, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(thermo_mod, "_load_hessian_model", lambda *a, **k: object())

    # Explicit H and a real conformer, so the dummy atom is the only defect.
    mol = Chem.AddHs(Chem.MolFromSmiles("*CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0, "test premise: must embed"
    mol.SetProp("_Name", "frag")
    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)
    out = tmp_path / "out.sdf"

    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        thermo_mod.calc_thermo(str(sdf), "AIMNET", use_gpu=False, out_path=str(out))

    results = list(Chem.SDMolSupplier(str(out), removeHs=False))
    assert len(results) == 1, "the dummy-atom record must be present in the output, not dropped"
    assert results[0].GetProp("Thermo_failed") == "dummy_atoms"
    dummy_warnings = [r for r in caplog.records if "dummy atom" in r.message]
    assert len(dummy_warnings) == 1, (
        "the dummy-atom record must be named exactly once -- calc_thermo reads "
        "`path` a single time, so a second, contradictory 'Skipping ...' line "
        f"from sdf_io must not appear; got {[r.message for r in caplog.records]}"
    )
