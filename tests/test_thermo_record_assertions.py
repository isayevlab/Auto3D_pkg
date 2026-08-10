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
