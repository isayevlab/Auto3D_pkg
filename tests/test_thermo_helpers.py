"""Fast unit tests for thermochemistry helper functions.

These cover the pure-Python helpers in Auto3D.ASE.thermo and do not run any
neural-network potential or thermodynamic calculation, so they stay in the
fast test suite (the main tests/test_thermo.py module is marked slow).
"""


def test_detect_geometry_linear_vs_nonlinear():
    from ase import Atoms
    from Auto3D.ASE.thermo import _detect_geometry
    co2 = Atoms("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    water = Atoms("OH2", [[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]])
    assert _detect_geometry(co2) == "linear"
    assert _detect_geometry(water) == "nonlinear"


def test_symmetry_number_basic():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    benzene = Chem.MolFromSmiles("c1ccccc1")  # no explicit H -> 6-ring -> >= 12
    assert _symmetry_number(benzene) >= 12
    chiral = Chem.MolFromSmiles("C[C@H](O)Cl")  # central C all-different -> 1
    assert _symmetry_number(chiral) == 1
