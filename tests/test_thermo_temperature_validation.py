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
