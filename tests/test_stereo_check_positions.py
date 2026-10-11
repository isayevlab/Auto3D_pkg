"""One helper writes a coordinate array onto a molecule's conformer."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def test_set_conformer_positions_overwrites_every_atom():
    from Auto3D.domain.stereo_check import set_conformer_positions

    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    coords = np.arange(mol.GetNumAtoms() * 3, dtype=float).reshape(-1, 3)
    set_conformer_positions(mol, coords)
    got = mol.GetConformer().GetPositions()
    np.testing.assert_allclose(got, coords)
