"""The conformer pool must be a property of this code, not of the RDKit build.

``EmbedMultipleConfs``'s keyword form cannot express ``onlyHeavyAtomsForRMS`` or
``useSymmetryForPruning`` -- they exist only on the parameters object. Left to
their defaults, the number of conformers ``pruneRmsThresh`` leaves behind
depends on which RDKit is installed: both default True on 2025.09 but have not
always, and ``pyproject.toml`` floors at ``rdkit>=2022.9.5`` with no upper
bound.

Switching to a parameters object is only safe if it is the *same*
parameterization the keyword form applied, which is what these tests pin. A
bare ``EmbedParameters()`` is not: it turns off the torsion knowledge ETKDG is
named for.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from Auto3D.constants import CONFORMER_RANDOM_SEED
from Auto3D.embedding import embed_params

# The keyword overload's documented C++ defaults -- the behaviour being preserved.
KWARG_DEFAULTS = {
    "useExpTorsionAnglePrefs": True,
    "useBasicKnowledge": True,
    "ETversion": 2,
    "useMacrocycleTorsions": True,
    "useMacrocycle14config": True,
    "useSmallRingTorsions": False,
    "enforceChirality": True,
    "clearConfs": True,
    "useRandomCoords": False,
    "randNegEig": True,
    "numZeroFail": 1,
    "ignoreSmoothingFailures": False,
}


@pytest.mark.parametrize("field,expected", sorted(KWARG_DEFAULTS.items()))
def test_matches_the_keyword_forms_defaults(field, expected):
    """Field by field, so a future RDKit changing a preset is caught here."""
    assert getattr(embed_params(n_threads=1, prune_rms_thresh=0.3), field) == expected


def test_the_determinism_flags_are_stated_not_defaulted():
    params = embed_params(n_threads=1, prune_rms_thresh=0.3)
    assert params.onlyHeavyAtomsForRMS is True
    assert params.useSymmetryForPruning is True


def test_the_seed_comes_from_the_shared_constant():
    assert embed_params(n_threads=1, prune_rms_thresh=0.3).randomSeed == CONFORMER_RANDOM_SEED


def test_a_bare_parameters_object_would_have_been_wrong():
    """The premise for using ETKDGv3: a bare object disables ETKDG's knowledge."""
    bare = rdDistGeom.EmbedParameters()
    assert bare.useExpTorsionAnglePrefs is False
    assert bare.useBasicKnowledge is False


@pytest.mark.parametrize("smiles", ["CCO", "OCC(O)CO", "CC(=O)Nc1ccc(O)cc1"])
def test_geometry_is_unchanged_by_the_switch(smiles):
    """The switch must move no atom. Compares against the keyword form."""
    def kwarg_form():
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMultipleConfs(
            mol, numConfs=30, randomSeed=CONFORMER_RANDOM_SEED,
            numThreads=1, pruneRmsThresh=0.3,
        )
        return mol

    def params_form():
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMultipleConfs(
            mol, numConfs=30,
            params=embed_params(n_threads=1, prune_rms_thresh=0.3),
        )
        return mol

    old, new = kwarg_form(), params_form()

    assert old.GetNumConformers() == new.GetNumConformers(), (
        "the parameters form kept a different number of conformers"
    )
    for i in range(old.GetNumConformers()):
        np.testing.assert_allclose(
            old.GetConformer(i).GetPositions(),
            new.GetConformer(i).GetPositions(),
            err_msg=f"conformer {i} of {smiles} moved",
        )
