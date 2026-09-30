"""P-C1: compile_model=True must not change energies. Reproduces on CPU."""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.engines.batch_opt.padding import pad_from_mols
from Auto3D.engines.model_factory import create_model

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CSCC", "OCC(O)CO", "Clc1ccccc1"]


def _mols():
    out = []
    for s in SMILES:
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        AllChem.EmbedMolecule(m, randomSeed=42)
        out.append(m)
    return out


def _energies_and_forces(adapter, mols):
    coord, species, charges, mask = pad_from_mols(mols, adapter, torch.device("cpu"))
    e, f = adapter(coord.clone(), species.clone(), charges.clone(), atom_mask=mask)
    return e.detach(), f.detach()


def test_ani2xt_compiled_matches_eager_on_cpu():
    mols = _mols()
    eager = create_model("ANI2xt", torch.device("cpu"), compile_model=False, use_cache=False)
    e0, f0 = _energies_and_forces(eager, mols)
    compiled = create_model("ANI2xt", torch.device("cpu"), compile_model=True, use_cache=False)
    e1, f1 = _energies_and_forces(compiled, mols)
    assert torch.allclose(e1, e0, atol=1e-5), (e1 - e0).abs().max()
    assert torch.allclose(f1, f0, atol=1e-4), (f1 - f0).abs().max()
    # second call, warm compile
    e2, _ = _energies_and_forces(compiled, mols)
    assert torch.allclose(e2, e0, atol=1e-5)
