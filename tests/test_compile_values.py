"""P-C1: compile_model=True must not change energies. Reproduces on CPU."""

import logging

import pytest
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


def test_ani2x_adapter_compile_hook_returns_model_unchanged(caplog):
    from Auto3D.engines.models.adapter import ANI2xAdapter

    adapter = ANI2xAdapter.__new__(ANI2xAdapter)  # bypass __init__: no torchani load
    model = torch.nn.Linear(1, 1)
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        out = adapter._compile(model)
    assert out is model
    assert any("ANI2x" in r.message for r in caplog.records)


@pytest.mark.slow  # loads torchani's 8-model ensemble
def test_ani2x_compile_request_warns_and_runs_eager(caplog):
    torchani = pytest.importorskip("torchani")
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        adapter = create_model("ANI2x", torch.device("cpu"), compile_model=True, use_cache=False)
    assert adapter._compiled is False
    assert any("ANI2x" in r.message and "compile" in r.message.lower() for r in caplog.records)


from Auto3D.foundation.exceptions import NumericalError


def test_create_model_refuses_a_compiled_adapter_that_disagrees_with_eager(monkeypatch):
    """A compile that silently changes the numbers must not reach the optimizer."""
    import Auto3D.engines.models.adapter as adapter_mod

    def _bad_compile(obj, **kwargs):
        # Simulate a numerically wrong compilation: +1 eV per molecule.
        if isinstance(obj, torch.nn.Module):

            class _Wrapped(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self._orig_mod = inner

                def forward(self, *a, **k):
                    return self._orig_mod(*a, **k) + 1.0

            return _Wrapped(obj)
        return lambda *a, **k: obj(*a, **k) + 1.0

    monkeypatch.setattr(adapter_mod.torch, "compile", _bad_compile)
    with pytest.raises(NumericalError, match="compiled"):
        create_model("ANI2xt", torch.device("cpu"), compile_model=True, use_cache=False)
