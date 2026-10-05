"""P-C1: compile_model=True must not change energies. Reproduces on CPU."""

import logging

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.engines.batch_opt.padding import pad_from_mols
from Auto3D.engines.model_factory import create_model
from Auto3D.foundation.exceptions import NumericalError
from tests.helpers_custom_nnp import ScriptableNNP

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


def _bad_compile(obj, **kwargs):
    """Stand in for a torch.compile that silently changes the numbers.

    Two shapes, because the two adapters hand ``_try_compile`` different
    objects:

    * A whole ``nn.Module`` -- ``BaseModelAdapter._compile``, which is what
      ``CustomModelAdapter`` uses. The wrapper adds 1.0 to the module's
      returned energy, i.e. 1 eV per molecule for an adapter whose model
      already answers in eV.
    * A plain function -- ``ANI2xtAdapter._compile`` compiles
      ``ANI2xt._atom_energies_fn``, not the module. The lambda adds 1.0 to
      every PER-ATOM row that function returns, and it works in Hartree, so
      the molecular energy ends up wrong by ``n_atoms * 27.2`` eV rather than
      by 1 eV.

    Either way the disagreement is orders of magnitude above
    ``COMPILE_PROBE_TOLERANCE_EV`` (1e-3 eV), which is all the probe tests need.
    """
    if isinstance(obj, torch.nn.Module):

        class _Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self._orig_mod = inner

            def forward(self, *a, **k):
                return self._orig_mod(*a, **k) + 1.0

        return _Wrapped(obj)
    return lambda *a, **k: obj(*a, **k) + 1.0


def _saved_eager_nnp(tmp_path):
    """An eager (non-TorchScript) custom NNP on disk, so compile_model applies.

    ``CustomModelAdapter`` refuses to compile a TorchScript archive (it is
    already a compiled graph), so only the ``torch.save`` form reaches
    ``_try_compile`` and therefore the factory's probe.
    """
    path = tmp_path / "eager_nnp.pt"
    torch.save(ScriptableNNP(), str(path))
    return str(path)


@pytest.mark.slow
def test_ani2xt_compiled_matches_eager_on_cpu():
    pytest.importorskip("torchani")
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


def test_create_model_refuses_a_compiled_adapter_that_disagrees_with_eager(monkeypatch):
    """A compile that silently changes the numbers must not reach the optimizer."""
    pytest.importorskip("torchani")
    import Auto3D.engines.models.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.torch, "compile", _bad_compile)
    with pytest.raises(NumericalError, match="compiled"):
        create_model("ANI2xt", torch.device("cpu"), compile_model=True, use_cache=False)


class TestCustomNnpProbeSeam:
    """The custom-NNP branch of the factory runs the same probe (C-5).

    Torchani-free on purpose: the two ANI2xt tests above skip on the
    ``ani=false`` CI legs, and the probe seam -- the code path that decides
    whether a compiled adapter is allowed to reach the optimizer -- had no
    coverage at all on those legs.
    """

    def test_a_compiled_custom_nnp_that_agrees_with_eager_is_accepted(self, tmp_path):
        path = _saved_eager_nnp(tmp_path)
        mols = _mols()

        eager = create_model(path, torch.device("cpu"), compile_model=False, use_cache=False)
        e0, f0 = _energies_and_forces(eager, mols)
        # create_model itself probes compiled-vs-eager and raises NumericalError
        # on disagreement, so simply returning is already part of the assertion.
        compiled = create_model(path, torch.device("cpu"), compile_model=True, use_cache=False)
        assert compiled._compiled is True, "test premise: an eager .pt must be compiled"
        e1, f1 = _energies_and_forces(compiled, mols)

        assert torch.allclose(e1, e0, atol=1e-5), (e1 - e0).abs().max()
        assert torch.allclose(f1, f0, atol=1e-4), (f1 - f0).abs().max()

    def test_a_compiled_custom_nnp_that_disagrees_with_eager_is_refused(
        self, tmp_path, monkeypatch
    ):
        import Auto3D.engines.models.adapter as adapter_mod

        path = _saved_eager_nnp(tmp_path)
        monkeypatch.setattr(adapter_mod.torch, "compile", _bad_compile)
        with pytest.raises(NumericalError, match="compiled"):
            create_model(path, torch.device("cpu"), compile_model=True, use_cache=False)
