"""The spin multiplicity travels with each molecule from the padded batch to the model.

A singlet is never passed: ``None`` stands for "every molecule is a singlet",
which is what every model assumes, so closed-shell runs take the code path they
always took and their numbers are bit-identical.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.engines.batch_opt.padding import PaddedBatch, pad_from_mols
from tests.helpers_adapter import FakeAdapter

AIMNET_INSTALLED = importlib.util.find_spec("aimnet") is not None
needs_aimnet = pytest.mark.skipif(not AIMNET_INSTALLED, reason="needs aimnet present")


def _embedded(smiles: str, seed: int = 7) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol.SetProp("_Name", smiles)
    return mol


def test_pad_from_mols_resolves_one_multiplicity_per_molecule(caplog):
    radical, ethanol = _embedded("[CH3]"), _embedded("CCO")
    with caplog.at_level(logging.WARNING):
        batch = pad_from_mols([radical, ethanol], FakeAdapter(), torch.device("cpu"))
    assert isinstance(batch, PaddedBatch)
    assert batch.mults.dtype == torch.float32 and batch.mults.shape == (2,)
    assert batch.mults.tolist() == [2.0, 1.0]
    coords, species, charges, mults, atom_mask = batch  # the five-way unpack
    assert mults is batch.mults
    # Padding runs once per conformer record; the open-shell notice is the Optimizer's, once per run.
    assert not [r for r in caplog.records if "open-shell" in r.getMessage()]


def test_the_resolver_warns_about_an_open_shell_only_when_asked(caplog):
    from Auto3D.domain.multiplicity import resolve_multiplicity

    radical = _embedded("[CH3]")
    with caplog.at_level(logging.WARNING):
        assert resolve_multiplicity(radical, warn_open_shell=False) == 2
    assert not [r for r in caplog.records if "open-shell" in r.getMessage()]
    with caplog.at_level(logging.WARNING):
        assert resolve_multiplicity(radical) == 2  # the thermo path's default, as in 3.2.1
    notices = [r.getMessage() for r in caplog.records if "open-shell" in r.getMessage()]
    assert len(notices) == 1 and "multiplicity 2" in notices[0] and "closed-shell" not in notices[0]


def test_sub_slices_all_five_fields_together():
    batch = pad_from_mols(
        [_embedded("[CH3]"), _embedded("CCO"), _embedded("[OH]")],
        FakeAdapter(),
        torch.device("cpu"),
    )
    part = batch.sub([0, 2])
    assert torch.equal(part.mults, batch.mults[[0, 2]])
    assert part.mults.tolist() == [2.0, 2.0]


class _RecordingNet:
    """A BatchedEnergyForce stand-in that records what forward_batched receives."""

    def __init__(self):
        self.calls: list[dict] = []

    def forward_batched(self, coord, numbers, charges, atom_mask=None, **kwargs):
        self.calls.append(dict(kwargs))
        energy = coord.new_zeros(coord.shape[0], dtype=torch.float64)
        return energy, torch.zeros_like(coord)


def _state(net, batch_size: int, mults):
    from Auto3D.engines.batch_opt.optimization_engine import n_steps

    coord = torch.zeros(batch_size, 3, 3)
    state = dict(
        coord=coord,
        numbers=torch.ones(batch_size, 3, dtype=torch.long),
        charges=torch.zeros(batch_size),
        mults=mults,
        converged_mask=torch.zeros(batch_size, dtype=torch.bool),
        nn=net,
        fmax=torch.zeros(batch_size),
        energy=torch.zeros(batch_size, dtype=torch.float64),
    )
    n_steps(state, 1, 0.01, 250)
    return state


def test_a_singlet_batch_passes_no_multiplicity_keyword():
    net = _RecordingNet()
    _state(net, 2, mults=None)
    assert net.calls and all("mults" not in call for call in net.calls)


def test_an_open_shell_batch_passes_the_sliced_multiplicities():
    net = _RecordingNet()
    _state(net, 2, mults=torch.tensor([2.0, 1.0]))
    assert net.calls and all(
        torch.equal(call["mults"], torch.tensor([2.0, 1.0])) for call in net.calls
    )


def test_ensemble_opt_passes_no_keyword_for_an_all_singlet_batch_and_the_multiplicities_otherwise():
    """Through the real n_steps: a batch of ones is stored as None, so a 3.2.1-era
    forward_batched double never sees the keyword; an open-shell batch does."""
    from Auto3D.engines.batch_opt.batchopt import ensemble_opt

    param = {
        "opt_steps": 1,
        "convergence_threshold": 0.01,
        "patience": 250,
        "batchsize_atoms": 1024,
    }
    coord = torch.zeros(2, 3, 3)
    numbers = torch.ones(2, 3, dtype=torch.long)
    net = _RecordingNet()
    ensemble_opt(
        net,
        coord.clone(),
        numbers,
        torch.zeros(2),
        param,
        torch.device("cpu"),
        mults=torch.ones(2),
    )
    assert net.calls and all("mults" not in call for call in net.calls)
    net = _RecordingNet()
    ensemble_opt(
        net,
        coord.clone(),
        numbers,
        torch.zeros(2),
        param,
        torch.device("cpu"),
        mults=torch.tensor([2.0, 1.0]),
    )
    assert net.calls and all(
        torch.equal(call["mults"], torch.tensor([2.0, 1.0])) for call in net.calls
    )


def _single(mol, adapter):
    """One molecule padded for ``adapter``: its coords, species, charges and atom mask."""
    batch = pad_from_mols([mol], adapter, torch.device("cpu"))
    return batch.coords, batch.species, batch.charges, batch.atom_mask


@needs_aimnet
@pytest.mark.slow
def test_the_nse_model_runs_a_radical_and_sees_its_multiplicity():
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("aimnet2-nse", torch.device("cpu"))
    assert adapter._calc.is_nse
    coords, species, charges, mask = _single(_embedded("[CH3]"), adapter)
    e_doublet, _ = adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
    e_forced_singlet, _ = adapter.forward(coords, species, charges, mask, mults=torch.tensor([1.0]))
    assert torch.isfinite(e_doublet).all()
    assert not torch.equal(e_doublet, e_forced_singlet)
    e_default, _ = adapter.forward(coords, species, charges, mask)  # None: singlets filled in
    assert torch.equal(e_default, e_forced_singlet)


@needs_aimnet
@pytest.mark.slow
def test_a_closed_shell_molecule_on_a_one_channel_model_is_bit_identical_with_and_without_the_keyword():
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("aimnet2", torch.device("cpu"))
    coords, species, charges, mask = _single(_embedded("CCO"), adapter)
    e_plain, f_plain = adapter.forward(coords, species, charges, mask)
    e_kw, f_kw = adapter.forward(coords, species, charges, mask, mults=torch.tensor([1.0]))
    assert torch.equal(e_plain, e_kw) and torch.equal(f_plain, f_kw)


@needs_aimnet
@pytest.mark.slow
def test_an_open_shell_molecule_on_a_one_channel_model_warns_once_and_is_evaluated_closed_shell(
    caplog,
):
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("aimnet2", torch.device("cpu"))
    coords, species, charges, mask = _single(_embedded("[CH3]"), adapter)
    e_plain, _ = adapter.forward(coords, species, charges, mask)
    with caplog.at_level(logging.WARNING):
        e_kw, _ = adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
        adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
    assert torch.equal(e_plain, e_kw)
    warnings = [r for r in caplog.records if "aimnet2-nse" in r.getMessage()]
    assert len(warnings) == 1


class _SpinlessNNP(torch.nn.Module):
    """A custom NNP written against the 3.2.1 contract: forward(species, coords, charges), no spin input."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, species, coords, charges):
        mask = (species != self.species_pad).unsqueeze(-1)
        return (coords * mask).pow(2).sum(dim=(1, 2))


def test_a_custom_nnp_without_a_spin_input_is_unchanged_by_the_keyword(tmp_path):
    from Auto3D.engines.models.adapter import CustomModelAdapter

    path = tmp_path / "spinless.pt"
    # Pickled by reference to this module, as the contract tests do.
    torch.save(_SpinlessNNP(), str(path))
    adapter = CustomModelAdapter(str(path), torch.device("cpu"))
    coords = torch.randn(2, 3, 3)
    species = torch.tensor([[1, 6, 8], [1, 6, -1]])
    charges = torch.zeros(2)
    e_plain, f_plain = adapter.forward(coords, species, charges)
    e_kw, f_kw = adapter.forward(coords, species, charges, mults=torch.tensor([2.0, 1.0]))
    assert torch.equal(e_plain, e_kw) and torch.equal(f_plain, f_kw)


def test_ani2xt_is_unchanged_by_the_keyword():
    pytest.importorskip("torchani")
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("ANI2xt", torch.device("cpu"))
    coords, species, charges, mask = _single(_embedded("CCO"), adapter)
    e_plain, f_plain = adapter.forward(coords, species, charges, mask)
    e_kw, f_kw = adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
    assert torch.equal(e_plain, e_kw) and torch.equal(f_plain, f_kw)


class _KeywordRecordingAdapter(FakeAdapter):
    """FakeAdapter that keeps the keyword arguments of the last forward, so a test can tell "not passed" from "passed None"."""

    def forward(self, coords, species, charges, atom_mask=None, **kwargs):
        self.last_kwargs = dict(kwargs)
        return super().forward(coords, species, charges, atom_mask)


def test_the_thermo_calculator_passes_the_multiplicity_only_above_one():
    from Auto3D.entry.ASE.thermo.calculator import Calculator, mol2atoms

    adapter = _KeywordRecordingAdapter()
    calc = Calculator(adapter, charge=0)  # the contract check at construction passes
    atoms = mol2atoms(_embedded("[CH3]"))
    atoms.calc = calc
    atoms.get_potential_energy()
    assert adapter.last_kwargs == {}
    calc.set_mult(2)
    atoms.get_potential_energy()  # the parameter change invalidated ASE's cache, as set_charge does
    assert torch.equal(adapter.last_kwargs["mults"], torch.tensor([2.0]))
