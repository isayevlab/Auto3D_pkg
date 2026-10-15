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
    # Zero forces converge every row on the first step, so the net sees the
    # loop's one call and the final recompute's -- and both must agree.
    assert len(net.calls) == 2
    assert all("mults" not in call for call in net.calls)


def test_an_open_shell_batch_passes_the_sliced_multiplicities():
    net = _RecordingNet()
    _state(net, 2, mults=torch.tensor([2.0, 1.0]))
    assert len(net.calls) == 2  # the loop's call and the final recompute's
    assert all(torch.equal(call["mults"], torch.tensor([2.0, 1.0])) for call in net.calls)


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
    # The doublet is the methyl radical's ground state, so it must lie lower.
    assert float(e_doublet) < float(e_forced_singlet)
    e_default, _ = adapter.forward(coords, species, charges, mask)  # None: singlets filled in
    assert torch.equal(e_default, e_forced_singlet)


@needs_aimnet
@pytest.mark.slow
def test_triplet_oxygen_lies_below_singlet_oxygen_on_the_nse_model():
    """Both are physical states of O2 and the triplet is the ground state.

    The radical test above compares a real state with a forced one; this pair
    is two states the model is meant to tell apart, so it checks the spin input
    is doing chemistry rather than merely changing the number.
    """
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("aimnet2-nse", torch.device("cpu"))
    coords, species, charges, mask = _single(_embedded("O=O"), adapter)
    e_triplet, _ = adapter.forward(coords, species, charges, mask, mults=torch.tensor([3.0]))
    e_singlet, _ = adapter.forward(coords, species, charges, mask, mults=torch.tensor([1.0]))
    assert torch.isfinite(e_triplet).all()
    assert float(e_triplet) < float(e_singlet)


@needs_aimnet
@pytest.mark.slow
def test_the_nse_hessian_depends_on_the_multiplicity():
    """The native Hessian route carries the spin state too, so an open-shell
    record's frequencies belong to the state its energy was computed in."""
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("aimnet2-nse", torch.device("cpu"))
    coords, species, charges, _mask = _single(_embedded("[CH3]"), adapter)
    h_doublet = adapter.analytic_hessian(coords, species, charges, mults=torch.tensor([2.0]))
    h_singlet = adapter.analytic_hessian(coords, species, charges, mults=torch.tensor([1.0]))
    assert torch.isfinite(h_doublet).all()
    assert not torch.equal(h_doublet, h_singlet)


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
    e_plain, f_plain = adapter.forward(coords, species, charges, mask)
    with caplog.at_level(logging.WARNING):
        e_kw, f_kw = adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
        adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
    assert torch.equal(e_plain, e_kw) and torch.equal(f_plain, f_kw)
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


@pytest.mark.slow
def test_ani2xt_is_unchanged_by_the_keyword():
    pytest.importorskip("torchani")
    from Auto3D.engines.model_factory import create_model

    adapter = create_model("ANI2xt", torch.device("cpu"))
    coords, species, charges, mask = _single(_embedded("CCO"), adapter)
    e_plain, f_plain = adapter.forward(coords, species, charges, mask)
    e_kw, f_kw = adapter.forward(coords, species, charges, mask, mults=torch.tensor([2.0]))
    assert torch.equal(e_plain, e_kw) and torch.equal(f_plain, f_kw)


def test_the_thermo_calculator_passes_the_multiplicity_only_above_one():
    """``FakeAdapter`` records whether the keyword was passed AT ALL, which is
    what distinguishes "a singlet is never passed" from "passed None"."""
    from Auto3D.entry.ASE.thermo.calculator import Calculator, mol2atoms

    adapter = FakeAdapter()
    calc = Calculator(adapter, charge=0)  # the contract check at construction passes
    atoms = mol2atoms(_embedded("[CH3]"))
    atoms.calc = calc
    atoms.get_potential_energy()
    assert adapter.calls and not adapter.calls[-1]["mults_passed"]
    calc.set_mult(2)
    atoms.get_potential_energy()  # the parameter change invalidated ASE's cache, as set_charge does
    assert adapter.calls[-1]["mults_passed"]
    assert torch.equal(adapter.calls[-1]["mults"], torch.tensor([2.0]))


def test_the_thermo_calculator_sends_what_ases_own_set_wrote():
    """``calc.set(mult=...)`` bypasses the property setter, so the tensor
    ``calculate`` sends has to come from the parameter, not from a copy."""
    from Auto3D.entry.ASE.thermo.calculator import Calculator, mol2atoms

    adapter = FakeAdapter()
    calc = Calculator(adapter, charge=0)
    atoms = mol2atoms(_embedded("[CH3]"))
    atoms.calc = calc
    calc.set(mult=2)  # ASE's own parameter write, not calc.mult = 2
    atoms.get_potential_energy()
    assert torch.equal(adapter.calls[-1]["mults"], torch.tensor([2.0]))


def test_vib_hessian_hands_the_multiplicity_to_a_native_hessian():
    from Auto3D.entry.ASE.thermo.vibrations import vib_hessian

    mol = _embedded("[CH3]")
    n = mol.GetNumAtoms()
    adapter = FakeAdapter(hessian=torch.zeros(n, 3, n, 3, dtype=torch.float64))
    vib_hessian(mol, object(), adapter, torch.device("cpu"), multiplicity=2)
    hessians = [c for c in adapter.calls if c["kind"] == "analytic_hessian"]
    assert len(hessians) == 1 and hessians[0]["mults_passed"]
    assert torch.equal(hessians[0]["mults"], torch.tensor([2.0]))

    singlet = FakeAdapter(hessian=torch.zeros(n, 3, n, 3, dtype=torch.float64))
    vib_hessian(mol, object(), singlet, torch.device("cpu"))  # the default: a singlet
    assert singlet.calls and not any(c["mults_passed"] for c in singlet.calls)


def test_vib_hessian_hands_the_multiplicity_down_the_autograd_route():
    """An adapter with no native Hessian is differentiated through ``energy``,
    which must see the same spin state the native route would have."""
    from Auto3D.entry.ASE.thermo.vibrations import vib_hessian

    adapter = FakeAdapter()  # hessian=None: differentiate `energy`
    vib_hessian(_embedded("[CH3]"), object(), adapter, torch.device("cpu"), multiplicity=2)
    energies = [c for c in adapter.calls if c["kind"] == "energy"]
    assert energies and all(c["mults_passed"] for c in energies)
    assert all(torch.equal(c["mults"], torch.tensor([2.0])) for c in energies)


def test_do_mol_thermo_tells_the_attached_calculator_the_multiplicity():
    """The record's energy and its Hessian must come from one spin state."""
    from Auto3D.entry.ASE.thermo.calculator import Calculator, mol2atoms
    from Auto3D.entry.ASE.thermo.driver import do_mol_thermo

    adapter = FakeAdapter()
    calc = Calculator(adapter, charge=0)
    mol = _embedded("[CH3]")
    atoms = mol2atoms(mol)
    atoms.calc = calc
    do_mol_thermo(mol, atoms, adapter, torch.device("cpu"), multiplicity=2)
    assert calc.parameters["mult"] == 2
    forwards = [c for c in adapter.calls if c["kind"] == "forward"]
    assert forwards and all(c["mults_passed"] for c in forwards)


def test_the_single_point_path_passes_the_multiplicities_and_nothing_for_singlets(
    tmp_path, monkeypatch
):
    """``calc_spe``'s own hand-off, with the real padder and no model loaded."""
    import Auto3D.entry._run_setup as run_setup
    import Auto3D.entry.SPE as spe_mod

    captured: list[dict] = []

    class _RecordingWrapper:
        def __init__(self, adapter, *args, **kwargs):
            pass

        def energy_batched(self, coords, numbers, charges, atom_mask=None, **kwargs):
            captured.append(dict(kwargs))
            return torch.zeros(coords.shape[0], dtype=torch.float64)

    monkeypatch.setattr(run_setup, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter(species_pad=0))
    monkeypatch.setattr(spe_mod, "BatchedEnergyForce", _RecordingWrapper)

    def _run(mols, name):
        sdf = tmp_path / f"{name}.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            for mol in mols:
                writer.write(mol)
        spe_mod.calc_spe(
            str(sdf), "AIMNET", use_gpu=False, out_path=str(tmp_path / f"{name}_out.sdf")
        )

    _run([_embedded("[CH3]"), _embedded("CCO")], "open_shell")
    assert torch.equal(captured[-1]["mults"], torch.tensor([2.0, 1.0]))
    _run([_embedded("CCO"), _embedded("CO")], "singlets")
    assert captured[-1] == {}  # a singlet is never passed


def test_a_species_drawn_closed_shell_is_reported_once_per_run(tmp_path, monkeypatch, caplog):
    """O=O carries no radical electrons, so only this check can see it.

    Two buckets, each holding one such record: the notice is the
    ``Optimizer``'s, once per instance, not once per bucket.
    """
    import Auto3D.engines.batch_opt.batchopt as bo
    from tests.helpers_batchopt import make_fake_ensemble_opt

    monkeypatch.setattr(bo, "ensemble_opt", make_fake_ensemble_opt())
    # One molecule per bucket, so two records make two buckets regardless of
    # how close their atom counts are.
    monkeypatch.setattr(bo.Optimizer, "BUCKET_MAX_COUNT", 1)

    sdf = tmp_path / "oxygen.sdf"
    with Chem.SDWriter(str(sdf)) as writer:
        for i in range(2):
            mol = _embedded("O=O")
            mol.SetProp("_Name", f"o2_{i}")
            writer.write(mol)

    engine = bo.Optimizer(
        str(sdf),
        str(tmp_path / "oxygen_out.sdf"),
        adapter=FakeAdapter(),
        device=torch.device("cpu"),
        config={
            "opt_steps": 1,
            "convergence_threshold": 0.01,
            "patience": 1,
            "batchsize_atoms": 1024,
        },
    )
    with caplog.at_level(logging.WARNING):
        engine.run()
    notices = [r for r in caplog.records if "drawn closed-shell" in r.getMessage()]
    assert len(notices) == 1, [r.getMessage() for r in caplog.records]


def test_a_drawn_closed_shell_record_with_its_ground_state_set_is_not_reported(
    tmp_path, monkeypatch, caplog
):
    """The notice names records optimized at the wrong state, so a record whose
    'multiplicity' property already says 3 must not produce one."""
    import Auto3D.engines.batch_opt.batchopt as bo
    from tests.helpers_batchopt import make_fake_ensemble_opt

    monkeypatch.setattr(bo, "ensemble_opt", make_fake_ensemble_opt())

    sdf = tmp_path / "oxygen.sdf"
    with Chem.SDWriter(str(sdf)) as writer:
        mol = _embedded("O=O")
        mol.SetProp("_Name", "o2_triplet")
        mol.SetProp("multiplicity", "3")
        writer.write(mol)

    engine = bo.Optimizer(
        str(sdf),
        str(tmp_path / "oxygen_out.sdf"),
        adapter=FakeAdapter(),
        device=torch.device("cpu"),
        config={
            "opt_steps": 1,
            "convergence_threshold": 0.01,
            "patience": 1,
            "batchsize_atoms": 1024,
        },
    )
    with caplog.at_level(logging.WARNING):
        engine.run()
    notices = [r for r in caplog.records if "drawn closed-shell" in r.getMessage()]
    assert notices == [], [r.getMessage() for r in caplog.records]
