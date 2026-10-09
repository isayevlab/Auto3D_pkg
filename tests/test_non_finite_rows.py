"""A molecule whose energy goes non-finite leaves the optimization as not converged;
the rest of its bucket finishes (R38). The single-point path keeps raising."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.engines.batch_opt.model_wrapper import EnForce_ANI
from Auto3D.engines.batch_opt.optimization_engine import n_steps
from Auto3D.foundation.exceptions import NumericalError
from tests.helpers_adapter import FakeAdapter


def _nan_on_tagged_row(coords, species, charges):
    """Poison whichever row carries species 2, wherever it lands in a sub-batch.

    Tagged by species identity, not by position: ``_run_in_sub_batches``'s
    bisection re-slices ``coords``/``species`` into ever-smaller sub-batches,
    so a position-based poison (e.g. ``poison[1] = nan``) would tag a
    DIFFERENT row once the original row 1 is isolated into its own size-1
    sub-batch (or go out of bounds once a sub-batch shrinks below index 1).
    Species travels with its row through every slice, so tagging by species
    keeps poisoning the same molecule regardless of how it gets sliced.
    """
    energy = coords.pow(2).sum(dim=(1, 2))
    poison = torch.zeros_like(energy)
    poison[(species == 2).any(dim=1)] = float("nan")
    return energy + poison


def _nan_on_oxygen(coords, species, charges):
    """Poison any row containing oxygen (species 8) -- water, not methane."""
    energy = coords.pow(2).sum(dim=(1, 2))
    poison = torch.zeros_like(energy)
    poison[(species == 8).any(dim=1)] = float("nan")
    return energy + poison


def _validating(adapter):
    """Wrap the fake's forward with the production NaN gate so it raises like a real adapter."""
    from Auto3D.engines.models.adapter import _validate_outputs

    raw = adapter.forward

    def forward(coords, species, charges, atom_mask=None):
        e, f = raw(coords, species, charges, atom_mask=atom_mask)
        _validate_outputs(e, f)
        return e, f

    adapter.forward = forward
    return adapter


def _state(adapter, batch=4, atoms=3):
    coord = torch.randn(batch, atoms, 3, dtype=torch.float)
    numbers = torch.ones(batch, atoms, dtype=torch.long)
    # Tags row 1 so `_nan_on_tagged_row` keeps poisoning THIS row through any
    # bisection, rather than whatever row happens to sit at position 1 of a
    # shrinking sub-batch.
    numbers[1] = 2
    return {
        "coord": coord,
        "numbers": numbers,
        "charges": torch.zeros(batch, dtype=torch.float),
        "nn": EnForce_ANI(adapter, 1024),
        "converged_mask": torch.zeros(batch, dtype=torch.bool),
        "fmax": torch.full((batch,), 999.0),
        "energy": torch.full((batch,), 999.0, dtype=torch.double),
    }


def test_non_finite_row_is_dropped_and_the_rest_converge():
    adapter = _validating(FakeAdapter(energy_fn=_nan_on_tagged_row))
    state = _state(adapter)
    n_steps(state, n=200, opttol=1e-3, patience=250)
    assert not bool(state["converged_mask"][1])
    assert bool(state["non_finite"][1])
    assert torch.isnan(state["energy"][1])
    assert bool(state["converged_mask"][[0, 2, 3]].all())
    assert torch.isfinite(state["energy"][[0, 2, 3]]).all()


def test_energy_batched_still_raises_on_non_finite():
    adapter = _validating(FakeAdapter(energy_fn=_nan_on_tagged_row))
    nn = EnForce_ANI(adapter, 1024)
    coords = torch.randn(4, 3, 3)
    species = torch.ones(4, 3, dtype=torch.long)
    species[1] = 2
    with pytest.raises(NumericalError):
        nn.energy_batched(coords, species, torch.zeros(4))


def test_optimizing_run_writes_non_finite_row_not_converged(tmp_path):
    """End to end: `optimizing.run` on a two-record SDF, one record poisoned.

    The first record (Converged=True) and the second (Converged=False,
    Optimization_failed=non_finite_energy, no E_tot) are the R65 contract
    `batchopt.run` must honor regardless of which bucket(s) the two records
    land in.
    """
    from Auto3D.engines.batch_opt.batchopt import optimizing
    from Auto3D.foundation.utils.convergence import OPTIMIZATION_FAILED_PROP

    methane = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(methane, randomSeed=1)
    methane.SetProp("_Name", "methane")
    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(water, randomSeed=1)
    water.SetProp("_Name", "water")

    inp = tmp_path / "in.sdf"
    with Chem.SDWriter(str(inp)) as w:
        w.write(methane)
        w.write(water)

    adapter = _validating(FakeAdapter(energy_fn=_nan_on_oxygen))
    out = tmp_path / "out.sdf"
    opt = optimizing(
        str(inp),
        str(out),
        adapter=adapter,
        device=torch.device("cpu"),
        config={"opt_steps": 200, "opttol": 1e-3, "patience": 50, "batchsize_atoms": 1024},
    )
    opt.run()

    mols = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(str(out), removeHs=False)}
    assert set(mols) == {"methane", "water"}

    assert mols["methane"].GetProp("Converged") == "True"
    assert mols["methane"].HasProp("E_tot")

    assert mols["water"].GetProp("Converged") == "False"
    assert mols["water"].GetProp(OPTIMIZATION_FAILED_PROP) == "non_finite_energy"
    assert not mols["water"].HasProp("E_tot")
    assert not mols["water"].HasProp("fmax")
