"""One adapter, one invocation convention, for the whole Hessian path.

``calc_thermo`` used to reach a model through four different conventions, three
of them keyed on an engine-name *string*:

* ``Calculator.__init__(..., model_name=...)`` -- dispatched species through a
  name-keyed converter,
* ``mol2aimnet_input(..., model_name=...)`` -- the same, again,
* ``_load_hessian_model(model_name, device)`` -- two different RETURN TYPES
  (a bare fp64 ``nn.Module`` for ANI/custom, an ``AIMNet2Calculator`` for
  aimnet), selected by folding the name,
* ``aimnet_hessian_helper(coord, numbers, charge, model, model_name)`` -- a
  fourth calling convention with a per-engine argument order and its own
  Hartree->eV factor, plus a fifth dispatch (``isinstance(model,
  AIMNet2Calculator)``) in ``vib_hessian`` to choose analytic vs autograd.

Each of those duplicated knowledge the adapters already own, and each could
disagree with the others: an aimnet registry alias such as ``aimnet2-2025``
matched no branch in ``aimnet_hessian_helper`` and raised, while the same name
resolved fine everywhere else.

The tests below pin the replacement. Every entry point takes a
:class:`~Auto3D.models.contract.ModelAdapter` and asks *it*; the analytic-vs-
autograd choice is a capability query (``analytic_hessian`` returning ``None``
means "differentiate ``energy()``"), not a type test; and no engine-name string
survives on the path. Nothing here loads a neural network potential -- the one
adapter is a recording double, and the reference Hessian is analytic.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from Auto3D.ASE.thermo import calculator as _calculator

pytest.importorskip("ase")
pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

import Auto3D.ASE.thermo.driver as thermo_mod  # noqa: E402
from Auto3D.models.contract import missing_adapter_members  # noqa: E402
from tests.helpers_adapter import AdapterModuleMixin  # noqa: E402

#: An atomic-number -> species remap no engine name could ever produce, so an
#: assertion that it reached a call site proves the value came from THIS object.
SENTINEL_OFFSET = 1000


class _RecordingAdapter(AdapterModuleMixin, nn.Module):
    """The single adapter every Hessian-path entry point is driven with.

    ``E = sum(coords**2)`` so the exact Hessian is ``2*I`` -- an analytic
    reference, not a tolerance. Records which member each caller used and the
    species tensor it was handed.
    """

    def __init__(self, analytic: torch.Tensor | None = None) -> None:
        super().__init__()
        self.analytic = analytic
        self.calls: list[str] = []
        self.species_seen: list[list[int]] = []
        self.coord_dtypes: list[torch.dtype] = []

    def to_species(self, atomic_numbers):
        self.calls.append("to_species")
        return [SENTINEL_OFFSET + int(z) for z in atomic_numbers]

    def energy(self, coords, species, charges, atom_mask=None):
        self.calls.append("energy")
        self.species_seen.append([int(v) for v in species.reshape(-1).tolist()])
        self.coord_dtypes.append(coords.dtype)
        return coords.pow(2).sum(dim=(1, 2))

    def forward(self, coords, species, charges, atom_mask=None):
        self.calls.append("forward")
        self.species_seen.append([int(v) for v in species.reshape(-1).tolist()])
        self.coord_dtypes.append(coords.dtype)
        coords = coords if coords.requires_grad else coords.requires_grad_(True)
        energy = coords.pow(2).sum(dim=(1, 2))
        grad = torch.autograd.grad([energy.sum()], [coords], create_graph=False)[0]
        return energy, -grad

    def analytic_hessian(self, coords, species, charges):
        self.calls.append("analytic_hessian")
        self.species_seen.append([int(v) for v in species.reshape(-1).tolist()])
        return self.analytic


def _water() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", "water")
    return mol


class _InertCalculator:
    """Stands in for the ASE calculator ``vib_hessian`` attaches to ``Atoms``.

    ``vib_hessian`` never asks it for anything (``VibrationsData`` takes the
    Hessian matrix directly), so an object is enough and no model is involved.
    """


class TestOneAdapterDrivesEveryHessianEntryPoint:
    """The keystone: a single object supplies species, energy and Hessian."""

    def test_no_entry_point_on_the_path_takes_a_model_name(self):
        """A name string is what made three dispatches able to disagree.

        ``calc_thermo``'s own ``model_name`` parameter is deliberately excluded:
        that is the user-facing engine selector, resolved once by
        ``resolve_engine_name``/``ModelFactory`` before any model exists. What
        must not survive is a name threaded *past* construction, where an
        adapter is already in hand.
        """
        for fn in (
            _calculator.Calculator.__init__,
            thermo_mod.mol2aimnet_input,
            thermo_mod.vib_hessian,
            thermo_mod.do_mol_thermo,
        ):
            assert "model_name" not in inspect.signature(fn).parameters, (
                f"{fn.__qualname__} still dispatches on an engine name"
            )

    def test_the_fourth_calling_convention_is_gone(self):
        """``aimnet_hessian_helper`` was a model interface neither Protocol described."""
        assert not hasattr(thermo_mod, "aimnet_hessian_helper")

    def test_the_name_keyed_species_converter_is_gone(self):
        """``to_model_species`` existed only for callers holding a name, not a model."""
        from Auto3D.models import species as species_mod

        assert not hasattr(species_mod, "to_model_species")
        assert "to_model_species" not in species_mod.__all__

    def test_thermo_no_longer_carries_its_own_hartree_factor(self):
        """One conversion, in ``Auto3D.constants``, reached via the adapters."""
        assert not hasattr(thermo_mod, "hartree2ev")

    def test_one_adapter_supplies_species_energy_and_hessian(self):
        """Drive all three entry points with the SAME object and no name.

        The sentinel remap is the proof: it can only have come from
        ``adapter.to_species``, so the species convention and the padding
        values are guaranteed to originate from one place (audit C3/C4).
        """
        mol = _water()
        device = torch.device("cpu")
        adapter = _RecordingAdapter()
        expected = [SENTINEL_OFFSET + a.GetAtomicNum() for a in mol.GetAtoms()]

        # 1. mol2aimnet_input
        batch = thermo_mod.mol2aimnet_input(mol, device, adapter=adapter)
        assert [int(v) for v in batch["numbers"].reshape(-1).tolist()] == expected

        # 2. the ASE Calculator
        calc = _calculator.Calculator(adapter, 0, device=device)
        atoms = thermo_mod.mol2atoms(mol)
        atoms.calc = calc
        atoms.get_potential_energy()
        assert expected in adapter.species_seen

        # 3. vib_hessian
        adapter.species_seen.clear()
        vib = thermo_mod.vib_hessian(mol, _InertCalculator(), adapter, device)
        assert expected in adapter.species_seen
        assert vib.get_hessian_2d().shape == (3 * mol.GetNumAtoms(),) * 2


class TestAnalyticVersusAutograd:
    """The choice is a capability, not a third-party ``isinstance``."""

    def test_autograd_reproduces_the_exact_analytic_hessian(self):
        """``E = sum(coords**2)`` has Hessian ``2*I`` exactly, not approximately."""
        mol = _water()
        adapter = _RecordingAdapter(analytic=None)

        vib = thermo_mod.vib_hessian(mol, _InertCalculator(), adapter, torch.device("cpu"))
        hess = vib.get_hessian_2d()

        n = 3 * mol.GetNumAtoms()
        assert np.array_equal(hess, 2.0 * np.eye(n)), (
            "the autograd path did not reproduce the analytic Hessian exactly"
        )
        assert "energy" in adapter.calls
        assert "forward" not in adapter.calls, (
            "the autograd path went through forward(), paying for forces it "
            "then discarded -- and downcasting an fp64 request in two adapters"
        )

    def test_an_adapter_with_an_analytic_hessian_is_used_instead_of_autograd(self):
        """Spy on both: the native Hessian wins and ``energy`` is never called."""
        mol = _water()
        n_atoms = mol.GetNumAtoms()
        native = torch.arange((n_atoms * 3) ** 2, dtype=torch.double).reshape(
            n_atoms, 3, n_atoms, 3
        )
        # Symmetric, because VibrationsData validates that.
        native = 0.5 * (
            native + native.reshape(n_atoms * 3, n_atoms * 3).T.reshape(n_atoms, 3, n_atoms, 3)
        )
        adapter = _RecordingAdapter(analytic=native)

        vib = thermo_mod.vib_hessian(mol, _InertCalculator(), adapter, torch.device("cpu"))

        assert "analytic_hessian" in adapter.calls
        assert "energy" not in adapter.calls, (
            "the native analytic Hessian was computed AND then discarded in "
            "favour of differentiating energy()"
        )
        assert np.allclose(vib.get_hessian_2d(), native.reshape(n_atoms * 3, n_atoms * 3).numpy())

    def test_the_default_capability_is_none_not_an_exception(self):
        """Every in-tree adapter inherits "no native Hessian" from the base."""
        from Auto3D.models.adapter import BaseModelAdapter

        assert (
            BaseModelAdapter.analytic_hessian(
                object(), torch.zeros(1, 1, 3), torch.zeros(1, 1), torch.zeros(1)
            )
            is None
        )

    def test_the_aimnet_calculator_escape_hatch_is_gone(self):
        """``AIMNet2Adapter.calculator`` existed only to reach the native Hessian.

        With ``analytic_hessian`` on the contract there is no reason for a
        third-party calculator type to leak into Auto3D's control flow, and
        ``vib_hessian`` no longer imports one.
        """
        from Auto3D.models.adapter import AIMNet2Adapter

        assert not hasattr(AIMNet2Adapter, "calculator")
        # Bytecode, not source text: the docstring names the removed type on
        # purpose, to record what the capability replaced.
        referenced = thermo_mod.vib_hessian.__code__.co_names
        assert "AIMNet2Calculator" not in referenced
        assert "isinstance" not in referenced, "vib_hessian is dispatching on a type again"


class TestTheHessianPathStaysFloat64:
    """The silent-fp32-revert guard (B1 risk 10).

    ``ANI2xAdapter.forward`` and ``CustomModelAdapter.forward`` call
    ``coords.float()``. Routing the Hessian through ``forward`` would answer an
    fp64 request in fp32 with no error and no warning -- only a wrong number.
    """

    def test_energy_receives_the_dtype_the_hessian_was_built_at(self):
        mol = _water()
        adapter = _RecordingAdapter()

        thermo_mod.vib_hessian(mol, _InertCalculator(), adapter, torch.device("cpu"))

        assert adapter.coord_dtypes, "the adapter was never called"
        assert set(adapter.coord_dtypes) == {torch.float64}, (
            f"the Hessian was built at {set(adapter.coord_dtypes)}, not float64"
        )


class TestLoadHessianModelReturnsOneType:
    """``_load_hessian_model`` returned an adapter OR a calculator; now: an adapter."""

    def _fake_factory(self, monkeypatch):
        calls: dict = {}

        class _FakeModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

        class _FakeAdapter(AdapterModuleMixin, nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _FakeModule()

            def forward(self, coords, species, charges, atom_mask=None):
                energy = coords.pow(2).sum(dim=(1, 2))
                return energy, torch.zeros_like(coords)

        def _create(name, device, compile_model=None, use_cache=True):
            calls["args"] = (name, device, compile_model, use_cache)
            return _FakeAdapter()

        monkeypatch.setattr(thermo_mod, "create_model", _create)
        return calls

    @pytest.mark.parametrize("name", ["ANI2xt", "ani2x", "AIMNET", "aimnet2-2025", "aimnet2-nse"])
    def test_every_engine_name_yields_a_conforming_adapter(self, name, monkeypatch):
        """One return type. An aimnet registry alias no longer needs a branch.

        ``aimnet2-2025`` used to reach ``aimnet_hessian_helper``'s ``else`` and
        raise "cannot evaluate model_name"; there is no name-keyed evaluator
        left to fall through.
        """
        self._fake_factory(monkeypatch)

        result = thermo_mod._load_hessian_model(name, torch.device("cpu"))

        assert missing_adapter_members(result) == [], (
            f"_load_hessian_model({name!r}) returned "
            f"{type(result).__name__}, which does not satisfy ModelAdapter"
        )

    def test_the_ani_branch_still_upcasts_in_place_and_uncached(self, monkeypatch):
        """fp64 for the autograd Hessian, and never a cached instance.

        The factory cache is shared with the fp32 adapter the optimization half
        of the same ``calc_thermo`` call uses, so upcasting a cached entry would
        silently move that run to two precisions (B1 risk 6).
        """
        calls = self._fake_factory(monkeypatch)

        result = thermo_mod._load_hessian_model("ANI2xt", torch.device("cpu"))

        assert calls["args"] == ("ANI2xt", torch.device("cpu"), False, False)
        assert result.model.weight.dtype == torch.float64
