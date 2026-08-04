"""A single-point energy must not pay for a backward pass it discards (M39).

``calc_spe`` called ``EnForce_ANI.forward_batched`` and threw the forces away:

    es, fs = model.forward_batched(...)   # fs is never read again

Every adapter's ``forward`` derives forces unconditionally -- ``ANI2xtAdapter``,
``ANI2xAdapter`` and ``CustomModelAdapter`` each call ``torch.autograd.grad``,
and ``AIMNet2Adapter`` asks its calculator for ``forces=True`` -- so an SPE ran
a full backward pass per sub-batch for a tensor nobody looked at.

``ModelAdapter.energy`` already exists (it is what the autograd-Hessian path
differentiates) and is energy-only. These tests pin an energy-only *batched*
path built on it, and pin the two properties that make it safe to use:

* the energies are **bit-identical** to ``forward_batched``'s -- SPE writes
  ``E_hartree`` into an SDF, so a value that moved would be a silently wrong
  number, not a performance detail;
* a non-finite energy is still rejected. ``forward``'s ``_validate_outputs``
  was the only NaN gate on this path, and skipping it would have turned
  ``auto3d energy``'s exit-5 diagnosis into an SDF full of ``nan``.

Nothing here loads a neural network potential.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from Auto3D.batch_opt.model_wrapper import EnForce_ANI
from Auto3D.exceptions import NumericalError
from tests.helpers_adapter import AdapterModuleMixin, FakeAdapter


class _GradCountingAdapter(AdapterModuleMixin, nn.Module):
    """An adapter shaped like the real ones: ``forward`` backwards, ``energy`` does not.

    ``E = sum(coords**2)`` per molecule, so ``forward``'s autograd force is
    ``-2*coords`` and ``energy`` is the same energy computed without a
    ``torch.autograd.grad`` call -- the exact split every shipped adapter has.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sub_batch_sizes: list[int] = []

    def _energies(self, coords: torch.Tensor) -> torch.Tensor:
        return coords.pow(2).sum(dim=(1, 2))

    def forward(self, coords, species, charges, atom_mask=None):
        self.sub_batch_sizes.append(int(coords.shape[0]))
        coords = coords if coords.requires_grad else coords.requires_grad_(True)
        energy = self._energies(coords)
        grad = torch.autograd.grad([energy.sum()], [coords], create_graph=False)[0]
        return energy, -grad

    def energy(self, coords, species, charges, atom_mask=None):
        self.sub_batch_sizes.append(int(coords.shape[0]))
        return self._energies(coords)


def _batch(n_mols: int = 6, n_atoms: int = 4):
    coords = torch.arange(
        n_mols * n_atoms * 3, dtype=torch.float32
    ).reshape(n_mols, n_atoms, 3).requires_grad_(True)
    species = torch.ones(n_mols, n_atoms, dtype=torch.long)
    charges = torch.zeros(n_mols)
    atom_mask = torch.ones(n_mols, n_atoms, dtype=torch.bool)
    return coords, species, charges, atom_mask


def _count_autograd_grad(monkeypatch) -> list[int]:
    """Count every ``torch.autograd.grad`` call, wherever it is made from."""
    calls: list[int] = []
    real = torch.autograd.grad

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _spy)
    return calls


class TestEnergyBatchedSkipsTheBackwardPass:
    """The finding itself, measured rather than asserted by inspection."""

    def test_forward_batched_does_run_a_backward_pass(self, monkeypatch):
        """The premise. Without this, the next test proves nothing."""
        calls = _count_autograd_grad(monkeypatch)
        wrapper = EnForce_ANI(_GradCountingAdapter(), batchsize_atoms=8)

        wrapper.forward_batched(*_batch())

        assert sum(calls) > 0, "test premise: forward_batched must backward"

    def test_energy_batched_runs_no_backward_pass(self, monkeypatch):
        calls = _count_autograd_grad(monkeypatch)
        wrapper = EnForce_ANI(_GradCountingAdapter(), batchsize_atoms=8)

        wrapper.energy_batched(*_batch())

        assert sum(calls) == 0, (
            f"energy_batched still ran {sum(calls)} backward pass(es); the "
            "point of the energy-only path is that it runs none"
        )

    def test_the_energies_are_bit_identical_to_forward_batched(self):
        """Exact equality, not a tolerance: the operations are the same ones.

        ``calc_spe`` writes this number into ``E_hartree``. An energy-only path
        that changed it by an ULP would be a wrong reported energy, and the
        arithmetic here is identical, so ``torch.equal`` is the right assertion.
        """
        wrapper = EnForce_ANI(_GradCountingAdapter(), batchsize_atoms=8)

        es_forward, _ = wrapper.forward_batched(*_batch())
        es_energy = wrapper.energy_batched(*_batch())

        assert torch.equal(es_forward.detach(), es_energy.detach())

    def test_it_splits_into_exactly_the_same_sub_batches(self):
        """Same batching, so memory behavior is unchanged -- only the backward goes."""
        forward_adapter = _GradCountingAdapter()
        energy_adapter = _GradCountingAdapter()

        EnForce_ANI(forward_adapter, batchsize_atoms=8).forward_batched(*_batch())
        EnForce_ANI(energy_adapter, batchsize_atoms=8).energy_batched(*_batch())

        assert energy_adapter.sub_batch_sizes == forward_adapter.sub_batch_sizes
        assert len(forward_adapter.sub_batch_sizes) > 1, (
            "test premise: batchsize_atoms=8 with 4 atoms/mol must split"
        )

    def test_the_mask_is_sliced_per_sub_batch_just_like_forward_batched(self):
        """A padded batch must not lose its explicit mask (audit C13)."""
        seen: list[int] = []

        class _MaskAdapter(AdapterModuleMixin, nn.Module):
            def forward(self, coords, species, charges, atom_mask=None):
                raise AssertionError("energy_batched must not call forward")

            def energy(self, coords, species, charges, atom_mask=None):
                assert atom_mask is not None, "the mask must survive the split"
                seen.append(int(atom_mask.sum()))
                return coords.pow(2).sum(dim=(1, 2))

        coords, species, charges, atom_mask = _batch(n_mols=3, n_atoms=4)
        atom_mask[1, 3] = False
        atom_mask[2, 2:] = False

        EnForce_ANI(_MaskAdapter(), batchsize_atoms=4).energy_batched(
            coords, species, charges, atom_mask=atom_mask
        )

        assert seen == [4, 3, 2]


class TestTheOneValueThatMoves:
    """A custom NNP returning float64 energies is no longer rounded to float32.

    ``CustomModelAdapter.forward`` ends with ``energy.to(input_dtype)``, and
    ``input_dtype`` on this path is float32 (``pad_from_mols`` builds float32
    coordinates), so an SPE over a model that computes in double precision used
    to have its energy round-tripped through float32 before ``E_hartree`` was
    written -- a relative change of up to ~6e-8, around 1e-6 kcal/mol at a
    typical total energy. ``energy`` is dtype-preserving by contract (that is why
    the override exists at all: the Hessian path must not be answered in fp32),
    so the reported value is now the model's own.

    This is the ONLY reported number the M39 change moves. AIMNet2 (energies
    already float64), ANI2x (float32 both ways) and ANI2xt (float64 both ways)
    are bit-identical. Pinned here so the difference is a recorded decision
    rather than something a future reader has to rediscover.
    """

    @staticmethod
    def _fp64_custom_adapter():
        from Auto3D.models.adapter import CustomModelAdapter

        class _Fp64Model(nn.Module):
            def forward(self, species, coords, charges):
                # A value that is NOT representable in float32, so the two paths
                # differ observably rather than by luck. Kept connected to
                # `coords` (times zero) because `forward` differentiates it.
                zero = coords.to(torch.float64).sum(dim=(1, 2)) * 0.0
                return zero + (1.0 + 2.0**-40)

        adapter = CustomModelAdapter.__new__(CustomModelAdapter)
        nn.Module.__init__(adapter)
        adapter.model = _Fp64Model()
        adapter.coord_pad, adapter.species_pad = 0.0, -1
        return adapter

    def test_the_forward_path_rounded_it_to_float32(self):
        """The premise: this is what the old call site reported."""
        wrapper = EnForce_ANI(self._fp64_custom_adapter(), batchsize_atoms=1024)
        es, _ = wrapper.forward_batched(*_batch(n_mols=1, n_atoms=2))
        assert es.dtype is torch.float32
        assert float(es[0].detach()) == 1.0

    def test_the_energy_path_reports_the_models_own_value(self):
        wrapper = EnForce_ANI(self._fp64_custom_adapter(), batchsize_atoms=1024)
        es = wrapper.energy_batched(*_batch(n_mols=1, n_atoms=2))
        assert es.dtype is torch.float64
        assert float(es[0].detach()) == 1.0 + 2.0**-40


class TestEnergyBatchedStillGuardsTheNumbers:
    """``forward``'s ``_validate_outputs`` was this path's only NaN gate."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_energy_is_rejected(self, bad):
        class _BadAdapter(AdapterModuleMixin, nn.Module):
            def forward(self, coords, species, charges, atom_mask=None):
                raise AssertionError("unused")

            def energy(self, coords, species, charges, atom_mask=None):
                return torch.full((coords.shape[0],), bad)

        wrapper = EnForce_ANI(_BadAdapter(), batchsize_atoms=1024)

        with pytest.raises(NumericalError):
            wrapper.energy_batched(*_batch(n_mols=2, n_atoms=3))

    def test_finite_energies_pass(self):
        wrapper = EnForce_ANI(FakeAdapter(), batchsize_atoms=1024)
        es = wrapper.energy_batched(*_batch(n_mols=2, n_atoms=3))
        assert torch.isfinite(es).all()


class TestEnergyBatchedKeepsTheOomBehavior:
    """The shrink-and-stay-shrunk retry (audit M37) is shared, not duplicated."""

    def test_an_oom_halves_the_batch_and_the_run_completes(self):
        class _OomOnceAdapter(AdapterModuleMixin, nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sizes: list[int] = []
                self.raised = False

            def forward(self, coords, species, charges, atom_mask=None):
                raise AssertionError("unused")

            def energy(self, coords, species, charges, atom_mask=None):
                n = int(coords.shape[0])
                if not self.raised and n == 4:
                    self.raised = True
                    raise torch.cuda.OutOfMemoryError("simulated")
                self.sizes.append(n)
                return coords.pow(2).sum(dim=(1, 2))

        adapter = _OomOnceAdapter()
        # 4 atoms/mol, batchsize_atoms=16 -> 4 molecules per sub-batch.
        wrapper = EnForce_ANI(adapter, batchsize_atoms=16)

        es = wrapper.energy_batched(*_batch(n_mols=8, n_atoms=4))

        assert adapter.raised, "test premise: the first sub-batch must OOM"
        assert es.shape == (8,)
        assert adapter.sizes == [2, 2, 2, 2], (
            f"the halved batch size did not stick: {adapter.sizes}"
        )

    def test_a_single_molecule_that_still_ooms_raises_a_named_error(self):
        from Auto3D.exceptions import OptimizationError

        class _AlwaysOom(AdapterModuleMixin, nn.Module):
            def forward(self, coords, species, charges, atom_mask=None):
                raise AssertionError("unused")

            def energy(self, coords, species, charges, atom_mask=None):
                raise torch.cuda.OutOfMemoryError("simulated")

        wrapper = EnForce_ANI(_AlwaysOom(), batchsize_atoms=4)

        with pytest.raises(OptimizationError, match="batch size 1"):
            wrapper.energy_batched(*_batch(n_mols=2, n_atoms=4))


class TestEveryShippedAdapterEnergyIsBackwardFree:
    """The saving is real only if the adapters' own ``energy`` avoids autograd.

    ``AIMNet2Adapter`` is the deliberate exception and is asserted as such: its
    ``energy`` is contractually ``forward(...)[0]`` (the calculator's
    ``forces=True`` route), because that is the route documented to stay
    connected to ``coord`` in the autograd graph for a Hessian caller. Changing
    it would move the AIMNet2 energy onto the calculator's ``forces=False``
    external-module code path, which cannot be shown equal without loading a
    real model.
    """

    @staticmethod
    def _bypassed(cls, model):
        """An adapter instance with no weights loaded (``__init__`` skipped)."""
        adapter = cls.__new__(cls)
        nn.Module.__init__(adapter)
        adapter.model = model
        adapter.coord_pad = 0.0
        adapter.species_pad = -1
        return adapter

    def test_ani2xt_energy_runs_no_backward(self, monkeypatch):
        from Auto3D.models.adapter import ANI2xtAdapter

        class _Toy(nn.Module):
            def forward(self, species, coords, **kwargs):
                return coords.pow(2).sum(dim=(1, 2))

        calls = _count_autograd_grad(monkeypatch)
        adapter = self._bypassed(ANI2xtAdapter, _Toy())
        adapter.energy(
            torch.ones(1, 2, 3, dtype=torch.double),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1),
        )
        assert sum(calls) == 0

    def test_ani2x_energy_runs_no_backward(self, monkeypatch):
        from Auto3D.models.adapter import ANI2xAdapter

        class _Toy(nn.Module):
            def forward(self, species_coords):
                _species, coords = species_coords
                return type(
                    "SpeciesEnergies", (), {"energies": coords.pow(2).sum(dim=(1, 2))}
                )()

        calls = _count_autograd_grad(monkeypatch)
        adapter = self._bypassed(ANI2xAdapter, _Toy())
        adapter.energy(
            torch.ones(1, 2, 3, dtype=torch.double),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1),
        )
        assert sum(calls) == 0

    def test_custom_energy_runs_no_backward(self, monkeypatch):
        from Auto3D.models.adapter import CustomModelAdapter

        class _Toy(nn.Module):
            def forward(self, species, coords, charges):
                return coords.pow(2).sum(dim=(1, 2))

        calls = _count_autograd_grad(monkeypatch)
        adapter = self._bypassed(CustomModelAdapter, _Toy())
        adapter.energy(
            torch.ones(1, 2, 3, dtype=torch.double),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1),
        )
        assert sum(calls) == 0

    def test_aimnet2_energy_is_deliberately_the_forward_route(self):
        import inspect

        from Auto3D.models.adapter import AIMNet2Adapter

        source = inspect.getsource(AIMNet2Adapter.energy)
        assert "self.forward(" in source, (
            "AIMNet2Adapter.energy stopped routing through forward; that moves "
            "the default engine's energy onto the calculator's forces=False "
            "path, which needs a real-model equality check to justify"
        )


class TestCalcSpeUsesTheEnergyOnlyPath:
    """The call site itself, so the wiring cannot silently revert."""

    def test_calc_spe_never_calls_forward_batched(self, monkeypatch, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.SPE as spe_mod

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "ethanol")
        sdf = tmp_path / "in.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(mol)

        class _EnergyOnlyEnForce:
            def __init__(self, adapter, *a, **k):
                pass

            def forward_batched(self, *a, **k):
                raise AssertionError(
                    "calc_spe computed forces it does not use (M39)"
                )

            def energy_batched(self, coords, numbers, charges, atom_mask=None):
                return torch.full((coords.shape[0],), -1.0, dtype=torch.double)

        monkeypatch.setattr(
            spe_mod, "get_device", lambda *a, **k: torch.device("cpu")
        )
        monkeypatch.setattr(
            spe_mod, "create_model", lambda *a, **k: FakeAdapter()
        )
        monkeypatch.setattr(spe_mod, "EnForce_ANI", _EnergyOnlyEnForce)

        out = tmp_path / "out.sdf"
        spe_mod.calc_spe(
            str(sdf), "AIMNET", use_gpu=False, out_path=str(out)
        )

        written = [m for m in Chem.SDMolSupplier(str(out), removeHs=False)]
        assert len(written) == 1
        from Auto3D.constants import EV_TO_HARTREE

        assert float(written[0].GetProp("E_hartree")) == pytest.approx(
            -1.0 * EV_TO_HARTREE
        )
