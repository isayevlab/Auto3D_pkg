# tests/test_model_adapter.py
"""Unit tests for the Model Adapter module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Adapter classes are imported locally inside the tests that need them; the
# real-AIMNet2 tests now reuse the session-scoped ``aimnet_model`` fixture
# (see tests/conftest.py) so the model is loaded once per session, not per test.


@pytest.mark.slow
def test_model_adapter_interface(aimnet_model):
    """A concrete adapter should expose the ModelAdapter interface and a
    forward(coords, species, charges) -> (energy, forces) signature.

    Uses the session-scoped ``aimnet_model`` fixture (an ``AIMNet2Adapter``
    built once via ``create_model("AIMNET", ...)``) instead of loading the real
    AIMNet2 model per-test (~7s). The test is read-only (reads attributes and
    calls ``forward``), so sharing the adapter is safe.
    """
    device = torch.device("cpu")
    adapter = aimnet_model

    # Test interface attributes exist
    assert hasattr(adapter, "coord_pad")
    assert hasattr(adapter, "species_pad")
    assert hasattr(adapter, "device")

    # Test forward signature on two real methane molecules.
    coords = (
        torch.tensor(
            [
                [
                    [0.0, 0, 0],
                    [0.63, 0.63, 0.63],
                    [-0.63, -0.63, 0.63],
                    [0.63, -0.63, -0.63],
                    [-0.63, 0.63, -0.63],
                ]
            ]
        )
        .repeat(2, 1, 1)
        .to(device)
    )
    species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]], device=device)
    charges = torch.tensor([0.0, 0.0], device=device)

    energy, forces = adapter.forward(coords, species, charges)
    assert energy.shape == (2,)
    assert forces.shape == (2, 5, 3)


class TestModelAdapterProtocol:
    """Tests for the ModelAdapter protocol.

    It lives in ``Auto3D.models.contract`` alongside ``CustomNNP`` -- the
    interface it is constantly confused with -- so the two declarations cannot be
    read separately. ``Auto3D.models.adapter`` holds implementations only.
    """

    def test_protocol_declares_the_whole_contract(self):
        from Auto3D.models.contract import ModelAdapter

        for member in ("forward", "energy", "to_species"):
            assert hasattr(ModelAdapter, member)
        assert tuple(ModelAdapter.__annotations__) == ("coord_pad", "species_pad")

    def test_protocol_does_not_live_in_the_implementation_module(self):
        """A clean sweep: the old import path is gone, not aliased."""
        import Auto3D.models.adapter as adapter_mod

        assert not hasattr(adapter_mod, "ModelAdapter")

    def test_the_package_does_not_re_export_it(self):
        """``Auto3D.models.contract`` is the only path, in both directions.

        This test previously asserted the opposite -- that ``Auto3D.models``
        re-exported the name. That re-export put the *internal* adapter
        interface at a shallower path than the *public* custom-NNP contract
        (``Auto3D.models.contract.CustomNNP``, api.rst's only entry from this
        package), which is the precise confusion ``contract.py`` was created to
        end. ``Auto3D.models`` re-exports nothing now; see
        ``tests/test_import_boundaries.py::test_models_package_exposes_no_names``
        for the other six names that went with it.
        """
        import importlib

        import pytest

        models = importlib.import_module("Auto3D.models")
        assert not hasattr(models, "__all__")
        with pytest.raises(ImportError):
            exec("from Auto3D.models import ModelAdapter", {})  # noqa: S102

    def test_device_is_not_part_of_the_contract(self):
        """Dropped deliberately.

        Nothing outside an adapter reads ``adapter.device``
        (``BaseModelAdapter`` keeps ``self.device`` as an implementation detail),
        and requiring it would make every legitimate structural implementation --
        including test doubles that never touch a device -- non-conforming for no
        benefit.
        """
        from Auto3D.models.contract import ModelAdapter

        assert "device" not in ModelAdapter.__annotations__

    def test_issubclass_is_never_a_valid_question(self):
        """Pinned so nobody "improves" the EnForce_ANI gate into an issubclass:
        a Protocol with data members raises for it."""
        import pytest

        from Auto3D.models.contract import ModelAdapter

        with pytest.raises(TypeError):
            issubclass(dict, ModelAdapter)


class TestBaseModelAdapter:
    """Tests for the BaseModelAdapter base class."""

    def test_base_adapter_stores_model_and_device(self):
        """BaseModelAdapter should store model, device, and padding values."""
        from Auto3D.models.adapter import BaseModelAdapter

        # Create a mock model
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.parameters.return_value = iter([])

        device = torch.device("cpu")

        # We can't instantiate abstract class directly, so we need a concrete subclass
        class ConcreteAdapter(BaseModelAdapter):
            def forward(self, coords, species, charges):
                return torch.zeros(coords.shape[0]), torch.zeros_like(coords)

        adapter = ConcreteAdapter(mock_model, device, coord_pad=1.0, species_pad=-1)

        assert adapter.model == mock_model
        assert adapter.device == device
        assert adapter.coord_pad == 1.0
        assert adapter.species_pad == -1


@pytest.mark.slow
class TestAIMNet2Adapter:
    """Tests for the AIMNet2Adapter (aimnet-backed)."""

    def test_aimnet2_adapter_loads_default_model(self, aimnet_model):
        """AIMNet2Adapter resolves the 'aimnet2' registry name (no .jpt path).

        Uses the shared session ``aimnet_model`` fixture (built via
        ``create_model("AIMNET", ...)``, which resolves to the ``aimnet2``
        registry default) rather than reloading the real model. Read-only.
        """
        adapter = aimnet_model

        assert adapter.model_name == "aimnet2"
        # An underlying nn.Module is built from the aimnet registry.
        assert adapter.model is not None

    def test_aimnet2_adapter_has_correct_padding(self, aimnet_model):
        """AIMNet2Adapter should have coord_pad=0.0 and species_pad=0.

        Reuses the shared session ``aimnet_model`` fixture (read-only).
        """
        adapter = aimnet_model

        assert adapter.coord_pad == 0.0
        assert adapter.species_pad == 0

    # Note: the former test_aimnet_adapter_forward_calls_model (which mocked a
    # jit-loaded model and inspected the dict passed to it) is intentionally
    # dropped. The new adapter delegates to AIMNet2Calculator, and a real
    # forward pass is already covered end-to-end by
    # test_aimnet2_adapter_energy_forces_water / _padded_batch_matches_unpadded
    # below, which assert energy/force shapes and physical values.


class TestANI2xtAdapter:
    """Tests for the ANI2xt adapter."""

    def test_ani2xt_adapter_creates_model(self):
        """ANI2xtAdapter should create ANI2xt model."""
        # Import torchani to check if it's available (needed for ANI2xt)
        pytest.importorskip("torchani")

        from Auto3D.models.adapter import ANI2xtAdapter

        device = torch.device("cpu")
        adapter = ANI2xtAdapter(device)

        assert adapter.species_pad == -1
        assert adapter.coord_pad == 0.0

    def test_ani2xt_adapter_force_sign_with_toy_model(self):
        """``forces = -grad`` is duplicated once per adapter in adapter.py:
        ``ANI2xtAdapter.forward`` has its own copy, distinct from (and
        untested by) ``CustomModelAdapter``'s copy that
        ``test_custom_model_adapter_runs`` already checks (audit M32). A sign
        bug introduced independently in THIS copy would not be caught by that
        test, and ``test_ani2xt_adapter_creates_model`` above never calls
        ``.forward`` at all.

        ``BaseModelAdapter.__init__`` is called directly on a bypassed
        instance (skipping ``ANI2xtAdapter.__init__``, which imports the real
        bundled ANI2xt weights) and handed a toy quadratic model instead --
        same technique ``TestBaseModelAdapter`` already uses for a mock
        model, applied here so the REAL ``ANI2xtAdapter.forward`` runs.
        Hermetic: no NNP loaded, no torchani import.
        """
        from Auto3D.models.adapter import ANI2xtAdapter, BaseModelAdapter

        class _ToyANI2xtModel(torch.nn.Module):
            def forward(self, species, coords):
                return (coords**2).sum(dim=(1, 2))

        device = torch.device("cpu")
        adapter = ANI2xtAdapter.__new__(ANI2xtAdapter)
        BaseModelAdapter.__init__(adapter, _ToyANI2xtModel(), device, coord_pad=0.0, species_pad=-1)

        coords = torch.randn(2, 4, 3)
        species = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])
        charges = torch.zeros(2)
        energy, forces = adapter.forward(coords, species, charges)

        # _ToyANI2xtModel does not mask padding: E = sum(coords^2) over every
        # slot => dE/dx = 2*coords => F = -dE/dx = -2*coords, exactly like
        # test_custom_model_adapter_runs's reference calculation.
        torch.testing.assert_close(energy, (coords**2).sum(dim=(1, 2)), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(forces, -2.0 * coords, rtol=1e-5, atol=1e-6)


class TestANI2xAdapter:
    """Tests for the ANI2x adapter."""

    def test_ani2x_adapter_creates_model(self):
        """ANI2xAdapter should create ANI2x model from torchani."""
        # Import torchani to check if it's available
        pytest.importorskip("torchani")

        from Auto3D.models.adapter import ANI2xAdapter

        device = torch.device("cpu")
        adapter = ANI2xAdapter(device)

        # Verify adapter has correct padding values
        assert adapter.species_pad == -1
        assert adapter.coord_pad == 0.0

    def test_ani2x_adapter_force_sign_with_toy_model(self):
        """``forces = -grad`` is duplicated again in ``ANI2xAdapter.forward``
        -- a third, separate copy from ``ANI2xtAdapter``'s and
        ``CustomModelAdapter``'s (audit M32), also untested by
        ``test_custom_model_adapter_runs``.

        The toy model mimics torchani's ``SpeciesEnergies`` return shape (an
        object with a ``.energies`` attribute) rather than
        ``ANI2xtAdapter``'s plain-tensor return, since ``ANI2xAdapter.forward``
        calls ``self.model((species, coords)).energies`` and multiplies by
        ``HARTREE_TO_EV`` -- the toy divides by the same constant first so the
        expected force in eV is still the clean ``-2*coords``. Coordinates are
        float32 from the start (matching what ``ANI2xAdapter.forward`` casts
        to internally) so the adapter's own ``coords.float()`` cast is a
        no-op here and cannot be blamed for any looseness in the comparison
        (the brainstorm's dtype-cast risk flag for this specific test).
        Hermetic: no NNP loaded, no torchani import.
        """
        from collections import namedtuple

        from Auto3D.constants import HARTREE_TO_EV
        from Auto3D.models.adapter import ANI2xAdapter, BaseModelAdapter

        _SpeciesEnergies = namedtuple("SpeciesEnergies", ["species", "energies"])

        class _ToyANI2xModel(torch.nn.Module):
            def forward(self, species_coords):
                species, coords = species_coords
                energies = (coords**2).sum(dim=(1, 2)) / HARTREE_TO_EV
                return _SpeciesEnergies(species, energies)

        device = torch.device("cpu")
        adapter = ANI2xAdapter.__new__(ANI2xAdapter)
        BaseModelAdapter.__init__(adapter, _ToyANI2xModel(), device, coord_pad=0.0, species_pad=-1)

        coords = torch.randn(2, 4, 3, dtype=torch.float32)
        species = torch.tensor([[1, 6, 7, 8], [1, 6, 7, -1]])
        charges = torch.zeros(2)
        energy, forces = adapter.forward(coords, species, charges)

        torch.testing.assert_close(energy, (coords**2).sum(dim=(1, 2)), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(forces, -2.0 * coords, rtol=1e-5, atol=1e-6)


class TestCustomModelAdapter:
    """Tests for the CustomModelAdapter."""

    @patch.object(torch.jit, "load")
    def test_custom_adapter_loads_from_path(self, mock_load):
        """CustomModelAdapter should load model from provided path."""
        from Auto3D.models.adapter import CustomModelAdapter

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_model.coord_pad = 1.0
        mock_model.species_pad = -2
        # `load_custom_nnp` puts everything it returns in eval mode, so the
        # double has to answer .eval() the way a real nn.Module does -- with
        # itself -- rather than with a fresh MagicMock.
        mock_model.eval.return_value = mock_model
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        adapter = CustomModelAdapter("/path/to/model.pt", device)

        mock_load.assert_called_once_with("/path/to/model.pt", map_location=device)
        assert adapter.coord_pad == 1.0
        assert adapter.species_pad == -2

    @patch.object(torch.jit, "load")
    def test_custom_adapter_rejects_a_model_without_padding_attributes(self, mock_load):
        """No silent default: the padding values must come from the model.

        Until audit C12 this fell back to getattr defaults that DISAGREED
        between layers -- CustomModelAdapter substituted species_pad=-1 while
        BaseModelAdapter's own default was 0 -- so which atoms counted as
        padding depended on which layer supplied the value. Refusing is the
        only answer that cannot be silently wrong.
        """
        from Auto3D.exceptions import ModelLoadError
        from Auto3D.models.adapter import CustomModelAdapter

        # A model without coord_pad/species_pad.
        class MockModel:
            def parameters(self):
                return iter([])

            def eval(self):  # every real nn.Module has this; the loader calls it
                return self

        mock_load.return_value = MockModel()

        device = torch.device("cpu")
        with pytest.raises(ModelLoadError, match="coord_pad"):
            CustomModelAdapter("/path/to/model.pt", device)


def test_try_compile_uses_dynamic_default_mode(monkeypatch):
    import Auto3D.models.adapter as adapter

    captured = {}

    def fake_compile(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(adapter.torch, "compile", fake_compile)
    import torch.nn as nn

    m = nn.Linear(2, 2)
    adapter._try_compile(m)
    assert captured.get("mode") == "default"
    assert captured.get("dynamic") is True


@pytest.mark.slow
def test_aimnet2_adapter_energy_forces_water(aimnet_model):
    """Reuses the shared session ``aimnet_model`` adapter (read-only forward)
    instead of reloading the real AIMNet2 model (~7s)."""
    import torch

    ad = aimnet_model
    coord = torch.tensor([[[0.0, 0, 0], [0, 0, 0.97], [0, 0.92, -0.25]]])
    species = torch.tensor([[8, 1, 1]])
    charges = torch.tensor([0.0])
    e, f = ad.forward(coord, species, charges)
    assert e.shape == (1,)
    assert f.shape == (1, 3, 3)
    assert -3000 < float(e[0]) < -1000  # water total energy, eV
    assert ad.species_pad == 0 and ad.coord_pad == 0.0


@pytest.mark.slow
def test_aimnet2_adapter_padded_batch_matches_unpadded(aimnet_model):
    """Padded multi-size batch must give per-molecule energies equal to solo runs.

    Reuses the shared session ``aimnet_model`` adapter (read-only forward).
    """
    import torch

    ad = aimnet_model

    water_c = torch.tensor([[0.0, 0, 0], [0, 0, 0.97], [0, 0.92, -0.25]])
    water_n = torch.tensor([8, 1, 1])
    meth_c = torch.tensor(
        [
            [0.0, 0, 0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [0.63, -0.63, -0.63],
            [-0.63, 0.63, -0.63],
        ]
    )
    meth_n = torch.tensor([6, 1, 1, 1, 1])

    e_w, _ = ad.forward(water_c.unsqueeze(0), water_n.unsqueeze(0), torch.zeros(1))
    e_m, _ = ad.forward(meth_c.unsqueeze(0), meth_n.unsqueeze(0), torch.zeros(1))

    # padded batch: water padded to 5 with species_pad=0. The real-atom mask
    # is passed explicitly (as pad_from_mols returns it); the adapter must not
    # re-derive it from `species == species_pad`, which would also delete a
    # legitimate atomic number 0 (an R-group `*` atom) -- audit C13.
    bc = torch.zeros(2, 5, 3)
    bc[0, :3] = water_c
    bc[1, :5] = meth_c
    bn = torch.zeros(2, 5, dtype=torch.long)
    bn[0, :3] = water_n
    bn[1, :5] = meth_n
    bm = torch.zeros(2, 5, dtype=torch.bool)
    bm[0, :3] = True
    bm[1, :5] = True
    e_b, f_b = ad.forward(bc, bn, torch.zeros(2), atom_mask=bm)
    assert f_b.shape == (2, 5, 3)
    assert abs(float(e_b[0]) - float(e_w[0])) < 1e-2  # padded water == solo water (NaN-free!)
    assert abs(float(e_b[1]) - float(e_m[0])) < 1e-2
    # padded slots of water (rows 3,4) carry zero force
    assert torch.allclose(f_b[0, 3:], torch.zeros(2, 3), atol=1e-6)


def test_custom_model_adapter_runs(tmp_path):
    """Custom-NNP path: a scripted (species, coords, charges)->energies model
    must run through CustomModelAdapter and yield finite energy/forces."""
    import torch
    from Auto3D.models.adapter import CustomModelAdapter

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # INSTANCE, not class, attributes: torch.jit.save drops plain class
            # attributes, so a scripted model declaring them at class level
            # arrives with neither and is rejected by the contract check.
            self.coord_pad: float = 0.0
            self.species_pad: int = -1

        def forward(self, species, coords, charges):
            # simple harmonic-ish energy = sum of squared coords per molecule
            return (coords**2).sum(dim=(1, 2))

    p = tmp_path / "toy.pt"
    torch.jit.save(torch.jit.script(_Toy()), str(p))

    ad = CustomModelAdapter(str(p), torch.device("cpu"))
    coords = torch.randn(2, 4, 3)
    species = torch.tensor([[1, 6, 7, 8], [1, 1, 6, -1]])
    charges = torch.tensor([0.0, 0.0])
    e, f = ad.forward(coords, species, charges)
    assert e.shape == (2,)
    assert f.shape == (2, 4, 3)
    # _Toy does not mask padding, so E = sum(coords^2) over every slot =>
    # dE/dx = 2*coords => F = -dE/dx = -2*coords. Asserting the value, not
    # just finiteness, is what catches a sign flip in the adapter's force
    # path (audit M32).
    expected_forces = -2.0 * coords
    torch.testing.assert_close(f, expected_forces, rtol=1e-5, atol=1e-6)
    expected_energy = (coords**2).sum(dim=(1, 2))
    torch.testing.assert_close(e, expected_energy, rtol=1e-5, atol=1e-6)


class TestValidateOutputs:
    """_validate_outputs runs on every FIRE step; it must stay a no-op on finite
    inputs (single combined sync) and still raise on NaN/Inf with a clear message.
    """

    def test_finite_outputs_pass(self):
        from Auto3D.models.adapter import _validate_outputs

        energy = torch.tensor([-1.0, -2.0])
        forces = torch.zeros(2, 3, 3)
        assert _validate_outputs(energy, forces) is None

    def test_nan_energy_raises(self):
        from Auto3D.exceptions import NumericalError
        from Auto3D.models.adapter import _validate_outputs

        energy = torch.tensor([float("nan"), -2.0])
        forces = torch.zeros(2, 3, 3)
        with pytest.raises(NumericalError, match="NaN.*energy"):
            _validate_outputs(energy, forces)

    def test_inf_energy_raises(self):
        from Auto3D.exceptions import NumericalError
        from Auto3D.models.adapter import _validate_outputs

        energy = torch.tensor([float("inf"), -2.0])
        forces = torch.zeros(2, 3, 3)
        with pytest.raises(NumericalError, match="Inf.*energy"):
            _validate_outputs(energy, forces)

    def test_nan_forces_raises(self):
        from Auto3D.exceptions import NumericalError
        from Auto3D.models.adapter import _validate_outputs

        energy = torch.tensor([-1.0, -2.0])
        forces = torch.zeros(2, 3, 3)
        forces[1, 0, 0] = float("nan")
        with pytest.raises(NumericalError, match="NaN.*force"):
            _validate_outputs(energy, forces)

    def test_inf_forces_raises(self):
        from Auto3D.exceptions import NumericalError
        from Auto3D.models.adapter import _validate_outputs

        energy = torch.tensor([-1.0, -2.0])
        forces = torch.zeros(2, 3, 3)
        forces[0, 2, 1] = float("inf")
        with pytest.raises(NumericalError, match="Inf.*force"):
            _validate_outputs(energy, forces)


class TestAni2xtNetworksAreTableDriven:
    """The seven per-element MLPs are one factory + a width table (audit M61).

    They used to be seven copy-pasted ``nn.Sequential`` blocks, 69 lines
    differing only in three integers, which is how ``F_network`` and
    ``S_network`` ended up declared in the opposite order from the
    ``ModuleList`` they are placed into -- readable only by cross-checking two
    distant lines.

    The refactor is safe because ``nn.ModuleList.load_state_dict`` keys off
    POSITION (``"0.0.weight"``, ``"1.0.weight"``, ...), never off the Python
    variable a submodule was assigned to. That claim is not taken on trust: the
    test below loads the real shipped checkpoint into both the old hand-written
    structure and the new generated one and compares every tensor.
    """

    CHECKPOINT = "src/Auto3D/models/ani2xt_no_repulsion.pt"

    @staticmethod
    def _hand_written(aev_dim):
        """Verbatim transcription of the seven blocks as they were before M61,
        in their original ``ModuleList`` order (note F before S)."""
        from torch import nn

        def mlp(a, b, c, d):
            return nn.Sequential(
                nn.Linear(a, b),
                nn.CELU(0.1),
                nn.Linear(b, c),
                nn.CELU(0.1),
                nn.Linear(c, d),
                nn.CELU(0.1),
                nn.Linear(d, 1),
            )

        H_network = mlp(aev_dim, 256, 192, 160)
        C_network = mlp(aev_dim, 224, 192, 160)
        N_network = mlp(aev_dim, 192, 160, 128)
        O_network = mlp(aev_dim, 192, 160, 128)
        S_network = mlp(aev_dim, 160, 128, 96)
        F_network = mlp(aev_dim, 160, 128, 96)
        Cl_network = mlp(aev_dim, 160, 128, 96)
        return nn.ModuleList(
            [
                H_network,
                C_network,
                N_network,
                O_network,
                F_network,
                S_network,
                Cl_network,
            ]
        )

    def test_the_shipped_checkpoint_loads_identically_both_ways(self):
        """The tripwire for the whole refactor: same weights, tensor for tensor.

        Loading a ``state_dict`` is not running the model -- no inference, no
        torchani, no download; the checkpoint is bundled in
        ``src/Auto3D/models/``.
        """
        import torch
        from torch import nn

        from Auto3D.batch_opt.ANI2xt_no_rep import WIDTHS, _atomic_mlp

        checkpoint = torch.load(self.CHECKPOINT, map_location="cpu", weights_only=True)
        # Read the AEV width off the checkpoint rather than hardcoding it, so a
        # retrained model with a different AEV does not silently pass.
        aev_dim = checkpoint["0.0.weight"].shape[1]

        old = self._hand_written(aev_dim)
        new = nn.ModuleList([_atomic_mlp(aev_dim, widths) for widths in WIDTHS])

        old.load_state_dict(checkpoint)
        new.load_state_dict(checkpoint)

        old_state, new_state = old.state_dict(), new.state_dict()
        assert set(old_state) == set(new_state)
        for key in old_state:
            assert torch.equal(old_state[key], new_state[key]), (
                f"{key} differs: the ModuleList order changed, so this "
                f"checkpoint now routes an element to the wrong network"
            )

    def test_the_width_table_is_in_moduleList_order_including_f_before_s(self):
        """Fluorine at index 4 and sulfur at index 5, matching ANI2XT_INDEX.

        The old code declared ``S_network`` before ``F_network`` in the source but
        placed F before S in the ``ModuleList``. Since F, S and Cl happen to share
        the same widths the mix-up was harmless -- and undetectable. The table
        makes the order the only order there is.
        """
        from Auto3D.batch_opt.ANI2xt_no_rep import WIDTHS
        from Auto3D.models.species import ANI2XT_INDEX

        assert len(WIDTHS) == len(ANI2XT_INDEX) == 7
        # H, C, N, O, F, S, Cl
        assert WIDTHS == (
            (256, 192, 160),
            (224, 192, 160),
            (192, 160, 128),
            (192, 160, 128),
            (160, 128, 96),
            (160, 128, 96),
            (160, 128, 96),
        )

    def test_the_factory_builds_the_documented_shape(self):
        from torch import nn

        from Auto3D.batch_opt.ANI2xt_no_rep import _atomic_mlp

        net = _atomic_mlp(11, (5, 4, 3))
        kinds = [type(layer) for layer in net]
        assert kinds == [nn.Linear, nn.CELU, nn.Linear, nn.CELU, nn.Linear, nn.CELU, nn.Linear]
        assert [(l.in_features, l.out_features) for l in net if isinstance(l, nn.Linear)] == [
            (11, 5),
            (5, 4),
            (4, 3),
            (3, 1),
        ]
        assert all(l.alpha == 0.1 for l in net if isinstance(l, nn.CELU))


class TestEnergyIsDtypePreserving:
    """``energy()`` must answer in the dtype it was asked in.

    This is the single most likely silent numerical regression in the whole
    contract change, and it produces no error of any kind.
    ``ANI2xAdapter.forward`` and ``CustomModelAdapter.forward`` both call
    ``coords.float()`` -- correct for them, because they front float32 weights.
    But ``energy()`` exists so a caller can DIFFERENTIATE it (an fp64 Hessian,
    which ``ASE/thermo.py`` builds by promoting the wrapped module with
    ``.double()``). If ``energy`` were the inherited ``forward(...)[0]`` for those
    two adapters, an fp64 request would come back fp32 with nothing reported: the
    Hessian would simply be less accurate than the code around it promises.

    Each adapter is built by calling ``BaseModelAdapter.__init__`` on a bypassed
    instance and handing it a toy module that RECORDS the dtype it was fed -- the
    same technique the force-sign tests above use. Nothing is loaded, no torchani,
    no download.
    """

    @staticmethod
    def _bypassed(cls, model):
        from Auto3D.models.adapter import BaseModelAdapter

        adapter = cls.__new__(cls)
        BaseModelAdapter.__init__(
            adapter, model, torch.device("cpu"), coord_pad=0.0, species_pad=-1
        )
        return adapter

    def test_ani2x_energy_keeps_float64(self):
        from collections import namedtuple

        from Auto3D.constants import HARTREE_TO_EV
        from Auto3D.models.adapter import ANI2xAdapter

        _SpeciesEnergies = namedtuple("SpeciesEnergies", ["species", "energies"])
        seen: list = []

        class _Recorder(torch.nn.Module):
            def forward(self, species_coords):
                species, coords = species_coords
                seen.append(coords.dtype)
                return _SpeciesEnergies(species, (coords**2).sum(dim=(1, 2)) / HARTREE_TO_EV)

        adapter = self._bypassed(ANI2xAdapter, _Recorder())
        coords = torch.randn(2, 4, 3, dtype=torch.float64)
        species = torch.tensor([[1, 6, 7, 8], [1, 6, 7, -1]])
        charges = torch.zeros(2, dtype=torch.float64)

        energy = adapter.energy(coords, species, charges)
        assert seen == [torch.float64], (
            f"energy() handed the model {seen}; an fp64 caller silently got fp32"
        )
        assert energy.dtype is torch.float64

        # And forward() still downcasts, deliberately: that is the optimization
        # path, where float32 weights are the point.
        seen.clear()
        adapter.forward(coords.clone(), species, charges)
        assert seen == [torch.float32]

    def test_custom_energy_keeps_float64_for_coords_and_charges(self):
        from Auto3D.models.adapter import CustomModelAdapter

        seen: list = []

        class _Recorder(torch.nn.Module):
            def forward(self, species, coords, charges):
                seen.append((coords.dtype, charges.dtype))
                return (coords**2).sum(dim=(1, 2))

        adapter = self._bypassed(CustomModelAdapter, _Recorder())
        coords = torch.randn(2, 4, 3, dtype=torch.float64)
        species = torch.tensor([[1, 6, 7, 8], [1, 6, 7, -1]])
        charges = torch.zeros(2)

        energy = adapter.energy(coords, species, charges)
        assert seen == [(torch.float64, torch.float64)], (
            f"energy() handed the model {seen}; charges must follow coords so a "
            "model that concatenates them does not hit a dtype mismatch"
        )
        assert energy.dtype is torch.float64

        seen.clear()
        adapter.forward(coords.clone(), species, charges)
        assert seen == [(torch.float32, torch.float32)]

    def test_ani2xt_energy_accepts_a_non_leaf_tensor(self):
        """``ANI2xtAdapter.forward`` calls ``coords.requires_grad_(True)``, which
        raises on the non-leaf tensor an autograd Hessian hands in. Its own
        ``energy`` must not touch ``requires_grad`` at all."""
        from Auto3D.models.adapter import ANI2xtAdapter

        class _Toy(torch.nn.Module):
            def forward(self, species, coords):
                return (coords**2).sum(dim=(1, 2))

        adapter = self._bypassed(ANI2xtAdapter, _Toy())
        leaf = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        non_leaf = leaf * 2.0
        assert non_leaf.grad_fn is not None, "test premise: coords must be non-leaf"
        species = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])

        energy = adapter.energy(non_leaf, species, torch.zeros(2))
        assert energy.dtype is torch.float64
        # Still graph-connected: no internal no_grad, so a Hessian can be taken.
        assert energy.requires_grad
        (grad,) = torch.autograd.grad(energy.sum(), leaf)
        torch.testing.assert_close(grad, 8.0 * leaf)

    def test_energy_has_no_no_grad_anywhere(self):
        """The contract promises a graph-connected result for every adapter."""
        from tests.helpers_adapter import FakeAdapter

        adapter = FakeAdapter()
        coords = torch.randn(1, 3, 3, requires_grad=True)
        energy = adapter.energy(coords, torch.ones(1, 3, dtype=torch.long), torch.zeros(1))
        assert energy.requires_grad
