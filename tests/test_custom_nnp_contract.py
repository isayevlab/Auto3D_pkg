"""One custom-NNP contract, enforced where the model is loaded (audit C12).

Auto3D calls a user-supplied NNP as ``model(species, coords, charges) ->
energies`` and differentiates the result to get forces. That is the opposite
argument order from Auto3D's INTERNAL ``ModelAdapter`` interface
(``forward(coords, species, charges) -> (energies, forces)``), and a model
written against the wrong one produces an energy from transposed tensors that
only explodes later, inside ``torch.autograd.grad``. These tests pin the
rejection to load time.

Hermetic: every model here is a few-line ``torch.nn.Module`` written to
``tmp_path``. No real NNP is loaded and nothing is downloaded.
"""
from __future__ import annotations

import pytest
import torch

from Auto3D.exceptions import ModelLoadError
from Auto3D.models.contract import validate_custom_nnp
from Auto3D.models.loading import load_custom_nnp

CPU = torch.device("cpu")


# --- module-level so torch.save can pickle them by reference ----------------

class GoodNNP(torch.nn.Module):
    """The real contract: species first, energies only."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, species, coords, charges):
        mask = (species != self.species_pad).unsqueeze(-1)
        return (coords * mask).pow(2).sum(dim=(1, 2))


class TransposedNNP(torch.nn.Module):
    """Written against the internal adapter interface: coords first."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, coords, species, charges):
        return torch.zeros(coords.shape[0])


class NoPadsNNP(torch.nn.Module):
    """Correct forward, but no padding attributes."""

    def forward(self, species, coords, charges):
        return (coords ** 2).sum(dim=(1, 2))


class OnlyCoordPadNNP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0

    def forward(self, species, coords, charges):
        return (coords ** 2).sum(dim=(1, 2))


class TooFewArgsNNP(torch.nn.Module):
    """Forgot charges -- Auto3D always passes three positional arguments."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, species, coords):
        return (coords ** 2).sum(dim=(1, 2))


class ExoticNamesNNP(torch.nn.Module):
    """Right arity, names outside the known vocabulary -- must be accepted."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, a, b, c):
        return (b ** 2).sum(dim=(1, 2))


class VarArgsNNP(torch.nn.Module):
    """forward(*args) names nothing; arity is unknowable -- must be accepted."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, *args):
        return (args[1] ** 2).sum(dim=(1, 2))


class AliasNamesNNP(torch.nn.Module):
    """Synonyms in the RIGHT order (numbers/positions/charge) -- accepted."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, numbers, positions, charge):
        return (positions ** 2).sum(dim=(1, 2))


class AliasTransposedNNP(torch.nn.Module):
    """Synonyms in the WRONG order -- must still be caught."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, positions, numbers, charge):
        return (positions ** 2).sum(dim=(1, 2))


class OrderRecordingNNP(torch.nn.Module):
    """Records the rank of each tensor it is handed, to observe argument order."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1
        self.seen_ranks: list[int] = []

    def forward(self, species, coords, charges):
        self.seen_ranks = [species.dim(), coords.dim(), charges.dim()]
        return (coords ** 2).sum(dim=(1, 2))


def _save(model, tmp_path, name):
    path = tmp_path / name
    torch.save(model, str(path))
    return str(path)


# --- rejection --------------------------------------------------------------

def test_transposed_forward_is_rejected_at_load(tmp_path):
    """A model taking (coords, species, charges) must fail at load, naming the
    order Auto3D actually calls -- not deep inside torch.autograd.grad."""
    path = _save(TransposedNNP(), tmp_path, "transposed.pt")
    with pytest.raises(ModelLoadError) as excinfo:
        load_custom_nnp(path, CPU)
    message = str(excinfo.value)
    assert "species" in message and "coords" in message
    assert "species, coords, charges" in message, message


def test_transposed_forward_is_rejected_through_the_adapter(tmp_path):
    """The adapter is the production entry point; the refusal must reach it."""
    from Auto3D.models.adapter import CustomModelAdapter

    path = _save(TransposedNNP(), tmp_path, "transposed_adapter.pt")
    with pytest.raises(ModelLoadError):
        CustomModelAdapter(path, CPU)


def test_transposed_forward_is_rejected_by_input_validation(tmp_path):
    """utils.validation preflights a custom engine path; it must reject there
    too, before any conformer work starts."""
    from types import SimpleNamespace

    from Auto3D.utils.validation import check_input

    path = _save(TransposedNNP(), tmp_path, "transposed_preflight.pt")
    smi = tmp_path / "in.smi"
    smi.write_text("CCO test\n")
    args = SimpleNamespace(
        path=str(smi),
        input_format="smi",
        optimizing_engine=path,
        isomer_engine="rdkit",
        enumerate_tautomer=False,
        opt_steps=100,
        use_gpu=False,
        gpu_idx=0,
        capacity=42,
        k=1,
        window=False,
        max_confs=None,
        memory=None,
        verbose=False,
        job_name="",
    )
    with pytest.raises(ModelLoadError):
        check_input(args)


def test_alias_names_in_the_wrong_order_are_rejected(tmp_path):
    """Synonyms must not be an escape hatch from the order check."""
    path = _save(AliasTransposedNNP(), tmp_path, "alias_transposed.pt")
    with pytest.raises(ModelLoadError, match="species, coords, charges"):
        load_custom_nnp(path, CPU)


def test_missing_both_padding_attributes_are_rejected_at_load(tmp_path):
    """coord_pad/species_pad are part of the contract; absent, the layers used
    to disagree on the default, so a silent fallback is worse than a refusal."""
    path = _save(NoPadsNNP(), tmp_path, "nopads.pt")
    with pytest.raises(ModelLoadError) as excinfo:
        load_custom_nnp(path, CPU)
    message = str(excinfo.value)
    assert "coord_pad" in message and "species_pad" in message


def test_missing_one_padding_attribute_is_rejected_and_named(tmp_path):
    """The message must name the attribute that is actually missing."""
    path = _save(OnlyCoordPadNNP(), tmp_path, "onlyspecies.pt")
    with pytest.raises(ModelLoadError) as excinfo:
        load_custom_nnp(path, CPU)
    message = str(excinfo.value)
    assert "species_pad" in message
    assert "coord_pad," not in message  # only the missing one is listed


def test_wrong_arity_is_rejected_at_load(tmp_path):
    """Auto3D always passes three positional arguments."""
    path = _save(TooFewArgsNNP(), tmp_path, "twoargs.pt")
    with pytest.raises(ModelLoadError, match="three positional arguments"):
        load_custom_nnp(path, CPU)


# --- acceptance (a false rejection is a regression) -------------------------

def test_contract_conforming_model_loads_and_runs(tmp_path):
    """The happy path must still work, end to end through the adapter, with the
    force sign and value that autograd implies."""
    from Auto3D.models.adapter import CustomModelAdapter

    path = _save(GoodNNP(), tmp_path, "good.pt")
    adapter = CustomModelAdapter(path, CPU)
    assert adapter.coord_pad == 0.0
    assert adapter.species_pad == -1

    coords = torch.randn(2, 3, 3)
    species = torch.tensor([[1, 6, 8], [1, 6, -1]])
    charges = torch.zeros(2)
    # The ADAPTER takes coords first -- the reverse of the custom model.
    energy, forces = adapter.forward(coords, species, charges)
    mask = (species != -1).unsqueeze(-1)
    torch.testing.assert_close(energy, (coords * mask).pow(2).sum(dim=(1, 2)))
    torch.testing.assert_close(forces, -2.0 * coords * mask)


def test_unrecognized_parameter_names_are_accepted(tmp_path):
    """Order is unknowable from names like (a, b, c); guessing would falsely
    reject a working model, so the order check must stand down."""
    path = _save(ExoticNamesNNP(), tmp_path, "exotic.pt")
    assert isinstance(load_custom_nnp(path, CPU), torch.nn.Module)


def test_var_args_forward_is_accepted(tmp_path):
    """forward(*args) constrains neither arity nor order; accept it."""
    path = _save(VarArgsNNP(), tmp_path, "varargs.pt")
    assert isinstance(load_custom_nnp(path, CPU), torch.nn.Module)


def test_alias_names_in_the_right_order_are_accepted(tmp_path):
    """(numbers, positions, charge) is the contract spelled differently."""
    path = _save(AliasNamesNNP(), tmp_path, "alias_ok.pt")
    assert isinstance(load_custom_nnp(path, CPU), torch.nn.Module)


def test_torchscript_archive_is_accepted_despite_opaque_signature(tmp_path):
    """A loaded RecursiveScriptModule's forward is a pybind11 builtin with no
    Python signature, so inspect.signature raises. Rejecting it would break
    every TorchScript custom NNP; the attribute check must still apply."""
    import inspect

    from tests.helpers_custom_nnp import ScriptableNNP

    path = tmp_path / "scripted.pt"
    torch.jit.save(torch.jit.script(ScriptableNNP()), str(path))

    loaded = torch.jit.load(str(path), map_location=CPU)
    # Pin the premise: if a future torch exposes a signature here, this test is
    # no longer exercising the fallback and should be revisited.
    with pytest.raises((ValueError, TypeError)):
        inspect.signature(loaded.forward)

    model = load_custom_nnp(str(path), CPU)
    assert model.coord_pad == 0.0
    assert model.species_pad == -1


def test_torchscript_archive_without_instance_attributes_is_rejected(tmp_path):
    """TorchScript drops plain CLASS attributes, so such an archive genuinely
    arrives without coord_pad/species_pad and must be refused with advice."""
    from tests.helpers_custom_nnp import ClassAttrOnlyNNP

    path = tmp_path / "classattr.pt"
    torch.jit.save(torch.jit.script(ClassAttrOnlyNNP()), str(path))
    with pytest.raises(ModelLoadError, match="__constants__"):
        load_custom_nnp(str(path), CPU)


# --- the deleted public Protocol -------------------------------------------

def test_config_nnpmodel_protocol_is_gone():
    """Auto3D.NNPModel duplicated the contract in a module that never enforced
    it. One definition now, in Auto3D.models.contract."""
    import Auto3D
    import Auto3D.config

    assert "NNPModel" not in Auto3D.__all__
    assert not hasattr(Auto3D.config, "NNPModel")
    with pytest.raises(AttributeError):
        getattr(Auto3D, "NNPModel")  # noqa: B009 - the lookup IS the assertion


def test_custom_nnp_protocol_matches_what_the_adapter_calls(tmp_path):
    """The surviving Protocol must describe the call the adapter actually makes.

    Observed, not asserted from source: the model records the rank of each
    tensor it receives, so species (rank 2) arriving where coords (rank 3) is
    declared would show up here. This is the guarantee the deleted
    ``config.NNPModel`` named but never checked.
    """
    import inspect

    from Auto3D.models.adapter import CustomModelAdapter
    from Auto3D.models.contract import CustomNNP

    protocol_params = [
        p for p in inspect.signature(CustomNNP.forward).parameters if p != "self"
    ]
    assert protocol_params == ["species", "coords", "charges"]

    path = _save(OrderRecordingNNP(), tmp_path, "recorder.pt")
    adapter = CustomModelAdapter(path, CPU)
    adapter.forward(torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.long),
                    torch.zeros(2))
    # species -> rank 2, coords -> rank 3, charges -> rank 1, in that order.
    assert adapter.model.seen_ranks == [2, 3, 1], adapter.model.seen_ranks


def test_adapter_keeps_no_padding_fallback_of_its_own(monkeypatch, tmp_path):
    """The adapter must read the pads off the model, not substitute defaults.

    CustomModelAdapter used to call getattr(model, 'species_pad', -1) while
    BaseModelAdapter's own default was 0. With the loader validating the
    contract those fallbacks are dead code, but dead code that disagrees is
    exactly what re-grows: if the loader's check were ever loosened, a getattr
    fallback would resume silently inventing a padding value. This pins the
    adapter to fail loudly instead, by handing it a model the loader did not
    vet.
    """
    import Auto3D.models.adapter as adapter_mod
    from Auto3D.models.adapter import CustomModelAdapter

    class Unvetted(torch.nn.Module):
        def forward(self, species, coords, charges):
            return (coords ** 2).sum(dim=(1, 2))

    monkeypatch.setattr(
        adapter_mod, "load_custom_nnp", lambda path, device, **kw: Unvetted()
    )
    with pytest.raises(AttributeError, match="coord_pad"):
        CustomModelAdapter(str(tmp_path / "unused.pt"), CPU)


def test_base_adapter_species_pad_default_agrees_with_the_padding_layer():
    """One default, not two.

    BaseModelAdapter used to default species_pad to 0 while
    batch_opt.padding.pad_from_mols defaults it to -1, so a subclass that did
    not pass the value got a different notion of padding depending on which
    layer supplied it -- and 0 collides with ANI2xt's hydrogen index. -1 wins:
    it can be neither an atomic number nor a 0-based species index.
    """
    import inspect

    from Auto3D.batch_opt.padding import pad_from_mols
    from Auto3D.models.adapter import BaseModelAdapter

    adapter_default = inspect.signature(
        BaseModelAdapter.__init__
    ).parameters["species_pad"].default
    padding_default = inspect.signature(pad_from_mols).parameters["species_pad"].default

    assert adapter_default == padding_default == -1


def test_validate_custom_nnp_is_callable_directly():
    """The validator is the single owner of the contract; a duck-typed object
    that satisfies it must pass, and the same object minus a pad must not."""

    class Ok:
        coord_pad = 0.0
        species_pad = -1

        def forward(self, species, coords, charges):
            return coords

    validate_custom_nnp(Ok(), "<memory>")

    class Bad:
        coord_pad = 0.0

        def forward(self, species, coords, charges):
            return coords

    with pytest.raises(ModelLoadError, match="species_pad"):
        validate_custom_nnp(Bad(), "<memory>")
