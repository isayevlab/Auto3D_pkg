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

import inspect
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


class NoForwardOfItsOwn(torch.nn.Module):
    """Valid padding attributes, but no forward of its own.

    Defined at module level so torch.save can pickle it -- a function-local
    class cannot be saved.
    """

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1


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


class TooManyArgsNNP(torch.nn.Module):
    """Extra required positional argument -- Auto3D never passes a fourth."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1

    def forward(self, species, coords, charges, cutoff):
        return (coords ** 2).sum(dim=(1, 2)) * cutoff


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
    with pytest.raises(ModelLoadError, match="species, coords, charges"):
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
    with pytest.raises(ModelLoadError, match="species, coords, charges"):
        check_input(args)


def test_alias_names_in_the_wrong_order_are_rejected(tmp_path):
    """Synonyms must not be an escape hatch from the order check."""
    path = _save(AliasTransposedNNP(), tmp_path, "alias_transposed.pt")
    with pytest.raises(ModelLoadError, match="species, coords, charges"):
        load_custom_nnp(path, CPU)


def _model_with_param_names(species_name, coords_name, charges_name):
    """A plain object (no torch.save/pickle needed) whose forward's parameter
    NAMES are exactly the given ones, so validate_custom_nnp's order check
    (models/contract.py::_classify / _check_forward_signature) sees them.

    Not an nn.Module: this only needs to be introspectable by
    inspect.signature, which is all validate_custom_nnp actually uses, and
    dynamically-named parameters cannot be produced by a module-level
    class (needed elsewhere in this file for torch.save's pickling).
    """
    namespace: dict = {}
    exec(  # noqa: S102 - test-only, fixed trusted template, no user input
        f"def forward(self, {species_name}, {coords_name}, {charges_name}):\n"
        f"    return ({coords_name} ** 2).sum(dim=(1, 2))\n",
        namespace,
    )
    return type(
        "DynamicNamedNNP",
        (),
        {"coord_pad": 0.0, "species_pad": -1, "forward": namespace["forward"]},
    )()


# Full synonym vocabulary from models/contract.py's _SPECIES_NAMES/
# _COORDS_NAMES/_CHARGES_NAMES, covered at least once each, plus one
# mixed-case ("Numbers"/"Positions"/"Charge") combination to confirm the
# order check case-folds via _classify's ``name.lower()``.
ALIAS_VOCABULARY = [
    ("species", "coords", "charges"),
    ("numbers", "positions", "charge"),
    ("atomic_numbers", "coordinates", "charge"),
    ("atomicnumbers", "coord", "q"),
    ("z", "pos", "charges"),
    ("elements", "xyz", "charge"),
    ("Numbers", "Positions", "Charge"),
]


@pytest.mark.parametrize("species_name,coords_name,charges_name", ALIAS_VOCABULARY)
def test_alias_vocabulary_in_correct_order_is_accepted(
    species_name, coords_name, charges_name
):
    """Every recognized synonym, in the right order, must not be rejected --
    a false rejection here would break a working model that merely spelled
    the contract differently."""
    model = _model_with_param_names(species_name, coords_name, charges_name)
    validate_custom_nnp(model, "<memory>")  # must not raise


@pytest.mark.parametrize("species_name,coords_name,charges_name", ALIAS_VOCABULARY)
def test_alias_vocabulary_transposed_is_rejected(
    species_name, coords_name, charges_name
):
    """The same synonyms, transposed (coords first), must still be caught --
    synonyms are not an escape hatch from the order check, across the full
    vocabulary, not just the one numbers/positions/charge pair."""
    model = _model_with_param_names(coords_name, species_name, charges_name)
    with pytest.raises(ModelLoadError, match="species, coords, charges"):
        validate_custom_nnp(model, "<memory>")


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


def test_wrong_arity_too_many_required_args_is_rejected_at_load(tmp_path):
    """The ``> 3`` branch: a fourth REQUIRED positional argument is just as
    uncallable as the two-argument case above, but exercises the other half
    of ``len(positional) < 3 or len(required) > 3`` in
    ``models/contract.py::_check_forward_signature``."""
    path = _save(TooManyArgsNNP(), tmp_path, "fourargs.pt")
    with pytest.raises(ModelLoadError, match="three positional arguments") as excinfo:
        load_custom_nnp(path, CPU)
    assert "cutoff" in str(excinfo.value)


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


def test_the_adapters_pad_is_what_the_padder_writes():
    """One source, not two agreeing sources.

    This used to compare two independent DEFAULTS -- ``BaseModelAdapter``'s
    ``species_pad`` against ``pad_from_mols``'s own -- because each layer had its
    own, and they disagreed (0 vs -1, where 0 collides with ANI2xt's hydrogen
    index). ``pad_from_mols`` now has no pad parameter at all: it reads the value
    off the adapter it is padding for. So the assertion becomes the stronger,
    structural one -- whatever the adapter says is what lands in the tensor -- and
    the old comparison is not merely satisfied but meaningless.
    """
    import inspect

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.batch_opt.padding import pad_from_mols
    from Auto3D.models.adapter import BaseModelAdapter
    from tests.helpers_adapter import FakeAdapter

    assert "species_pad" not in inspect.signature(pad_from_mols).parameters
    assert "coord_pad" not in inspect.signature(pad_from_mols).parameters

    # -1 remains the safe default for a third-party subclass: it can be neither
    # a real atomic number nor a 0-based species index.
    assert inspect.signature(
        BaseModelAdapter.__init__
    ).parameters["species_pad"].default == -1

    mols = []
    for smiles in ("C", "O"):          # 5 atoms and 3, so the batch is padded
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
        mols.append(mol)
    adapter = FakeAdapter(coord_pad=7.25, species_pad=-99)
    coords, species, _charges, _mask = pad_from_mols(mols, adapter, CPU)
    assert species[1, 3:].tolist() == [-99, -99]
    assert torch.all(coords[1, 3:] == 7.25)


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


def test_a_module_inheriting_torchs_forward_stub_is_rejected(tmp_path):
    """A model with valid pads but no forward of its own must fail at LOAD.

    `getattr(model, "forward", None) is None` cannot catch this: torch gives
    every Module a real `forward` attribute -- `_forward_unimplemented(*input)`
    -- so the attribute always exists. And because that stub is VAR_POSITIONAL,
    the "forward(*args) accepts any arity" early-return would then ACCEPT it.

    Without an explicit check, such a model loads clean and raises
    NotImplementedError deep inside the batch optimization loop, after a job
    has already started -- precisely the deferred failure this validator was
    added to eliminate. Verified against the pre-fix code: it loaded silently.
    """
    import torch

    from Auto3D.exceptions import ModelLoadError
    from Auto3D.models.loading import load_custom_nnp

    # Preconditions: this is exactly the shape the two earlier checks miss.
    m = NoForwardOfItsOwn()
    assert getattr(m, "forward", None) is not None
    assert any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        for p in inspect.signature(m.forward).parameters.values()
    )

    path = tmp_path / "no_forward.pt"
    torch.save(m, str(path))

    with pytest.raises(ModelLoadError, match="no forward method of its own"):
        load_custom_nnp(str(path), torch.device("cpu"))


class TestExampleCustomNNPsDoNotPadWithARealElement:
    """The copyable ``userNNP2`` examples must not pad species with 0.

    Atomic number 0 is a real element in a batch Auto3D builds -- an R-group
    ``*`` atom -- and each example identifies padding with
    ``mask = species != self.species_pad``. With ``species_pad = 0`` that
    expression deletes every dummy atom before the energy call, so the model
    scores a molecule the user never submitted (audit C13). ``pad_from_mols``
    defaults to -1 and ``docs/source/howto/custom_nnp.rst`` already says -1;
    these three examples said 0, and they are the code users copy.

    The examples are driven through their OWN ``forward`` with the AIMNet2
    calculator replaced by a recorder, so no NNP is loaded and nothing is
    downloaded.
    """

    EXAMPLE_MODULES = ("tests.test_SPE", "tests.test_thermo", "tests.test_auto3D")

    @staticmethod
    def _padded_batch(model):
        """The batch Auto3D itself builds, with the example's own pad values."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.batch_opt.padding import pad_from_mols

        mols = []
        # Different sizes, so the batch really is padded; the first molecule
        # carries an R-group atom (Z=0).
        for smiles in ("*CCO", "C"):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
            mols.append(mol)
        # The example model IS the adapter here: it declares its own
        # coord_pad/species_pad and consumes raw atomic numbers, so the identity
        # `to_species` is what a custom NNP must get.
        model.to_species = list
        coords, species, charges, _atom_mask = pad_from_mols(mols, model, CPU)
        return mols, coords, species, charges

    @pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
    def test_every_submitted_atom_reaches_the_model(self, module_name):
        import importlib

        module = importlib.import_module(module_name)
        model = module.userNNP2()

        mols, coords, species, charges = self._padded_batch(model)

        received: dict = {}

        class _RecordingCalculator:
            """Stands in for AIMNet2Calculator; records what it is handed."""

            def __call__(self, inputs, forces=False):
                received.update(inputs)
                n_mols = int(inputs["mol_idx"].max().item()) + 1
                return {"energy": torch.zeros(n_mols)}

        # Pre-seed the lazily built backend so forward() never imports aimnet.
        model._calc = _RecordingCalculator()
        model._calc_device = species.device

        model(species, coords, charges)

        assert received, f"{module_name}.userNNP2 never called its calculator"
        counts = torch.bincount(
            received["mol_idx"], minlength=len(mols)
        ).tolist()
        expected = [mol.GetNumAtoms() for mol in mols]
        assert counts == expected, (
            f"{module_name}.userNNP2 passed {counts} atoms per molecule to the "
            f"calculator but was given {expected}: with species_pad="
            f"{model.species_pad!r}, its own 'species != species_pad' mask "
            "deletes real atoms"
        )
        assert 0 in received["numbers"].tolist(), (
            f"{module_name}.userNNP2 dropped the R-group (Z=0) atom, so it "
            "scored a different molecule than the one submitted"
        )


# --- the contract is derived from the Protocol, not retyped next to it ------

def test_required_attributes_tracks_the_protocol():
    """What the validator demands must be DERIVED from ``CustomNNP`` itself.

    ``REQUIRED_ATTRIBUTES`` used to be a hand-written tuple sitting a few lines
    above the Protocol that declares the same two members, with nothing linking
    them. This test enumerates the Protocol *here* rather than hardcoding names,
    so adding a data member to ``CustomNNP`` without the validator learning
    about it in the same edit goes red.
    """
    from Auto3D.models.contract import REQUIRED_ATTRIBUTES, CustomNNP

    declared = tuple(CustomNNP.__annotations__)
    assert REQUIRED_ATTRIBUTES == declared, (
        f"validator demands {REQUIRED_ATTRIBUTES} but CustomNNP declares "
        f"{declared}; the two must not be maintained separately"
    )

    class NoPadsAtAll:
        def forward(self, species, coords, charges):
            return coords

    with pytest.raises(ModelLoadError) as excinfo:
        validate_custom_nnp(NoPadsAtAll(), "<memory>")
    message = str(excinfo.value)
    for name in declared:
        assert name in message, (
            f"{name} is declared on CustomNNP but the rejection message does "
            f"not name it: {message}"
        )


def test_customnnp_data_members_are_exactly_the_two_padding_values():
    """A deliberate change-detector, not a restatement of the line above.

    ``validate_custom_nnp`` skips the ``forward`` signature check entirely for a
    TorchScript ``RecursiveScriptModule`` (its forward is a pybind11 builtin
    with no Python signature), so for every archive in the wild
    ``REQUIRED_ATTRIBUTES`` is the ONLY gate. Now that the tuple is derived from
    ``CustomNNP.__annotations__``, adding an annotated field to the Protocol --
    even "just for documentation" -- immediately rejects every existing archive
    that does not carry it. That is a breaking change and must be released as
    one; this test is what makes it impossible to do by accident.
    """
    from Auto3D.models.contract import CustomNNP

    assert tuple(CustomNNP.__annotations__) == ("coord_pad", "species_pad")


def test_customnnp_is_not_runtime_checkable():
    """``isinstance(x, CustomNNP)`` must raise, not answer.

    ``@runtime_checkable`` tests attribute *presence* only. Every
    ``torch.nn.Module`` has a ``forward`` attribute (torch installs
    ``Module.forward = _forward_unimplemented``), so the single most common real
    failure -- a saved module that never defined its own ``forward`` -- would
    pass an ``isinstance`` check while raising ``NotImplementedError`` deep in
    the optimization loop. A boolean also cannot carry the diagnosis
    ``validate_custom_nnp`` produces. So the honest answer to "can I check this
    at runtime?" is a ``TypeError`` pointing at the validator.
    """
    from Auto3D.models.contract import CustomNNP

    with pytest.raises(TypeError):
        isinstance(object(), CustomNNP)  # noqa: B015 - the call IS the assertion


def test_keyword_only_forward_message_is_not_its_own_demand():
    """The rejection must not render the signature it is asking for.

    The message used to comma-join parameter *names*, dropping ``*``, ``/`` and
    defaults, so a keyword-only ``forward(self, *, species, coords, charges)``
    was rejected with "has forward(species, coords, charges) ... Expected
    forward(self, species, coords, charges)" -- text that shows the author
    nothing wrong. Rendering the signature the way the interpreter does keeps
    the marker that actually explains the refusal.
    """
    from Auto3D.models.contract import EXPECTED_SIGNATURE

    class KeywordOnly:
        coord_pad = 0.0
        species_pad = -1

        def forward(self, *, species, coords, charges):
            return coords

    with pytest.raises(ModelLoadError) as excinfo:
        validate_custom_nnp(KeywordOnly(), "<memory>")
    message = str(excinfo.value)
    observed = message.split("Expected")[0]
    assert "*" in observed, (
        "the keyword-only marker is what explains the rejection, but the "
        f"message renders no '*': {message}"
    )
    assert EXPECTED_SIGNATURE in message


def test_positional_only_marker_survives_the_transposed_message():
    """The order-check message renders ``/`` too, for the same reason."""

    class Transposed:
        coord_pad = 0.0
        species_pad = -1

        def forward(self, coords, species, /, charges):
            return coords

    with pytest.raises(ModelLoadError) as excinfo:
        validate_custom_nnp(Transposed(), "<memory>")
    assert "/" in str(excinfo.value).split("but Auto3D calls")[0]


# --- inference mode ---------------------------------------------------------

def test_torchscript_archive_saved_in_train_mode_loads_in_eval_mode(tmp_path):
    """``torch.jit.save`` records ``training``; ``load_custom_nnp`` must clear it.

    The eager branch has always called ``.eval()``. The TorchScript branch
    returned ``torch.jit.load``'s result untouched, so an archive a user saved
    without calling ``.eval()`` first kept dropout and batchnorm live at
    inference.
    """
    from tests.helpers_custom_nnp import StochasticNNP

    model = StochasticNNP()
    model.train()
    path = tmp_path / "train_mode.pt"
    torch.jit.save(torch.jit.script(model), str(path))

    loaded = load_custom_nnp(str(path), CPU)

    assert loaded.training is False, (
        "a TorchScript archive saved in train mode stayed in train mode, so "
        "dropout/batchnorm run at inference"
    )


def test_torchscript_archive_saved_in_train_mode_gives_repeatable_energies(tmp_path):
    """Why the flag matters: FIRE cannot converge against a stochastic energy.

    Identical inputs must give one energy. Without the eval-mode guarantee this
    returns a different number nearly every call, and the optimizer burns its
    whole step budget before dropping the conformer as oscillating.
    """
    from tests.helpers_custom_nnp import StochasticNNP

    model = StochasticNNP()
    model.train()
    path = tmp_path / "train_mode_energies.pt"
    torch.jit.save(torch.jit.script(model), str(path))

    loaded = load_custom_nnp(str(path), CPU)

    species = torch.zeros(1, 8, dtype=torch.long)
    coords = torch.randn(1, 8, 3)
    charges = torch.zeros(1)
    energies = {round(float(loaded(species, coords, charges)), 8) for _ in range(5)}

    assert len(energies) == 1, f"energy is not repeatable across calls: {energies}"
