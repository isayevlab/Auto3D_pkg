"""The one setup prologue the three single-file entry points share.

``calc_spe``, ``opt_geometry`` and ``calc_thermo`` each used to open with its
own hand-rolled copy of the same nine steps, in the same order, each copy
carrying its own rationale comments -- three statements of one policy that had
already drifted on the one step where the order matters (``get_device`` ran
before the record read in ``opt_geometry`` and after it in the other two). This
module pins what replaced them: the step list, the step ORDER, and the two
"nothing usable" verdicts the three entry points pick between.

The order is asserted, not just the membership: every step after the first is
there because something before it must have failed first (an unrecognized
engine name must not cost a model download; an output collision must be
refused before a file is read), so a reordering is a behavior change even when
every step still runs.
"""

from __future__ import annotations

import pytest

from Auto3D.foundation.utils.sdf_io import classify_records
from tests.helpers_records import write_mixed_sdf

#: The documented order. `prepare_single_file_run` runs exactly these, exactly
#: once each, in exactly this sequence.
NINE = [
    "resolve_engine_name",
    "check_gpu_requested",
    "check_output_not_input",
    "configure_torch",
    "default_output_path",
    "check_output_overwrite",
    "classify_records",
    "check_engine_supports_molecules",
    "get_device",
]

#: What the recorders below return for the three steps whose return value the
#: prologue actually consumes. Sentinels rather than plausible values, so a
#: result the prologue derived some other way cannot pass for this one.
_RETURNS = {"default_output_path": "derived.sdf", "get_device": "cpu-sentinel"}


def _classified(tmp_path):
    """A real :class:`ClassifiedRecords`, from the shared mixed-record fixture.

    Built by the real classifier rather than by hand: ``ClassifiedRecords``
    enforces ``parsed == kept + skipped``, and a hand-built one that happens to
    satisfy it today would still be a second, independent answer to "what is a
    record" sitting in a test.
    """
    return classify_records(str(write_mixed_sdf(tmp_path / "mixed.sdf")))


def _record_calls(monkeypatch, module, calls, *, records):
    """Replace every collaborator on ``module`` with a recorder of its name.

    ``classify_records`` returns ``records``; the other two consumed results
    are the sentinels in :data:`_RETURNS`. ``TorchConfig`` is deliberately NOT
    replaced -- it is a value, and constructing the real one is part of what
    ``configure_torch`` is handed.
    """
    for name in NINE:

        def fake(*args, _name=name, **kwargs):
            calls.append(_name)
            return records if _name == "classify_records" else _RETURNS.get(_name)

        monkeypatch.setattr(module, name, fake)


def test_the_nine_steps_run_in_the_documented_order(monkeypatch, tmp_path):
    import Auto3D.entry._run_setup as rs

    calls: list[str] = []
    classified = _classified(tmp_path)
    _record_calls(monkeypatch, rs, calls, records=classified)

    setup = rs.prepare_single_file_run(
        str(tmp_path / "in.sdf"),
        "AIMNET",
        gpu_idx=0,
        use_gpu=False,
        allow_tf32=False,
        out_path=None,
        overwrite=True,
        tag="E",
    )

    assert calls == NINE
    assert setup.out_path == "derived.sdf"
    assert setup.device == "cpu-sentinel"
    # The partition is forwarded, not re-derived: identity on the record lists
    # and the unparseable COUNT (the message needs a number, not positions).
    assert setup.records == classified.kept
    assert setup.skipped == classified.skipped
    assert setup.unparseable == len(classified.unparseable)


def test_an_explicit_out_path_skips_the_derived_name(monkeypatch, tmp_path):
    """``-o`` is used verbatim, and the naming convention is not consulted.

    Not a cosmetic difference: ``default_output_path`` stats ``model_name`` to
    decide the ``userNNP`` tag, so deriving a name nobody asked for is work
    done against the filesystem for a result that is thrown away.
    """
    import Auto3D.entry._run_setup as rs

    calls: list[str] = []
    _record_calls(monkeypatch, rs, calls, records=_classified(tmp_path))

    setup = rs.prepare_single_file_run(
        str(tmp_path / "in.sdf"),
        "AIMNET",
        gpu_idx=0,
        use_gpu=False,
        allow_tf32=False,
        out_path="x.sdf",
        overwrite=True,
        tag="E",
    )

    assert calls == [name for name in NINE if name != "default_output_path"]
    assert setup.out_path == "x.sdf"


def test_require_records_names_the_counts_and_the_path():
    """The error a user reads when nothing in the file can be computed.

    Both halves are load-bearing: the path (a run can be handed several files)
    and the per-reason counts, which say WHICH defect dominates -- an SDF whose
    every record is a heavy-atom skeleton is a different mistake from one
    RDKit could not parse, and the fix differs.
    """
    from rdkit import Chem

    from Auto3D.entry._run_setup import RunSetup
    from Auto3D.foundation.exceptions import InputValidationError

    def _mol():
        return Chem.MolFromSmiles("CCO")

    setup = RunSetup(
        device=None,
        out_path="o.sdf",
        records=[],
        skipped=[(_mol(), "implicit_hydrogens"), (_mol(), "dummy_atoms")],
        unparseable=1,
    )

    with pytest.raises(
        InputValidationError,
        match=r"in\.sdf.*1 unparseable.*1 implicit_hydrogens.*1 dummy_atoms",
    ):
        setup.require_records("in.sdf")

    # Same object, the other verdict: `calc_thermo` marks these two records
    # `Thermo_failed` and writes them, so for it this file is not empty.
    setup.require_any("in.sdf")


def test_require_any_raises_only_for_a_file_with_nothing_parseable(tmp_path):
    """``calc_thermo``'s verdict, through the real prologue on real files.

    A defective record is something to mark, not a reason to refuse the file
    (D2), so only a file that yielded no ``Mol`` at all leaves it with nothing
    to write.
    """
    from Auto3D.entry._run_setup import prepare_single_file_run
    from Auto3D.foundation.exceptions import InputValidationError

    def _prepare(path):
        return prepare_single_file_run(
            str(path),
            "AIMNET",
            gpu_idx=0,
            use_gpu=False,
            allow_tf32=False,
            out_path=str(tmp_path / "out.sdf"),
            overwrite=True,
            tag="G",
        )

    unparseable = tmp_path / "bad.sdf"
    unparseable.write_text("not a real record\n$$$$\n")
    with pytest.raises(InputValidationError, match=r"bad\.sdf"):
        _prepare(unparseable).require_any(str(unparseable))

    # The mixed fixture keeps two good records and two defective ones, so
    # neither verdict fires on it.
    mixed = _prepare(write_mixed_sdf(tmp_path / "mixed.sdf"))
    mixed.require_any("mixed.sdf")
    mixed.require_records("mixed.sdf")


def test_skip_messages_reach_log_skipped(tmp_path, caplog):
    """A caller's own wording is used, and only for the reasons it words.

    ``calc_thermo`` hands over a PARTIAL table -- it marks three defects
    ``Thermo_failed`` and words those three, and leaves ``"unparseable"`` to
    the shared sentence because that record is dropped there like everywhere
    else. So the table must merge over the default rather than replace it.
    """
    import logging

    from Auto3D.entry._run_setup import prepare_single_file_run

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        prepare_single_file_run(
            str(path),
            "AIMNET",
            gpu_idx=0,
            use_gpu=False,
            allow_tf32=False,
            out_path=str(tmp_path / "out.sdf"),
            overwrite=True,
            tag="G",
            skip_messages={"implicit_hydrogens": "%s: the caller's own wording."},
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "skeleton: the caller's own wording." in messages, messages
    # Not overridden, so still reported -- in sdf_io's wording, not missing and
    # not a KeyError.
    assert any("could not parse it" in message for message in messages), messages


def _spy_on_the_prologue(monkeypatch, module, seen):
    """Record what ``module`` asks the prologue for, then run the real one."""
    import Auto3D.entry._run_setup as rs

    real = rs.prepare_single_file_run

    def spy(path, model_name, **kwargs):
        seen.append((path, model_name, kwargs))
        return real(path, model_name, **kwargs)

    monkeypatch.setattr(module, "prepare_single_file_run", spy)


def _one_good_record(tmp_path, name="ethanol"):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0, "test premise: must embed"
    mol.SetProp("_Name", name)
    path = tmp_path / "in.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return path


def _prepare_spe(monkeypatch, tmp_path):
    """``calc_spe`` with its model machinery stubbed, as test_SPE.py stubs it."""
    import torch

    import Auto3D.entry.SPE as spe_mod
    from tests.helpers_adapter import FakeAdapter

    class _FakeEnForce:
        def __init__(self, adapter):
            pass

        def energy_batched(self, coords, numbers, charges, atom_mask=None):
            return torch.zeros(coords.shape[0], dtype=torch.float64)

    def _fake_pad(mols, adapter, device):
        n = len(mols)
        return (
            torch.zeros(n, 1, 3),
            torch.zeros(n, 1, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            torch.ones(n, 1, dtype=torch.bool),
        )

    monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter(species_pad=0))
    monkeypatch.setattr(spe_mod, "EnForce_ANI", _FakeEnForce)
    monkeypatch.setattr(spe_mod, "pad_from_mols", _fake_pad)
    return spe_mod, spe_mod.calc_spe, _one_good_record(tmp_path)


def _prepare_geometry(monkeypatch, tmp_path):
    """``opt_geometry`` with ``optimizing`` stubbed, as test_ase_geometry.py stubs it."""
    from rdkit import Chem

    import Auto3D.entry.ASE.geometry as geo
    from tests.helpers_adapter import FakeAdapter

    path = _one_good_record(tmp_path)

    class _FakeOptimizing:
        def __init__(self, in_path, out_path, *, adapter, device, config):
            self._in_path = in_path
            self._out_path = out_path

        def run(self):
            mols = [m for m in Chem.SDMolSupplier(str(self._in_path), removeHs=False) if m]
            with Chem.SDWriter(str(self._out_path)) as writer:
                for mol in mols:
                    mol.SetProp("E_tot", "-1.0")  # Hartree, as optimizing.run() writes it
                    writer.write(mol)
            return True

    monkeypatch.setattr(geo, "create_model", lambda *a, **k: FakeAdapter())
    monkeypatch.setattr(geo, "optimizing", _FakeOptimizing)
    return geo, geo.opt_geometry, path


def _prepare_thermo(monkeypatch, tmp_path):
    """``calc_thermo`` on an all-defective file, as test_thermo_record_assertions.py does.

    One implicit-H record: the prologue marks it and the per-record loop
    iterates nothing, so the stubbed ``_load_hessian_model`` is never asked for
    a Hessian it cannot give.
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch import nn

    import Auto3D.entry.ASE.thermo.driver as thermo_mod
    from Auto3D.entry.ASE.thermo import calculator as _calculator
    from tests.helpers_adapter import AdapterModuleMixin

    class _StubNNP(AdapterModuleMixin, nn.Module):
        def forward(self, coords, species, charges, atom_mask=None):
            return torch.zeros(coords.shape[0], dtype=coords.dtype), torch.zeros_like(coords)

    stub = _StubNNP()
    monkeypatch.setattr(thermo_mod, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(_calculator, "create_model", lambda *a, **k: stub)
    monkeypatch.setattr(thermo_mod, "_load_hessian_model", lambda *a, **k: object())

    mol = Chem.MolFromSmiles("CCO")  # implicit hydrogens: defective on purpose
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0, "test premise: must embed"
    mol.SetProp("_Name", "skeleton")
    path = tmp_path / "in.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return thermo_mod, thermo_mod.calc_thermo, path


@pytest.mark.parametrize(
    ("prepare", "tag", "thermo"),
    [
        (_prepare_spe, "E", False),
        (_prepare_geometry, "opt", False),
        (_prepare_thermo, "G", True),
    ],
    ids=["calc_spe", "opt_geometry", "calc_thermo"],
)
def test_each_entry_point_calls_the_prologue_with_its_tag(
    monkeypatch, tmp_path, prepare, tag, thermo
):
    """Every entry point reaches the shared prologue, with its own output tag.

    The tag is the one thing the three do not share, so it is the one thing
    each has to pass -- and an entry point that quietly kept a private copy of
    any step would show up here as a prologue that was never called.
    """
    module, entry, path = prepare(monkeypatch, tmp_path)
    out = tmp_path / "out.sdf"
    seen: list[tuple] = []
    _spy_on_the_prologue(monkeypatch, module, seen)

    entry(
        str(path),
        "AIMNET",
        gpu_idx=1,
        use_gpu=False,
        allow_tf32=True,
        out_path=str(out),
        overwrite=True,
    )

    assert len(seen) == 1, "the prologue must run exactly once per call"
    seen_path, seen_model, kwargs = seen[0]
    assert (seen_path, seen_model) == (str(path), "AIMNET")
    assert {key: value for key, value in kwargs.items() if key != "skip_messages"} == {
        "gpu_idx": 1,
        "use_gpu": False,
        "allow_tf32": True,
        "out_path": str(out),
        "overwrite": True,
        "tag": tag,
    }
    if thermo:
        # calc_thermo is the one caller with its own wording, because it is the
        # one that keeps and marks the records the others drop.
        assert kwargs["skip_messages"] is module._THERMO_SKIP_MESSAGES
    else:
        assert kwargs.get("skip_messages") is None
