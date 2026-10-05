"""Every reader of an SDF agrees on what a record is, on one fixture.

The fixture has one unparseable, one implicit-hydrogen, one dummy-atom, and two good
records. The single-file entry points skip the defective ones (calc_thermo keeps and
marks them); main()'s SDF engine adds hydrogens and re-embeds, so for it only the
dummy-atom record is defective; `check_sdf_format` reports only the unparseable and
dummy-atom records, passing the implicit-hydrogen one over in silence for the same
reason. Both divergences are deliberate and pinned here.

`calc_spe` and `opt_geometry` are covered through the one prologue they now share
(`prepare_single_file_run`), which is where their read of the file happens; `calc_thermo`
has its own row because its verdict on a defective record differs. The classifier, the
tautomer selector, the SDF isomer engine, and `check_sdf_format` are the other readers
here.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from Auto3D.foundation.utils.sdf_io import classify_records
from tests.helpers_records import write_mixed_sdf


def test_classify_records_is_the_baseline(tmp_path):
    """What every other row in this module is compared against."""
    classified = classify_records(str(write_mixed_sdf(tmp_path / "mixed.sdf")))
    assert classified.names() == ["ethanol", "ethane"]


def test_select_tautomers_keeps_exactly_the_kept_records(tmp_path):
    """Tautomer selection reads through `iter_conformer_records`, so it sees
    exactly the classifier's `kept` records.

    Each fixture name carries no ``@tautN`` suffix, so every record is its own
    tautomer group and a ``k=1`` selection keeps all of them -- which makes the
    output names a direct readout of what the reader accepted.
    """
    from Auto3D.entry.tautomer import select_tautomers

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    out = select_tautomers(str(path), k=1)

    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None]
    kept = classify_records(str(path)).names()
    # Equality, not inclusion: ethanol and ethane have different formulas, so
    # they can never share a tautomer partition, and `k=1` keeps one winner
    # from each. A subset assertion would also pass on an empty output or one
    # that silently lost a good record -- the half most likely to break.
    assert set(names) == set(kept), f"the reader and the classifier disagree: {names} vs {kept}"
    assert "skeleton" not in names, "a heavy-atom skeleton was ranked by electronic energy"
    assert "frag" not in names, "an R-group placeholder was ranked as a species"


def test_the_sdf_isomer_engine_re_embeds_implicit_h_and_skips_only_the_dummy(tmp_path, caplog):
    """``main()``'s SDF path is the one reader an implicit-H record is fine for.

    ``RDKitSdfIsomer`` calls ``Chem.AddHs`` and re-embeds every record it keeps,
    so a heavy-atom skeleton is legitimate input rather than a defect -- the
    deliberate divergence from the single-file entry points above. Only the
    dummy-atom record is refused (N-M3: adding hydrogens cannot turn an R-group
    placeholder into a species), and the unparseable one is reported once.
    """
    import logging

    from Auto3D.engines.isomers import IsomerEngineFactory

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    out = tmp_path / "enumerated.sdf"
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        IsomerEngineFactory.create(
            "rdkit_sdf",
            input_path=str(path),
            output_path=str(out),
            max_confs=1,
            threshold=0.3,
            n_jobs=1,
            enumerate_isomers=False,
        ).run()

    # Names are <species>_<isomer>_<conformer>; no fixture name contains '_'.
    prefixes = {
        m.GetProp("_Name").split("_")[0]
        for m in Chem.SDMolSupplier(str(out), removeHs=False)
        if m is not None
    }
    assert prefixes == {"ethanol", "skeleton", "ethane"}
    assert "frag" not in prefixes

    messages = [r.getMessage() for r in caplog.records]
    assert sum("dummy atom" in m for m in messages) == 1, messages
    assert any("frag" in m for m in messages), messages
    # Wording-agnostic: this engine reports the unparseable record in its own
    # words today, and what is pinned here is that it reports it exactly once.
    assert sum("parse" in m for m in messages) == 1, messages


def test_check_sdf_format_reads_through_the_record_policy(tmp_path, caplog):
    """The unparseable record is reported with the shared wording; the dummy record is
    warned about once; the implicit-H record is counted and NOT warned about, because
    main()'s SDF engine adds hydrogens and re-embeds every record."""
    import logging
    from unittest.mock import MagicMock

    from Auto3D.orchestration.pipeline.input_checks import check_sdf_format

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    args = MagicMock()
    args.path = str(path)
    args.enumerate_isomer = False
    with caplog.at_level(logging.INFO, logger="Auto3D"):
        ani, only_aimnet = check_sdf_format(args)
    messages = [r.getMessage() for r in caplog.records]
    assert "Skipping record 1: RDKit could not parse it." in messages
    assert sum("dummy atom" in m for m in messages) == 1 and any("frag" in m for m in messages)
    assert not any("skeleton" in m for m in messages)
    assert any("There are 4 conformers" in m for m in messages)
    assert ani is True and only_aimnet == []
    # The WS2 order rule for this path too (the sibling test in
    # tests/test_validation.py covers only check_smi_format): the reassurance
    # must not land before its contradiction. This function has always
    # sequenced the two lines this way.
    order = [
        "dummy" if "dummy atom" in m else "valid"
        for m in messages
        if "dummy atom" in m or "are valid" in m
    ]
    assert order == ["dummy", "valid"], messages


def test_prepare_single_file_run_keeps_exactly_the_kept_records(tmp_path):
    """The shared prologue of `calc_spe`/`opt_geometry`/`calc_thermo`.

    All three read the file through one call now, so this row covers the read
    for all three: what survives, what is handed over to be marked, and how
    many positions had no `Mol` at all. ``AIMNET`` and ``use_gpu=False`` keep
    this in the fast tier -- the engine name resolves from the offline registry
    and the prologue constructs no model.
    """
    from Auto3D.entry._run_setup import prepare_single_file_run

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    setup = prepare_single_file_run(
        str(path),
        "AIMNET",
        gpu_idx=0,
        use_gpu=False,
        allow_tf32=False,
        out_path=None,
        overwrite=True,
        tag="E",
    )

    assert [m.GetProp("_Name") for m in setup.records] == ["ethanol", "ethane"]
    assert [(m.GetProp("_Name"), reason) for m, reason in setup.skipped] == [
        ("skeleton", "implicit_hydrogens"),
        ("frag", "dummy_atoms"),
    ]
    assert setup.unparseable == 1
    # The baseline above, reached through the prologue instead of directly.
    assert [m.GetProp("_Name") for m in setup.records] == classify_records(str(path)).names()
    assert setup.out_path.endswith("mixed_AIMNET_E.sdf")


def test_calc_spe_writes_exactly_the_kept_records(tmp_path, monkeypatch):
    """`calc_spe`'s OUTPUT is the classifier's `kept`, nothing else.

    The prologue row above pins the read; this one pins that the entry point
    writes what it read -- the defective records neither scored nor silently
    carried through. The model machinery is stubbed, since no test here may
    load an NNP; the record policy is upstream of it and runs for real.
    """
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

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    out = spe_mod.calc_spe(str(path), "AIMNET", use_gpu=False, out_path=str(tmp_path / "out.sdf"))

    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None]
    assert names == classify_records(str(path)).names()


@pytest.mark.slow
def test_calc_thermo_marks_the_skipped_records_and_computes_the_kept(tmp_path):
    """`calc_thermo` is the one reader that keeps a defective record.

    The others drop it; this one marks it `Thermo_failed` and writes it,
    because a defect in the INPUT is not a computation that failed and a caller
    filtering on that property must see the record rather than miss it.

    Real ANI2xt on CPU, and therefore slow: the stub-NNP pattern the fast-tier
    thermo rows use (tests/test_thermo_record_assertions.py) works only on an
    all-defective file, since a kept record reaches the stubbed Hessian model.
    This fixture has two kept records, which is the half that pattern cannot
    cover -- and the half that separates "marks the defective ones" from
    "marks everything".

    The kept records' verdict is asserted as "not one of the four skip
    reasons", not as success: whether ANI2xt relaxes ethanol to the threshold
    within the step budget is the potential's business, and a convergence
    marker here is a true outcome. Carrying a RECORD POLICY reason would not be.
    """
    pytest.importorskip("torchani")  # ANI2xt's AEV computer is torchani's

    from Auto3D.entry.ASE.thermo import calc_thermo
    from Auto3D.foundation.utils.convergence import THERMO_FAILED_PROP

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    out = calc_thermo(str(path), "ANI2xt", use_gpu=False, out_path=str(tmp_path / "out.sdf"))

    records = {
        m.GetProp("_Name"): m for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None
    }
    # Four, not five: the unparseable position has no Mol to mark, so it is the
    # only record of the file that does not reach the output.
    assert set(records) == {"ethanol", "ethane", "skeleton", "frag"}
    assert records["skeleton"].GetProp(THERMO_FAILED_PROP) == "implicit_hydrogens"
    assert records["frag"].GetProp(THERMO_FAILED_PROP) == "dummy_atoms"
    skip_reasons = {"unparseable", "no_conformer", "implicit_hydrogens", "dummy_atoms"}
    for name in ("ethanol", "ethane"):
        verdict = records[name].GetProp(THERMO_FAILED_PROP)
        assert verdict not in skip_reasons, f"{name} was marked with a record-policy reason"
