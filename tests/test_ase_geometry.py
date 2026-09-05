import pytest


def test_opt_geometry_names_output_by_model(monkeypatch, tmp_path):
    """Output filename must reflect the model, not always 'userNNP'."""
    import Auto3D.entry.ASE.geometry as geo

    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")  # contents irrelevant; we stub optimizing + supplier

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return True  # matches optimizing.run()'s real True-on-write contract

    monkeypatch.setattr(geo, "optimizing", _Stub)
    monkeypatch.setattr(geo.Chem, "SDMolSupplier", lambda *a, **k: [])
    import torch

    monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)

    # use_gpu=False: this test is about the output filename, not GPU
    # availability. The default use_gpu=True would make check_gpu_requested
    # (called before get_device, which is stubbed here anyway) fail fast with
    # GPUError on a CPU-only runner -- unrelated to what this test checks.
    out = geo.opt_geometry(str(sdf), "AIMNET", use_gpu=False)
    assert out.endswith("mols_AIMNET_opt.sdf")


def test_opt_geometry_skips_none_and_missing_etot(monkeypatch, tmp_path):
    """A None record or one lacking E_tot must be skipped, not crash the run
    (which would discard the whole completed optimization)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    import Auto3D.entry.ASE.geometry as geo

    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")  # contents irrelevant; optimizing + supplier are stubbed

    good = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(good, randomSeed=1)
    good.SetProp("_Name", "good")
    good.SetProp("E_tot", "-100.0")  # Hartree, as optimizing.run() writes it
    no_etot = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(no_etot, randomSeed=2)
    no_etot.SetProp("_Name", "no_etot")  # deliberately missing E_tot

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return True  # matches optimizing.run()'s real True-on-write contract

    monkeypatch.setattr(geo, "optimizing", _Stub)
    # The re-read supplier yields a None record, a record with no E_tot, and a
    # good one; only the good one should survive into the rewritten output.
    monkeypatch.setattr(geo.Chem, "SDMolSupplier", lambda *a, **k: [None, no_etot, good])
    import torch

    monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)

    # use_gpu=False: this test is about the None/missing-E_tot skip logic, not
    # GPU availability -- see the sibling test above for why the default
    # use_gpu=True would fail this on a CPU-only runner for an unrelated reason.
    out = geo.opt_geometry(str(sdf), "AIMNET", use_gpu=False)  # must not raise

    # Read the written file as text (the rdkit.Chem.SDMolSupplier monkeypatch
    # is module-global, so re-reading via it would return the stub list, not
    # the file). Only the good record should have been written.
    with open(out) as fh:
        text = fh.read()
    assert "good" in text
    assert "no_etot" not in text


class TestOptGeometryRaisesWhenNothingWasOptimized:
    """Issue 8: optimizing.run() returns early (no write) for an empty/missing/
    unparseable input, and opt_geometry must not treat that as success.

    Uses the REAL `optimizing` class (not a stub) so the guard is exercised
    against `optimizing.run()`'s actual return-value contract; only
    `create_model`/`get_device`/`configure_torch` are stubbed to avoid loading
    a real NNP or requiring a GPU.

    The input below is deliberately NOT a literal 0-byte file:
    `Chem.SDMolSupplier` raises its own OSError at construction for a truly
    empty file, at the earlier `check_engine_supports_molecules` read
    (unrelated to this guard) -- before `optimizing.run()` is ever reached.
    A single unparseable-but-non-empty record reaches `SDMolSupplier` as a
    `None` entry instead, which is what actually drives `optimizing.run()`'s
    "no valid molecules" early return (batch_opt/batchopt.py) that this guard
    exists to catch.
    """

    _UNPARSEABLE_SDF = "not a real record\n$$$$\n"

    def test_raises_optimization_error_on_unparseable_input(self, tmp_path, monkeypatch):
        import torch

        import Auto3D.entry.ASE.geometry as geo
        from Auto3D.foundation.exceptions import OptimizationError
        from tests.helpers_adapter import FakeAdapter

        bad = tmp_path / "bad.sdf"
        bad.write_text(self._UNPARSEABLE_SDF)

        monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)
        monkeypatch.setattr(geo, "create_model", lambda *a, **k: FakeAdapter())

        with pytest.raises(OptimizationError, match="bad.sdf"):
            geo.opt_geometry(str(bad), "AIMNET", use_gpu=False)

    def test_does_not_silently_return_a_stale_previous_output(self, tmp_path, monkeypatch):
        """The scenario that rules out a plain `os.path.exists(outpath)` guard.

        With overwrite=True (the default), a stale output from an earlier run
        already exists at the derived path. A run against an unparseable
        input must still raise -- not silently re-annotate and return that
        stale file as if it were produced by this call.
        """
        import torch

        import Auto3D.entry.ASE.geometry as geo
        from Auto3D.foundation.exceptions import OptimizationError
        from tests.helpers_adapter import FakeAdapter

        bad = tmp_path / "bad.sdf"
        bad.write_text(self._UNPARSEABLE_SDF)
        stale_out = tmp_path / "bad_AIMNET_opt.sdf"
        stale_out.write_text("STALE PREVIOUS RESULT\n")

        monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)
        monkeypatch.setattr(geo, "create_model", lambda *a, **k: FakeAdapter())

        with pytest.raises(OptimizationError):
            geo.opt_geometry(str(bad), "AIMNET", use_gpu=False)  # overwrite=True default

        assert stale_out.read_text() == "STALE PREVIOUS RESULT\n", (
            "the stale output was modified even though this run produced nothing"
        )
