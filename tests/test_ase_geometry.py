import pytest


@pytest.mark.slow
def test_opt_geometry_skips_implicit_hydrogen_records(tmp_path, caplog):
    """An implicit-H record must not be optimized as a bare heavy-atom
    skeleton (N-C1). `optimizing.run()` (Auto3D.engines.batch_opt.batchopt)
    reads its input through `iter_conformer_records` directly -- the same
    filter `calc_spe`/`opt_geometry`/`select_tautomers` apply to their own
    reads of `path` -- so the implicit-H record is skipped at the one place
    that actually does the optimizing and the writing, not pre-filtered into
    a separate copy of the input. Two records (one implicit-H, one
    explicit-H) so the surviving one proves the filter discriminates rather
    than dropping everything.

    Marked slow for its wall-clock cost (43.6 s on the 2026-10-02 durations
    run), not for GPU or network needs.
    """
    import logging

    pytest.importorskip("torchani")  # ANI2xt's AEV computer is torchani's; two CI legs lack it
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.ASE.geometry import opt_geometry

    no_h = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(no_h, randomSeed=1)
    no_h.SetProp("_Name", "noH")

    with_h = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(with_h, randomSeed=1)
    with_h.SetProp("_Name", "withH")

    p = tmp_path / "in.sdf"
    with Chem.SDWriter(str(p)) as w:
        w.write(no_h)
        w.write(with_h)

    # The bundled ANI2xt weights load on CPU in the fast tier already
    # (tests/test_ani2xt_atom_energies.py); no custom-NNP file is needed.
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        out = opt_geometry(str(p), "ANI2xt", use_gpu=False)

    results = [x for x in Chem.SDMolSupplier(out, removeHs=False) if x is not None]
    assert len(results) == 1
    assert results[0].GetProp("_Name") == "withH"
    assert not any(a.GetTotalNumHs() > 0 for a in results[0].GetAtoms())
    assert any("implicit hydrogen" in r.message for r in caplog.records)


def test_opt_geometry_raises_when_every_record_has_implicit_hydrogens(tmp_path):
    """The all-skipped case must fail fast, not silently return a bogus path.

    Before this record-policy fix, this exact input produced a 1-record
    output scoring the bare {C, C, O} skeleton instead of ethanol.
    """
    pytest.importorskip("torchani")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.ASE.geometry import opt_geometry
    from Auto3D.foundation.exceptions import OptimizationError

    no_h = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(no_h, randomSeed=1)
    no_h.SetProp("_Name", "noH")
    p = tmp_path / "in.sdf"
    with Chem.SDWriter(str(p)) as w:
        w.write(no_h)

    with pytest.raises(OptimizationError, match="in.sdf"):
        opt_geometry(str(p), "ANI2xt", use_gpu=False)


@pytest.mark.slow
def test_opt_geometry_names_output_by_model(monkeypatch, tmp_path):
    """Output filename must reflect the model, not always 'userNNP'.

    Marked slow for its wall-clock cost (5.5 s on the 2026-10-02 durations
    run), not for GPU or network needs.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    import Auto3D.entry.ASE.geometry as geo

    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")  # contents irrelevant; we stub optimizing + supplier

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return True  # matches optimizing.run()'s real True-on-write contract

    # A real record with a conformer, not an empty list: opt_geometry now
    # raises fast (before loading a model) when `iter_conformer_records`
    # yields nothing, so this test's stubbed supplier must still yield
    # something for that guard not to fire before the filename logic below
    # is ever reached.
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol.SetProp("_Name", "m1")

    monkeypatch.setattr(geo, "optimizing", _Stub)
    monkeypatch.setattr(geo.Chem, "SDMolSupplier", lambda *a, **k: [mol])
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
    """Issue 8: opt_geometry must not treat "nothing to optimize" as success.

    Two different guards can fire, and the two tests below exercise the one
    that actually fires for an unparseable-input file: `input_mols =
    list(iter_conformer_records(path))` comes back empty (the record is
    logged and skipped by `iter_conformer_records` itself), which trips
    `opt_geometry`'s own empty-filter fail-fast check -- BEFORE
    `create_model`/`optimizing` are ever reached. That is why neither test
    below stubs `create_model`: doing so would silently pass even if the
    fail-fast check were deleted, since the stub would never be called
    either way.

    `test_raises_optimization_error_when_optimizing_reports_no_write` below
    pins the OTHER guard -- `optimizing.run()` returning `False` -- which
    needs a non-empty `input_mols` to reach at all, so it stubs `optimizing`
    itself (the real `optimizing` class is not used anywhere in this class;
    `get_device`/`configure_torch` are stubbed throughout just to avoid a
    real device lookup or global torch config changes under test).
    """

    _UNPARSEABLE_SDF = "not a real record\n$$$$\n"

    def test_raises_optimization_error_on_unparseable_input(self, tmp_path, monkeypatch):
        import torch

        import Auto3D.entry.ASE.geometry as geo
        from Auto3D.foundation.exceptions import OptimizationError

        bad = tmp_path / "bad.sdf"
        bad.write_text(self._UNPARSEABLE_SDF)

        monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)

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

        bad = tmp_path / "bad.sdf"
        bad.write_text(self._UNPARSEABLE_SDF)
        stale_out = tmp_path / "bad_AIMNET_opt.sdf"
        stale_out.write_text("STALE PREVIOUS RESULT\n")

        monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)

        with pytest.raises(OptimizationError):
            geo.opt_geometry(str(bad), "AIMNET", use_gpu=False)  # overwrite=True default

        assert stale_out.read_text() == "STALE PREVIOUS RESULT\n", (
            "the stale output was modified even though this run produced nothing"
        )

    def test_raises_optimization_error_when_optimizing_reports_no_write(
        self, tmp_path, monkeypatch
    ):
        """`wrote_output is False` must still raise with a non-empty `input_mols`.

        The two tests above both resolve `input_mols == []` and never reach
        `optimizing` at all -- so neither exercises the `if not wrote_output:
        raise` guard in `opt_geometry` any more. This test gives `opt_geometry`
        one real, valid record (so the empty-filter fail-fast does not fire)
        and stubs `optimizing` itself to report `run() -> False`, the contract
        the real class uses for "nothing was written" (missing/empty/no
        parseable record in ITS OWN read of `path` -- a residual case, since a
        non-empty `input_mols` here already implies the real `optimizing`
        would find the same record through its own, identical
        `iter_conformer_records` filter).
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.entry.ASE.geometry as geo
        from Auto3D.foundation.exceptions import OptimizationError
        from tests.helpers_adapter import FakeAdapter

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        mol.SetProp("_Name", "m1")
        sdf = tmp_path / "mols.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        class _StubReportsNoWrite:
            def __init__(self, *a, **k):
                pass

            def run(self):
                return False  # matches optimizing.run()'s real False-on-skip contract

        monkeypatch.setattr(geo, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(geo, "configure_torch", lambda *a, **k: None)
        monkeypatch.setattr(geo, "create_model", lambda *a, **k: FakeAdapter())
        monkeypatch.setattr(geo, "optimizing", _StubReportsNoWrite)

        with pytest.raises(OptimizationError, match="mols.sdf"):
            geo.opt_geometry(str(sdf), "AIMNET", use_gpu=False)
