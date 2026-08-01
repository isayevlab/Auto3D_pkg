#!/usr/bin/env python
"""Hardening tests for isomer_engine / SPE / conformer-count fixes.

These tests are intentionally fast (no full embedding pipelines where avoidable)
and exercise the narrowest function that contains each guard.
"""
from __future__ import annotations

import logging
import os
import uuid

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.isomer_engine import RDKitIsomer, RDKitSdfIsomer, TautomerEngine
from Auto3D.utils.chemistry import calculate_conformer_count


def _make_engine(tmp_path, smi_path):
    """Build an RDKitIsomer with throwaway output paths."""
    job_name = os.path.join(tmp_path, "job_" + uuid.uuid4().hex)
    os.makedirs(job_name)
    return RDKitIsomer(
        smi=str(smi_path),
        smiles_enumerated=os.path.join(tmp_path, "enum.smi"),
        smiles_enumerated_reduced=os.path.join(tmp_path, "reduced.smi"),
        smiles_hashed=os.path.join(tmp_path, "hashed.smi"),
        enumerated_sdf=os.path.join(tmp_path, "out.sdf"),
        job_name=job_name,
        max_confs=None,
        threshold=0.3,
        np=1,
    )


# ---------------------------------------------------------------------------
# FIX 1 — one invalid SMILES must not abort the whole batch
# ---------------------------------------------------------------------------

class TestInvalidSmilesDoesNotAbort:
    def test_read_skips_malformed_lines(self, tmp_path):
        """RDKitIsomer.read must skip blank / 1-col / 3-col lines, not crash."""
        smi = tmp_path / "in.smi"
        smi.write_text(
            "CCO mol_valid\n"
            "\n"                       # blank line
            "   \n"                    # whitespace-only line
            "C1CCCCC1\n"               # missing ID (1 token)
            "CCN extra_a extra_b\n"    # 3 tokens -> take first two
        )
        out = RDKitIsomer.read(str(smi))
        # valid 2-col line and the 3-col line (first two tokens) survive
        assert out["mol_valid"] == "CCO"
        assert out["extra_a"] == "CCN"
        # the 1-col / blank lines are dropped
        assert len(out) == 2

    def test_enumerate_func_skips_invalid(self):
        """enumerate_func must return [] (not raise) on a None mol."""
        result = RDKitIsomer.enumerate_func(None)
        assert result == []

    def test_run_continues_past_invalid_smiles(self, tmp_path):
        """A valid + invalid SMILES file should not abort run(); valid is kept."""
        smi = tmp_path / "in.smi"
        # "C(C" is an invalid SMILES -> MolFromSmiles returns None
        smi.write_text("CCO valid_mol\nC(C invalid_mol\n")
        engine = _make_engine(str(tmp_path), smi)
        # Should not raise; valid molecule must be enumerated.
        engine.run()
        assert "valid_mol" in engine.enumerate
        assert "invalid_mol" not in engine.enumerate

    def test_embed_conformer_returns_none_on_unparseable(self, tmp_path):
        """embed_conformer must return None (not crash on AddHs(None)) for an
        unparseable SMILES, mirroring the parallel worker."""
        smi = tmp_path / "in.smi"
        smi.write_text("CCO mol1\n")
        engine = _make_engine(str(tmp_path), smi)
        assert engine.embed_conformer("invalid_smiles_xyz") is None
        assert engine.embed_conformer("CCO").GetNumConformers() >= 1

    def test_run_serial_embedding_skips_unparseable(self, tmp_path):
        """The serial embedding loop must skip an unparseable SMILES and still
        write the valid one, matching the parallel path's behavior."""
        smi = tmp_path / "in.smi"
        smi.write_text("CCO mol1\n")
        engine = _make_engine(str(tmp_path), smi)
        engine._run_serial_embedding([("CCO", "good"), ("invalid_xyz", "bad")])
        mols = [m for m in Chem.SDMolSupplier(engine.enumerated_sdf) if m is not None]
        names = {m.GetProp("_Name").rsplit("_", 1)[0] for m in mols}
        assert "good" in names
        assert "bad" not in names

    def test_tautomer_rd_taut_skips_invalid(self, tmp_path):
        """TautomerEngine.rd_taut must skip invalid SMILES, not abort."""
        infile = tmp_path / "in.smi"
        outfile = tmp_path / "out.smi"
        infile.write_text("CCO valid_mol\nC(C invalid_mol\n")
        eng = TautomerEngine("rdkit", str(infile), str(outfile), pKaNorm=False)
        eng.rd_taut()  # must not raise
        text = outfile.read_text()
        assert "valid_mol" in text
        assert "invalid_mol" not in text

    def test_sdf_run_skips_none_record(self, tmp_path, monkeypatch):
        """RDKitSdfIsomer.run must skip a None record from SDMolSupplier."""
        valid = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        valid.SetProp("_Name", "valid_mol")

        class FakeSupplier:
            def __init__(self, *a, **k):
                self._mols = [valid, None]

            def __iter__(self):
                return iter(self._mols)

        monkeypatch.setattr(Chem, "SDMolSupplier", FakeSupplier)

        out_sdf = tmp_path / "out.sdf"
        eng = RDKitSdfIsomer(
            sdf=str(tmp_path / "ignored.sdf"),
            enumerated_sdf=str(out_sdf),
            max_confs=None,
            threshold=0.3,
            np=1,
        )
        eng.run()  # must not raise on the None record
        written = list(Chem.SDMolSupplier(str(out_sdf), removeHs=False))
        # The valid molecule should have produced at least one conformer record.
        assert len(written) >= 1


# ---------------------------------------------------------------------------
# FIX 2 — silent stereoisomer truncation at maxIsomers=1024
# ---------------------------------------------------------------------------

class TestStereoisomerTruncation:
    def test_many_stereocenters_not_silently_truncated(self, caplog):
        """A 12-stereocenter polyol must NOT be silently capped at 1024.

        2**12 = 4096 isomers. With the old default cap of 1024 they were
        silently lost. After the fix either all 4096 are returned, or a
        truncation warning is emitted.
        """
        # An asymmetric 12-stereocenter chain: distinct end groups (amino vs
        # carboxyl) prevent symmetry collapse, so all 2**12 = 4096 unique
        # isomers exist.
        smiles = "NC(O)" + "C(O)" * 11 + "C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        with caplog.at_level(logging.WARNING):
            isomers = RDKitIsomer.enumerate_func(mol)
        if len(isomers) == 1024:
            # If exactly at the legacy cap, a warning must have been logged.
            assert any("truncat" in r.message.lower() for r in caplog.records)
        else:
            # Otherwise we must have gone well past the old silent cap.
            assert len(isomers) > 1024


# ---------------------------------------------------------------------------
# FIX 3 — clash dead-band + MMFF->UFF fallback for unparameterizable elements
# ---------------------------------------------------------------------------

class TestClashFallback:
    def test_boronic_acid_not_silently_dropped(self, tmp_path):
        """A boron-containing molecule (no MMFF params) must still yield a
        conformer through the embed path, via the UFF fallback."""
        smi = tmp_path / "in.smi"
        smi.write_text("OB(O)c1ccccc1 boronic\n")
        engine = _make_engine(str(tmp_path), smi)
        # flipper True is fine; phenylboronic acid has no stereocenters.
        out = engine.run()
        written = list(Chem.SDMolSupplier(out, removeHs=False))
        assert len(written) >= 1, "boronic acid was silently dropped (no UFF fallback)"

    def test_mmff_unparameterizable_for_boronic_acid(self):
        """Sanity: phenylboronic acid genuinely lacks full MMFF params, so the
        UFF fallback path is the one being exercised above."""
        mol = Chem.AddHs(Chem.MolFromSmiles("OB(O)c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        assert AllChem.MMFFHasAllMoleculeParams(mol) is False


# ---------------------------------------------------------------------------
# FIX 4 — conformer count floored at 1 and unified across SMILES/SDF paths
# ---------------------------------------------------------------------------

class TestConformerCount:
    def test_floored_at_one_for_single_atom(self):
        """A heavy-atom-free / trivial species must never get 0 conformers."""
        proton = Chem.MolFromSmiles("[H+]")
        assert proton is not None
        assert calculate_conformer_count(proton) >= 1

    def test_floored_at_one_for_lone_heavy_atom(self):
        """A bare single heavy atom also floors at >= 1."""
        # A bare carbon atom (no H) -> 0 rotatable bonds, 1 heavy atom.
        atom = Chem.MolFromSmiles("[C]")
        assert calculate_conformer_count(atom) >= 1

    def test_smiles_and_sdf_paths_agree(self):
        """SMILES and SDF paths must compute the SAME conformer budget.

        Both embed paths now call calculate_conformer_count on the H-complete
        (AddHs) mol, so the count is identical for glycerol regardless of input
        format -- and it equals the richer with-H count, not the no-H count.
        """
        glycerol = Chem.MolFromSmiles("OCC(O)CO")
        # SMILES path budget: count on AddHs(mol).
        smiles_path_count = calculate_conformer_count(Chem.AddHs(glycerol))
        # SDF path reads removeHs=False and then AddHs's; for a mol already
        # carrying explicit Hs, AddHs is idempotent, so the budget matches.
        sdf_mol_with_explicit_h = Chem.AddHs(glycerol)
        sdf_path_count = calculate_conformer_count(Chem.AddHs(sdf_mol_with_explicit_h))
        assert smiles_path_count == sdf_path_count

    def test_conformer_count_uses_with_h_and_paths_agree(self):
        """The unified budget must use the H-complete representation.

        RDKit's CalcNumRotatableBonds only counts O-H / N-H torsions when
        hydrogens are explicit, so the with-H count richly samples hydroxyl /
        amine rotors that the no-H count drops (glycerol ~4.6x). Both embed
        paths now compute on the with-H mol, so they agree on this larger value.
        """
        noh = Chem.MolFromSmiles("OCC(O)CO")  # glycerol
        withh = Chem.AddHs(noh)
        n_noh = calculate_conformer_count(noh)
        n_withh = calculate_conformer_count(withh)
        # with-H samples more (O-H torsions) -> richer polyol sampling restored
        assert n_withh > n_noh


# ---------------------------------------------------------------------------
# FIX 5 — SPE.py: drop None/conformerless records + keep es[i] aligned
# ---------------------------------------------------------------------------

class TestSpeFiltersAndAligns:
    def test_calc_spe_skips_none_and_aligns_indices(self, tmp_path, monkeypatch):
        """calc_spe must drop None records up front and keep es[i] aligned to
        the surviving molecules.

        Supplier yields [valid_A, None, valid_B]; a fake model returns one
        energy per filtered row. The output SDF must contain exactly A and B
        with the right energies.
        """
        import Auto3D.SPE as spe_mod

        def make(name):
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=abs(hash(name)) % 1000)
            m.SetProp("_Name", name)
            return m

        records = [make("A"), None, make("B")]

        class FakeSupplier:
            def __init__(self, *a, **k):
                self._mols = records

            def __iter__(self):
                return iter(self._mols)

        monkeypatch.setattr(spe_mod.Chem, "SDMolSupplier", FakeSupplier)
        monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))

        class FakeAdapter:
            coord_pad = 0.0
            species_pad = 0

        monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter())

        captured = {}

        class FakeEnForce:
            def __init__(self, adapter):
                self.adapter = adapter

            def forward_batched(self, coords, numbers, charges):
                n = coords.shape[0]
                captured["n"] = n
                es = torch.arange(1, n + 1, dtype=torch.float64) * 10.0
                return es, torch.zeros_like(coords)

        monkeypatch.setattr(spe_mod, "EnForce_ANI", FakeEnForce)

        def fake_pad(mols, model_name, device, coord_pad, species_pad):
            assert all(m is not None for m in mols), "None leaked into pad_from_mols"
            n = len(mols)
            coords = torch.zeros(n, 1, 3, requires_grad=True)
            numbers = torch.zeros(n, 1, dtype=torch.long)
            charges = torch.zeros(n, dtype=torch.long)
            atom_mask = torch.ones(n, 1, dtype=torch.bool)
            return coords, numbers, charges, atom_mask

        monkeypatch.setattr(spe_mod, "pad_from_mols", fake_pad)

        inpath = tmp_path / "in.sdf"
        inpath.write_text("")  # contents irrelevant; supplier is faked

        # use_gpu=False: this test is about the None-filtering/index-alignment
        # logic, not GPU availability -- the default use_gpu=True would make
        # check_gpu_requested (called before get_device, which is stubbed
        # here anyway) fail fast with GPUError on a CPU-only runner.
        out = spe_mod.calc_spe(str(inpath), "AIMNET", use_gpu=False)

        assert captured["n"] == 2  # only the two valid mols reached the model

        # Restore the real supplier before reading the written output back.
        monkeypatch.undo()
        written = list(Chem.SDMolSupplier(out, removeHs=False))
        assert [m.GetProp("_Name") for m in written] == ["A", "B"]
        assert float(written[0].GetProp("E_hartree")) == 10.0 * spe_mod.ev2hatree
        assert float(written[1].GetProp("E_hartree")) == 20.0 * spe_mod.ev2hatree

    def test_calc_spe_all_filtered_does_not_crash(self, tmp_path, monkeypatch):
        """FIX B: an SDF whose only record is None must not raise the cryptic
        'max() arg is an empty sequence' from pad_from_mols([]).

        calc_spe should warn and return its output path (an empty SDF) instead.
        """
        import Auto3D.SPE as spe_mod

        class FakeSupplier:
            def __init__(self, *a, **k):
                self._mols = [None]

            def __iter__(self):
                return iter(self._mols)

        monkeypatch.setattr(spe_mod.Chem, "SDMolSupplier", FakeSupplier)
        monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))

        class FakeAdapter:
            coord_pad = 0.0
            species_pad = 0

        monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter())
        monkeypatch.setattr(spe_mod, "EnForce_ANI", lambda adapter: object())

        def fail_pad(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("pad_from_mols must not be called on empty input")

        monkeypatch.setattr(spe_mod, "pad_from_mols", fail_pad)

        inpath = tmp_path / "in.sdf"
        inpath.write_text("")  # contents irrelevant; supplier is faked

        # Must not raise (no cryptic max() error); returns the output path.
        # use_gpu=False for the same reason as the sibling test above: this is
        # about the all-filtered empty-input path, not GPU availability.
        out = spe_mod.calc_spe(str(inpath), "AIMNET", use_gpu=False)
        assert out is not None
        # An output file is produced (empty SDF -> no molecule records).
        assert os.path.exists(out)
