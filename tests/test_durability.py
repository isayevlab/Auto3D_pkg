"""Rewriting a file in place must never be able to destroy it.

reorder_sdf writes to a temp file and os.replace()s it, and cleans the temp up
on failure -- but nothing tested that path (M33). opt_geometry does the
opposite: it truncates the file it just read, so a failure mid-write destroys
a completed optimization whose only copy lives there (C14).

TestReorderSdfDurability closes the M33 coverage gap and must PASS: the
tmp+os.replace pattern in reorder_sdf already works. TestOptGeometryDurability
and TestSameFileGuard are tripwires for the still-open C14 defect and must
XFAIL(strict=True) -- if a future fix makes either XPASS, strict mode turns
that into a hard failure so the xfail marker has to be removed deliberately.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.utils.file_ops import reorder_sdf

# Captured once, at import time, before any test monkeypatches Chem.SDWriter.
# Constructing a real Chem.SDWriter on an existing path truncates it
# immediately (verified empirically: size drops to 0 before any write() call),
# so stand-ins below that need the *real* truncate-on-open behavior go through
# this reference instead of the (possibly patched) `Chem.SDWriter` attribute.
_real_sdwriter = Chem.SDWriter


def _write_sdf(path, names):
    """Write one embedded ethanol conformer per name."""
    with Chem.SDWriter(str(path)) as w:
        for name in names:
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol.SetProp("_Name", name)
            mol.SetProp("ID", name)
            w.write(mol)


class TestReorderSdfDurability:
    """A failed reorder must leave the original file intact."""

    def test_original_survives_a_writer_failure(self, job_dir, monkeypatch):
        """If SDWriter raises mid-rewrite, the input SDF must be unchanged.

        The stand-in wraps the *real* SDWriter (via the module-level
        ``_real_sdwriter`` captured before any monkeypatching -- same
        technique as ``FlakyWriter`` in ``TestOptGeometryDurability`` below)
        so opening it performs a genuine truncate-on-open against whatever
        path ``reorder_sdf`` actually hands it. A stub that never touches
        disk would pass identically whether ``reorder_sdf`` writes to a tmp
        file (correct) or directly to ``sdf`` (a regression) -- this makes
        the test sensitive to *which path* gets opened.
        """
        sdf = job_dir / "out.sdf"
        smi = job_dir / "in.smi"
        _write_sdf(sdf, ["a", "b"])
        smi.write_text("CCO a\nCCO b\n")

        original = sdf.read_bytes()

        class ExplodingWriter:
            def __init__(self, path, *a, **k):
                self._real = _real_sdwriter(path, *a, **k)

            def write(self, *a, **k):
                raise RuntimeError("disk full")

            def close(self):
                self._real.close()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._real.close()
                return False

        monkeypatch.setattr(Chem, "SDWriter", ExplodingWriter)

        with pytest.raises(Exception):
            reorder_sdf(str(sdf), str(smi))

        assert sdf.read_bytes() == original, "the original SDF was corrupted"

    def test_no_temp_file_is_left_behind(self, job_dir, monkeypatch):
        """A failed reorder must not leave a .tmp artifact next to the output.

        ``boom`` actually opens (and closes) the real writer at whatever path
        it is given before raising, so a genuine tmp file exists on disk
        ahead of the simulated crash -- otherwise there would be nothing for
        the cleanup code (``file_ops.py``'s ``tmp_path.unlink()``) to ever
        leave behind if that cleanup were removed, and this test would pass
        vacuously.
        """
        sdf = job_dir / "out.sdf"
        smi = job_dir / "in.smi"
        _write_sdf(sdf, ["a"])
        smi.write_text("CCO a\n")

        def boom(path, *a, **k):
            w = _real_sdwriter(path, *a, **k)
            w.close()
            raise RuntimeError("disk full")

        monkeypatch.setattr(Chem, "SDWriter", boom)
        with pytest.raises(Exception):
            reorder_sdf(str(sdf), str(smi))

        leftovers = [p.name for p in job_dir.iterdir() if ".tmp" in p.name]
        assert not leftovers, f"temp files left behind: {leftovers}"


class TestOptGeometryDurability:
    """opt_geometry must not truncate the file it is rewriting."""

    @pytest.mark.xfail(
        strict=True,
        reason="C14: ASE/geometry.py:106-116 opens SDWriter on the same path it "
        "just read with SDMolSupplier, so a failure between those lines destroys "
        "the completed optimization -- optimizing.run() wrote its only copy there",
    )
    def test_input_survives_a_failed_rewrite(self, job_dir, monkeypatch):
        """A write failure partway through the unit-conversion rewrite pass must
        not lose the completed optimization.

        Fork resolution (see task-5 brief, Step 2): `geometry._annotate_and_rewrite`
        does not exist -- ASE/geometry.py:106-116 inlines the read-then-rewrite
        with no importable helper, and Phase 6 (not this task) owns extracting
        one. Chosen option: "Preferred" -- drive the real `opt_geometry` with
        the batch optimizer (Auto3D.batch_opt.batchopt.optimizing) monkeypatched
        so no NNP loads. This keeps the test hermetic and in the fast tier while
        leaving the vulnerable geometry.py:106-116 pass running unmodified.

        Opening Chem.SDWriter on an existing path truncates it immediately (see
        `_real_sdwriter` note above), so FlakyWriter below wraps the *real*
        writer -- letting that real truncation happen exactly as it would in
        production -- and only injects the write() failure after one record,
        reproducing an interrupted rewrite against an already-truncated file.
        """
        import Auto3D.ASE.geometry as geometry

        sdf = job_dir / "in.sdf"
        _write_sdf(sdf, ["m1", "m2"])
        outpath = job_dir / "opt_out.sdf"

        completed = {}

        class FakeOptimizing:
            """Stand-in for batch_opt.optimizing: skips NNP inference entirely.

            Simulates a completed optimization by writing E_tot-bearing records
            straight to outpath with the real SDWriter -- this is "the only
            copy" referenced in the xfail reason above.
            """

            def __init__(self, path, outpath, model_name, device, opt_config):
                self._path = path
                self._outpath = outpath

            def run(self):
                mols = list(Chem.SDMolSupplier(str(self._path), removeHs=False))
                with _real_sdwriter(str(self._outpath)) as w:
                    for i, mol in enumerate(mols):
                        mol.SetProp("E_tot", str(1.0 + i))
                        w.write(mol)
                completed["bytes"] = Path(self._outpath).read_bytes()

        monkeypatch.setattr(geometry, "optimizing", FakeOptimizing)

        class FlakyWriter:
            """Wraps the real SDWriter so opening it truncates `outpath` exactly
            like production, then fails after one successful write -- leaving
            the file truncated/partial, the same way a real disk-full crash
            would.
            """

            def __init__(self, path, *a, **k):
                self._real = _real_sdwriter(path, *a, **k)
                self._n = 0

            def write(self, mol):
                self._n += 1
                if self._n >= 2:
                    raise RuntimeError("disk full")
                self._real.write(mol)

            def close(self):
                self._real.close()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._real.close()
                return False

        monkeypatch.setattr(geometry.Chem, "SDWriter", FlakyWriter)

        with pytest.raises(Exception):
            # Model name must resolve (C11 added an engine-name guard inside
            # opt_geometry itself): "fake-model" was never load-bearing here
            # -- FakeOptimizing.__init__ doesn't even store it -- the point of
            # this test is durability of the input file against a failed
            # rewrite, not the model name, so a valid engine name is used.
            geometry.opt_geometry(
                str(sdf), "AIMNET", out_path=str(outpath), use_gpu=False
            )

        assert "bytes" in completed, "the fake optimizer never ran"
        assert outpath.read_bytes() == completed["bytes"], (
            "opt_geometry truncated its own output; a completed optimization was lost"
        )


class TestSameFileGuard:
    """Writing output over the input must be refused or staged atomically."""

    @pytest.mark.xfail(
        strict=True,
        reason="C14: calc_spe / opt_geometry / calc_thermo do not guard "
        "out_path == input path, so `auto3d energy mols.sdf -o mols.sdf` "
        "overwrites the user's input with no atomic staging",
    )
    def test_output_equal_to_input_is_rejected(self, job_dir, monkeypatch):
        """Passing out_path == path must raise rather than silently clobber input.

        calc_spe reads every input molecule into a Python list before it ever
        opens a writer, so exercising it with a real "AIMNET" run would not
        even reproduce a truncation -- it would just quietly succeed and
        overwrite the input, which is the very absence of a guard this test is
        checking for. To stay within the box's "never load an NNP" constraint,
        the model machinery (get_device/create_model/EnForce_ANI/pad_from_mols)
        is stubbed out the same way tests/test_isomer_engine_hardening.py
        already does for calc_spe; the guard being tested for (or its absence)
        sits earlier in the real function, so calc_spe itself still runs for
        real here.
        """
        import Auto3D.SPE as spe_mod
        from Auto3D.exceptions import Auto3DError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])

        monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))

        class FakeAdapter:
            coord_pad = 0.0
            species_pad = 0

        monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter())

        class FakeEnForce:
            def __init__(self, adapter):
                pass

            def forward_batched(self, coords, numbers, charges):
                n = coords.shape[0]
                return torch.ones(n, dtype=torch.float64), torch.zeros_like(coords)

        monkeypatch.setattr(spe_mod, "EnForce_ANI", FakeEnForce)

        def fake_pad(mols, model_name, device, coord_pad, species_pad):
            n = len(mols)
            coords = torch.zeros(n, 1, 3)
            numbers = torch.zeros(n, 1, dtype=torch.long)
            charges = torch.zeros(n, dtype=torch.long)
            atom_mask = torch.ones(n, 1, dtype=torch.bool)
            return coords, numbers, charges, atom_mask

        monkeypatch.setattr(spe_mod, "pad_from_mols", fake_pad)

        # use_gpu=False: this test is about the missing same-file guard (C14),
        # not GPU availability. The default use_gpu=True made check_gpu_requested
        # (which now runs before any of the same-file logic below it) raise
        # GPUError -- itself an Auto3DError -- on a CPU-only runner, which
        # satisfied this test's `pytest.raises((Auto3DError, ValueError))` for
        # the wrong reason and turned the intended XFAIL into an XPASS (a hard
        # failure under strict=True). use_gpu=False does not weaken what this
        # test proves: it lets calc_spe run past the GPU check (as it always
        # does on a GPU-equipped box) so the test again exercises -- and still
        # finds absent -- the same-file guard itself, consistently regardless
        # of whether CUDA is present.
        with pytest.raises((Auto3DError, ValueError)):
            spe_mod.calc_spe(str(sdf), "AIMNET", out_path=str(sdf), use_gpu=False)
