"""Rewriting a file in place must never be able to destroy it.

reorder_sdf writes to a temp file and os.replace()s it, and cleans the temp up
on failure -- but nothing tested that path (M33). opt_geometry used to do the
opposite: it truncated the file it had just read, so a failure mid-write
destroyed a completed optimization whose only copy lived there (C14).

TestReorderSdfDurability closes the M33 coverage gap and must PASS: the
tmp+os.replace pattern in reorder_sdf already works. TestOptGeometryDurability
and TestAmendConfigurationDurability must now also PASS -- Phase 6 gave both
call sites the same tmp+os.replace staging. TestSameFileGuard was the tripwire
for the other half of C14 (no out_path == path guard) and must now also PASS:
`Auto3D.utils.validation.check_output_not_input` refuses that case in all three
entry points, so the xfail(strict=True) marker it carried was removed with the
fix. Every class in this file is now a plain regression test -- this file
carries no xfail.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import os
import stat

from Auto3D.utils.file_ops import reorder_sdf


class TestStagingLocation:
    """The staged temp file must be a SIBLING of its target.

    `os.replace` is only atomic within one filesystem and raises
    `OSError: [Errno 18] EXDEV` across them, so staging beside the target is
    the property that makes the whole durability fix work. The end-to-end
    durability tests below do NOT pin it: they patch `Chem.SDWriter` module-
    globally, so the injected failure fires wherever the temp file happens to
    live, and their leftover scans only read `job_dir`. Dropping `dir=` from
    `_stage_beside` would leave every one of them green while breaking
    `opt_geometry` on any box where the temp dir is a different mount from the
    output directory -- a separate `/tmp` tmpfs being the common case.
    """

    def test_temp_file_is_created_beside_its_target(self, job_dir):
        from Auto3D.ASE.geometry import _stage_beside

        target = job_dir / "out.sdf"
        target.write_text("placeholder\n")

        tmp_path = _stage_beside(str(target))
        try:
            assert Path(tmp_path).parent == Path(os.path.realpath(str(job_dir))), (
                f"temp file {tmp_path} is not beside its target {target}; "
                "os.replace would raise EXDEV whenever the two differ"
            )
        finally:
            os.unlink(tmp_path)

    def test_temp_file_inherits_the_target_mode(self, job_dir):
        """mkstemp creates 0600 and os.replace carries the SOURCE mode, so
        without this the rewrite would silently tighten every output file."""
        from Auto3D.ASE.geometry import _stage_beside

        target = job_dir / "out.sdf"
        target.write_text("placeholder\n")
        os.chmod(target, 0o644)

        tmp_path = _stage_beside(str(target))
        try:
            assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o644
        finally:
            os.unlink(tmp_path)

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

    def test_input_survives_a_failed_rewrite(self, job_dir, monkeypatch):
        """A write failure partway through the unit-conversion rewrite pass must
        not lose the completed optimization.

        Phase 6 fixed C14 by extracting `geometry._annotate_and_rewrite`, which
        stages the eV->hartree pass into a sibling temp file and `os.replace`s
        it into position, so this test now passes (the xfail(strict=True)
        tripwire it carried was removed with that fix). It drives the real
        `opt_geometry` with the batch optimizer
        (Auto3D.batch_opt.batchopt.optimizing) monkeypatched so no NNP loads,
        keeping the test hermetic and in the fast tier while the real rewrite
        pass runs unmodified.

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
            straight to outpath with the real SDWriter. This is the only copy
            of that finished optimization -- `optimizing.run()` writes nowhere
            else -- which is exactly why truncating `outpath` to rewrite it was
            unrecoverable before C14 was fixed.
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

        # The staging temp file must be cleaned up on failure, otherwise every
        # crashed run leaves an orphan .sdf next to the user's output.
        leftovers = sorted(
            p.name for p in job_dir.iterdir() if p.name not in {"in.sdf", "opt_out.sdf"}
        )
        assert not leftovers, f"temp files left behind: {leftovers}"


class TestAmendConfigurationDurability:
    """amend_configuration_w must not truncate the file it is rewriting."""

    def test_input_survives_a_failed_rewrite(self, job_dir, monkeypatch):
        """A write failure partway through the amend pass must not lose the input.

        ``amend_configuration_w`` read the .smi file and then reopened the same
        path with ``open(smi, "w+")``, which truncates on open -- so a crash
        partway through the write loop left a partial file and the original
        stereoisomer enumeration was gone (C14, same shape as opt_geometry).

        The stand-in below wraps the *real* ``open`` so the truncate-on-open
        happens exactly as in production against whatever path the function
        actually opens for writing; a stub that never touched disk would pass
        identically whether the function writes to a temp file (correct) or
        straight to ``smi`` (the regression). Only write-mode opens inside
        ``job_dir`` are wrapped, so ``amend_configuration``'s own read of the
        input -- and anything pytest opens -- goes through untouched.
        """
        import builtins

        from Auto3D.utils.stereochemistry import amend_configuration_w

        smi = job_dir / "in.smi"
        # Two records under one id => two output lines, so the injected failure
        # lands *partway* through the write rather than before it starts.
        smi.write_text("C[C@H](O)F mol_1\nC[C@@H](O)F mol_2\n")
        original = smi.read_bytes()

        real_open = builtins.open

        class FlakyFile:
            """Writes the first line for real, then fails like a full disk."""

            def __init__(self, real):
                self._real = real
                self._n = 0

            def write(self, s):
                self._n += 1
                if self._n >= 2:
                    raise RuntimeError("disk full")
                return self._real.write(s)

            def close(self):
                self._real.close()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._real.close()
                return False

        def flaky_open(file, mode="r", *a, **k):
            handle = real_open(file, mode, *a, **k)
            if "w" in mode and str(file).startswith(str(job_dir)):
                return FlakyFile(handle)
            return handle

        monkeypatch.setattr(builtins, "open", flaky_open)

        with pytest.raises(RuntimeError):
            amend_configuration_w(str(smi))

        monkeypatch.undo()

        assert smi.read_bytes() == original, (
            "amend_configuration_w truncated its own input; the file is unrecoverable"
        )
        leftovers = sorted(p.name for p in job_dir.iterdir() if p.name != "in.smi")
        assert not leftovers, f"temp files left behind: {leftovers}"


class TestSameFileGuard:
    """Writing output over the input must be refused, in every entry point.

    `check_output_not_input` is one shared function called from calc_spe,
    opt_geometry and calc_thermo, so each of the three gets its own test here:
    a guard wired into only two of the three would otherwise sail through with
    the third silently still destroying its caller's input.

    Every test asserts `ConfigurationError`, never the base `Auto3DError`.
    `check_gpu_requested` runs *before* this guard in all three functions and
    raises `GPUError` -- also an `Auto3DError` -- so a base-class assertion is
    satisfied on any CPU-only runner without ever reaching the subject. That
    exact defect shipped once already (see the Phase 5 note that used to live
    in this class), and `use_gpu=False` below plus the narrow exception type
    are what keep it from recurring.
    """

    def test_output_equal_to_input_is_rejected(self, job_dir, monkeypatch):
        """Passing out_path == path must raise rather than silently clobber input.

        calc_spe reads every input molecule into a Python list before it ever
        opens a writer, so exercising it with a real "AIMNET" run would not
        even reproduce a truncation -- it would just quietly succeed and
        overwrite the input, which is exactly the damage the guard prevents.
        To stay within the box's "never load an NNP" constraint, the model
        machinery (get_device/create_model/EnForce_ANI/pad_from_mols) is
        stubbed out the same way tests/test_isomer_engine_hardening.py already
        does for calc_spe; the guard sits earlier in the real function, so
        calc_spe itself still runs for real here -- and if the guard were
        removed, those stubs would let the run complete and overwrite `sdf`
        without raising, failing this test.
        """
        import Auto3D.SPE as spe_mod
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        original = sdf.read_bytes()

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

        # The stubs above are benign, so on their own they would let the guard
        # move BELOW model construction and this test would stay green while
        # `auto3d energy huge.sdf -o huge.sdf` loaded a full AIMNet2 before
        # refusing. Make reaching model construction itself the failure, the
        # way the calc_thermo test below already does. Still loads no NNP.
        def _never(*args, **kwargs):
            raise AssertionError(
                "calc_spe reached model construction; the same-file guard must "
                "refuse out_path == path before any model is built"
            )

        monkeypatch.setattr(spe_mod, "create_model", _never)

        # use_gpu=False so check_gpu_requested (which runs first) cannot raise
        # GPUError and satisfy the assertion below for the wrong reason -- see
        # the class docstring.
        with pytest.raises(ConfigurationError, match="same file"):
            spe_mod.calc_spe(str(sdf), "AIMNET", out_path=str(sdf), use_gpu=False)

        assert sdf.read_bytes() == original, "calc_spe modified the input file"

    def test_opt_geometry_rejects_output_equal_to_input(self, job_dir, monkeypatch):
        """Same guard, second entry point.

        `optimizing` is replaced with a stand-in that writes real records to
        whatever outpath it is handed (so no NNP loads, per the box limits).
        Without the guard, opt_geometry would run that stand-in to completion
        against the input file and return normally -- no exception, input
        destroyed -- which is what makes this test go red if the
        check_output_not_input call in opt_geometry is removed.
        """
        import Auto3D.ASE.geometry as geometry
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1", "m2"])
        original = sdf.read_bytes()

        class FakeOptimizing:
            def __init__(self, path, outpath, model_name, device, opt_config):
                self._path = path
                self._outpath = outpath

            def run(self):
                mols = list(Chem.SDMolSupplier(str(self._path), removeHs=False))
                with _real_sdwriter(str(self._outpath)) as w:
                    for i, mol in enumerate(mols):
                        mol.SetProp("E_tot", str(1.0 + i))
                        w.write(mol)

        monkeypatch.setattr(geometry, "optimizing", FakeOptimizing)

        # FakeOptimizing is benign, so it alone would let the guard move below
        # model construction with this test still green -- see the calc_spe
        # test above. Make reaching the device/model step the failure itself.
        def _never(*args, **kwargs):
            raise AssertionError(
                "opt_geometry reached model construction; the same-file guard "
                "must refuse out_path == path before any model is built"
            )

        monkeypatch.setattr(geometry, "get_device", _never)

        with pytest.raises(ConfigurationError, match="same file"):
            geometry.opt_geometry(
                str(sdf), "AIMNET", out_path=str(sdf), use_gpu=False
            )

        assert sdf.read_bytes() == original, "opt_geometry modified the input file"

    def test_calc_thermo_rejects_output_equal_to_input(self, job_dir, monkeypatch):
        """Same guard, third entry point.

        calc_thermo's first irreversible step after the guard is model
        construction (`_load_hessian_model` / `model_name2model_calculator`),
        which would download and load an NNP. Both are replaced with stubs that
        fail loudly instead: the guard is specified to run *before* any model
        construction, so reaching either one is itself the bug. If the
        check_output_not_input call in calc_thermo is removed, the stub fires
        and this test goes red (and still no NNP is loaded).
        """
        import Auto3D.ASE.thermo as thermo_mod
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        original = sdf.read_bytes()

        def _never(*args, **kwargs):
            raise AssertionError(
                "calc_thermo reached model construction; the same-file guard "
                "must refuse out_path == path before any model is built"
            )

        monkeypatch.setattr(thermo_mod, "_load_hessian_model", _never)
        monkeypatch.setattr(thermo_mod, "model_name2model_calculator", _never)

        with pytest.raises(ConfigurationError, match="same file"):
            thermo_mod.calc_thermo(
                str(sdf), "AIMNET", out_path=str(sdf), use_gpu=False
            )

        assert sdf.read_bytes() == original, "calc_thermo modified the input file"

    def test_guard_compares_real_paths_not_strings(self, job_dir, monkeypatch):
        """A bare relative name, `./mols.sdf`, and a symlink all name the same
        file as the absolute input path, and a string comparison would let
        every one of them through.

        Note which spellings actually discriminate. `job_dir` is already
        absolute, so `str(sdf)` and `os.path.abspath(str(sdf))` are the
        IDENTICAL string -- both are caught by a naive string comparison too,
        and neither tests anything about path resolution. Only the three
        non-identical spellings below carry weight, which is why the bare
        relative name (`mols.sdf`, via monkeypatch.chdir) is included: it is
        the case this test is named for, and it was previously described in
        this docstring without ever being passed.

        The guard must also stay quiet for a genuinely different output path --
        a guard that always raised would satisfy every test above.
        """
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.utils.validation import check_output_not_input

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        link = job_dir / "alias.sdf"
        link.symlink_to(sdf)

        # None (use the default output name) and a different file: allowed.
        check_output_not_input(str(sdf), None)
        check_output_not_input(str(sdf), str(job_dir / "other.sdf"))

        # Same file, spelled every way that differs from the input string:
        # all refused. `str(sdf)` itself is omitted -- it is the trivial case
        # already covered by the three entry-point tests above.
        monkeypatch.chdir(job_dir)
        for spelling in (
            "mols.sdf",  # bare relative, resolved against the cwd
            os.path.join(".", "mols.sdf"),
            os.path.join(str(job_dir), ".", "mols.sdf"),
            str(link),  # symlink to the input
        ):
            assert spelling != str(sdf), (
                f"{spelling!r} is string-identical to the input, so it cannot "
                "show that the guard resolves paths"
            )
            with pytest.raises(ConfigurationError, match="same file"):
                check_output_not_input(str(sdf), spelling)

    def test_a_hardlink_to_the_input_is_refused(self, job_dir):
        """One file under two names must be caught, not just one path spelled
        two ways.

        `cp -l mols.sdf results.sdf` makes a single inode reachable by two
        distinct names, so `os.path.realpath` resolves them to two DIFFERENT
        strings and a realpath-only guard lets the write through -- destroying
        the input it was supposed to protect. `os.path.samefile` compares
        st_dev/st_ino, so it sees one file. Verified before the fix: the guard
        returned normally here.

        This is also the case that stands in for case-insensitive filesystems
        (macOS APFS, Windows NTFS -- both supported): `Mols.sdf` and `mols.sdf`
        are likewise one inode whose real paths differ. That cannot be tested
        directly on this ext4 box, but it takes the identical samefile path.
        """
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.utils.validation import check_output_not_input

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["a"])
        hardlink = job_dir / "results.sdf"
        os.link(sdf, hardlink)

        # Precondition: this is exactly the case realpath cannot see.
        assert os.path.realpath(str(sdf)) != os.path.realpath(str(hardlink))
        assert os.path.samefile(str(sdf), str(hardlink))

        with pytest.raises(ConfigurationError, match="same file"):
            check_output_not_input(str(sdf), str(hardlink))

    def test_a_genuinely_different_output_is_still_allowed(self, job_dir):
        """Negative control: without this, a guard that always raised would
        satisfy every other test in this class."""
        from Auto3D.utils.validation import check_output_not_input

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["a"])

        # Both an existing sibling and a not-yet-created output must pass.
        other = job_dir / "other.sdf"
        _write_sdf(other, ["b"])
        check_output_not_input(str(sdf), str(other))
        check_output_not_input(str(sdf), str(job_dir / "not_created_yet.sdf"))


class TestConformerRankerSameFileGuard:
    """ConformerRanker is a fourth public writer with the same exposure.

    The audit found the same-file hazard by grepping for `check_gpu_requested`
    call sites, which by construction can only return functions that are
    ALREADY guarded -- so a writer that never had a GPU check was invisible to
    that search. `ConformerRanker` reads `input_path` and opens
    `Chem.SDWriter(self.out_path)` in `run()`, so the same-file case replaces
    the user's input with the selected subset.
    """

    def test_output_equal_to_input_is_rejected_at_construction(self, job_dir):
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.ranking import ConformerRanker

        sdf = job_dir / "opt.sdf"
        _write_sdf(sdf, ["a", "b"])
        original = sdf.read_bytes()

        with pytest.raises(ConfigurationError, match="same file"):
            ConformerRanker(
                input_path=str(sdf), out_path=str(sdf), threshold=0.3, k=1
            )

        assert sdf.read_bytes() == original, "the input was modified anyway"

    def test_a_different_output_still_constructs(self, job_dir):
        """Negative control: the guard must not reject ordinary use."""
        from Auto3D.ranking import ConformerRanker

        sdf = job_dir / "opt.sdf"
        _write_sdf(sdf, ["a"])
        ranker = ConformerRanker(
            input_path=str(sdf), out_path=str(job_dir / "ranked.sdf"),
            threshold=0.3, k=1,
        )
        assert ranker.out_path.endswith("ranked.sdf")


class _StopAfterEncodingError(Exception):
    """Sentinel: the pipeline reached the phase right after encoding.

    Raised from a stubbed ``_setup_logging`` so ``run()`` executes Phase 1 for
    real -- validation, job-directory creation, ``encode_ids`` -- and then
    unwinds through the same ``finally`` a real run does, without forking a
    worker or loading an NNP (both disallowed on this box).
    """


def _stop_after_encoding(self):
    raise _StopAfterEncodingError()


class TestEncodedInputStaging:
    """`auto3d run mols.smi` must not touch a user's `mols_encoded.smi`.

    ``encode_ids`` names its output ``<stem>_encoded.<ext>``, derived from the
    input. That collides with a perfectly ordinary user file, and the collision
    used to be silent and doubly fatal: ``encode_ids`` opened the path for
    writing (truncating it), and ``run()``'s ``finally`` then ``unlink()``ed
    it. The comment on that unlink claimed its ``is_file()`` test "guards
    against ever unlinking anything but a real encoded input" -- it cannot tell
    Auto3D's file from the user's, and both halves of the damage were
    reproduced against a file containing "IRREPLACEABLE USER DATA".

    The fix stages the encoded copy inside the run's own job directory, which
    ``_setup_job_directory`` creates with a bare ``mkdir()`` immediately
    before. That directory is provably new, which is what makes both the write
    and the unlink safe -- so the second test below pins the ``mkdir()``, not
    just the location.
    """

    def test_a_users_encoded_file_is_untouched_byte_for_byte(
        self, job_dir, monkeypatch
    ):
        """The whole point: run the pipeline over `mols.smi` with a user's
        `mols_encoded.smi` sitting beside it, and that file must come back
        identical -- not merely still present.

        Byte comparison, not `exists()`: the old code overwrote the file with
        encoded output *and then* deleted it, so an existence check alone
        would also have been satisfied by an overwrite that happened not to be
        cleaned up. `id_mapping` is asserted too, because the run must
        *succeed* past encoding -- a fix that refused the run whenever the
        encoded name was taken would leave the file intact and still be wrong.
        """
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

        smi = job_dir / "mols.smi"
        smi.write_text("CCO ethanol\n")
        users_file = job_dir / "mols_encoded.smi"
        users_file.write_bytes(b"IRREPLACEABLE USER DATA\n")
        original = users_file.read_bytes()

        monkeypatch.setattr(
            WorkflowOrchestrator, "_setup_logging", _stop_after_encoding
        )
        # ANI2xt is bundled, so Phase 1's real preflight_model stays offline
        # (no registry lookup, no download); use_gpu=False keeps the GPU check
        # from failing for an unrelated reason.
        orch = WorkflowOrchestrator(
            Auto3DOptions(
                path=str(smi), k=1, use_gpu=False, optimizing_engine="ANI2xt"
            )
        )

        with pytest.raises(_StopAfterEncodingError):
            orch.run()

        assert orch.id_mapping == {"ethanol": 0}, (
            "encoding never completed, so this test would pass even if the "
            "run were simply refused whenever mols_encoded.smi exists"
        )
        assert users_file.exists(), "the user's file was deleted by the run"
        assert users_file.read_bytes() == original, (
            "the user's mols_encoded.smi was overwritten with encoded output"
        )

        # The encoded copy went into the run's own directory, and the cleanup
        # in run()'s finally removed *that* file rather than the user's.
        assert orch.input_path.parent == orch.job_dir, (
            f"encoded input {orch.input_path} was not staged inside the job "
            f"directory {orch.job_dir}"
        )
        assert not orch.input_path.exists()

    def test_the_job_directory_is_never_reused(self, job_dir, monkeypatch):
        """`_setup_job_directory` must fail on a name collision, not merge.

        Everything above rests on the job directory being new: that is the
        only reason `_encode_input` may write into it without an existence
        check and `run()`'s `finally` may unlink out of it. A stray
        `exist_ok=True` would silently restore the original defect one level
        up -- Auto3D would clobber and delete inside a directory the user
        already owned -- while leaving the first test green, since it never
        collides on the job name.
        """
        from Auto3D.config import Auto3DOptions
        from Auto3D.workflow import WorkflowOrchestrator

        smi = job_dir / "mols.smi"
        smi.write_text("CCO ethanol\n")
        # job_name is pinned so the directory name is predictable; the real
        # default is a microsecond timestamp.
        squatted = job_dir / "mols_pinned"
        squatted.mkdir()
        precious = squatted / "results.sdf"
        precious.write_bytes(b"EARLIER RUN\n")

        monkeypatch.setattr(
            WorkflowOrchestrator, "_setup_logging", _stop_after_encoding
        )
        orch = WorkflowOrchestrator(
            Auto3DOptions(
                path=str(smi), k=1, use_gpu=False,
                optimizing_engine="ANI2xt", job_name="pinned",
            )
        )

        with pytest.raises(FileExistsError):
            orch.run()

        assert precious.read_bytes() == b"EARLIER RUN\n"
        assert sorted(p.name for p in squatted.iterdir()) == ["results.sdf"], (
            "the run wrote into a directory it did not create"
        )


class TestOutputOverwriteGuard:
    """An existing output file must not be truncated without consent.

    `auto3d energy junk.sdf --no-gpu -o precious.sdf` exited 0, printed
    "Wrote precious.sdf", and left precious.sdf at 0 bytes: `Chem.SDWriter`
    truncates on open, and a run whose every record failed to parse then wrote
    nothing back. `auto3d config init` has refused to clobber since it
    shipped; the calculators did not.

    `check_output_overwrite` is one shared function called from calc_spe,
    opt_geometry, calc_thermo and ConformerRanker, so -- exactly as for
    `TestSameFileGuard` above -- each gets its own test here: a guard wired
    into three of the four would otherwise sail through with the fourth
    silently still destroying its caller's file.

    Every test below passes `overwrite=False` explicitly, because the API
    parameter deliberately defaults to True: `--force` is a CLI concept, and
    a strict API default would break every existing Python caller that
    re-runs into the same output path. The CLI is what supplies False (see
    `tests/test_cli_property_commands.py`, which pins that wiring); these
    tests pin what the API does once it is supplied.

    Every test asserts `ConfigurationError`, never the base `Auto3DError`:
    `check_gpu_requested` runs earlier in all three API functions and raises
    `GPUError`, which is also an `Auto3DError` and would satisfy a base-class
    assertion on any CPU-only runner without ever reaching the subject.
    `use_gpu=False` below is the other half of that protection.
    """

    def test_calc_spe_refuses_an_existing_output(self, job_dir, monkeypatch):
        import Auto3D.SPE as spe_mod
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        precious = job_dir / "precious.sdf"
        precious.write_bytes(b"IRREPLACEABLE USER DATA\n")
        original = precious.read_bytes()

        # The guard is specified to run before any model construction, so
        # reaching either of these is itself the bug -- and it keeps the test
        # from loading an NNP if the guard is ever removed.
        def _never(*args, **kwargs):
            raise AssertionError(
                "calc_spe reached model construction; the overwrite guard "
                "must refuse an existing output before any model is built"
            )

        monkeypatch.setattr(spe_mod, "get_device", _never)
        monkeypatch.setattr(spe_mod, "create_model", _never)

        with pytest.raises(ConfigurationError, match="already exists"):
            spe_mod.calc_spe(
                str(sdf), "AIMNET", out_path=str(precious), use_gpu=False,
                overwrite=False,
            )

        assert precious.read_bytes() == original

    def test_calc_spe_overwrites_when_the_caller_asks(self, job_dir, monkeypatch):
        """Negative control: `overwrite=True` (what `--force` sets) must still
        replace the file, and the API default must stay permissive so no
        existing Python caller breaks.

        Without this, a guard that refused unconditionally would satisfy every
        other test in this class.
        """
        import Auto3D.SPE as spe_mod

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        stale = job_dir / "stale.sdf"
        stale.write_bytes(b"OLD RESULTS\n")

        class FakeAdapter:
            coord_pad = 0.0
            species_pad = 0

        class FakeEnForce:
            def __init__(self, adapter):
                pass

            def forward_batched(self, coords, numbers, charges):
                n = coords.shape[0]
                return torch.ones(n, dtype=torch.float64), torch.zeros_like(coords)

        def fake_pad(mols, model_name, device, coord_pad, species_pad):
            n = len(mols)
            return (
                torch.zeros(n, 1, 3),
                torch.zeros(n, 1, dtype=torch.long),
                torch.zeros(n, dtype=torch.long),
                torch.ones(n, 1, dtype=torch.bool),
            )

        monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter())
        monkeypatch.setattr(spe_mod, "EnForce_ANI", FakeEnForce)
        monkeypatch.setattr(spe_mod, "pad_from_mols", fake_pad)

        out = spe_mod.calc_spe(
            str(sdf), "AIMNET", out_path=str(stale), use_gpu=False, overwrite=True
        )

        assert out == str(stale)
        assert stale.read_bytes() != b"OLD RESULTS\n", "the run did not write"
        assert "E_hartree" in stale.read_text()

        # And the default is permissive: the same call without `overwrite`
        # must also go through, which is what keeps every pre-existing
        # Python-API caller working.
        spe_mod.calc_spe(str(sdf), "AIMNET", out_path=str(stale), use_gpu=False)

    def test_the_derived_default_output_path_is_guarded_too(
        self, job_dir, monkeypatch
    ):
        """The guard is on the resolved path, not on `out_path is not None`.

        A second `auto3d energy mols.sdf` writes `mols_AIMNET_E.sdf` -- the
        first run's results -- with no `-o` anywhere in sight. Checking only
        the explicit `-o` would leave that case silently destructive.
        """
        import Auto3D.SPE as spe_mod
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        earlier = job_dir / "mols_AIMNET_E.sdf"
        earlier.write_bytes(b"FIRST RUN RESULTS\n")

        def _never(*args, **kwargs):
            raise AssertionError("model construction reached")

        monkeypatch.setattr(spe_mod, "get_device", _never)
        monkeypatch.setattr(spe_mod, "create_model", _never)

        with pytest.raises(ConfigurationError, match="already exists"):
            spe_mod.calc_spe(str(sdf), "AIMNET", use_gpu=False, overwrite=False)

        assert earlier.read_bytes() == b"FIRST RUN RESULTS\n"

    def test_opt_geometry_refuses_an_existing_output(self, job_dir, monkeypatch):
        import Auto3D.ASE.geometry as geometry
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        precious = job_dir / "precious.sdf"
        precious.write_bytes(b"IRREPLACEABLE USER DATA\n")
        original = precious.read_bytes()

        def _never(*args, **kwargs):
            raise AssertionError(
                "opt_geometry reached model construction; the overwrite guard "
                "must refuse an existing output before any model is built"
            )

        monkeypatch.setattr(geometry, "get_device", _never)
        monkeypatch.setattr(geometry, "optimizing", _never)

        with pytest.raises(ConfigurationError, match="already exists"):
            geometry.opt_geometry(
                str(sdf), "AIMNET", out_path=str(precious), use_gpu=False,
                overwrite=False,
            )

        assert precious.read_bytes() == original

    def test_calc_thermo_refuses_an_existing_output(self, job_dir, monkeypatch):
        import Auto3D.ASE.thermo as thermo_mod
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        precious = job_dir / "precious.sdf"
        precious.write_bytes(b"IRREPLACEABLE USER DATA\n")
        original = precious.read_bytes()

        def _never(*args, **kwargs):
            raise AssertionError(
                "calc_thermo reached model construction; the overwrite guard "
                "must refuse an existing output before any model is built"
            )

        monkeypatch.setattr(thermo_mod, "_load_hessian_model", _never)
        monkeypatch.setattr(thermo_mod, "model_name2model_calculator", _never)

        with pytest.raises(ConfigurationError, match="already exists"):
            thermo_mod.calc_thermo(
                str(sdf), "AIMNET", out_path=str(precious), use_gpu=False,
                overwrite=False,
            )

        assert precious.read_bytes() == original

    def test_conformer_ranker_refuses_an_existing_output(self, job_dir):
        """Fourth writer, same exposure -- and the same construction-time
        check as its same-file guard, so it fails before any work."""
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.ranking import ConformerRanker

        sdf = job_dir / "opt.sdf"
        _write_sdf(sdf, ["a", "b"])
        precious = job_dir / "precious.sdf"
        precious.write_bytes(b"IRREPLACEABLE USER DATA\n")

        with pytest.raises(ConfigurationError, match="already exists"):
            ConformerRanker(
                input_path=str(sdf), out_path=str(precious), threshold=0.3,
                k=1, overwrite=False,
            )

        assert precious.read_bytes() == b"IRREPLACEABLE USER DATA\n"

        # Default stays permissive: Auto3D's own pipeline always writes into a
        # job directory it just created, and every existing caller relies on it.
        ranker = ConformerRanker(
            input_path=str(sdf), out_path=str(precious), threshold=0.3, k=1
        )
        assert ranker.out_path == str(precious)

    def test_the_guard_permits_everything_it_should(self, job_dir):
        """Negative controls for the shared function itself."""
        from Auto3D.utils.validation import check_output_overwrite

        existing = job_dir / "there.sdf"
        existing.write_text("x")

        # Nothing to write; a path that does not exist yet; and an existing
        # path the caller explicitly consented to replace.
        check_output_overwrite(None, False)
        check_output_overwrite(job_dir / "not_created_yet.sdf", False)
        check_output_overwrite(existing, True)

    def test_the_guard_accepts_str_and_path_alike(self, job_dir):
        """Call sites pass both: calc_thermo resolves a `Path`, opt_geometry
        an `os.path.join` string. A guard that only handled one would silently
        no-op for the other."""
        from Auto3D.exceptions import ConfigurationError
        from Auto3D.utils.validation import check_output_overwrite

        existing = job_dir / "there.sdf"
        existing.write_text("x")

        for spelling in (str(existing), existing):
            with pytest.raises(ConfigurationError, match="already exists"):
                check_output_overwrite(spelling, False)

    def test_the_same_file_guard_still_applies_under_force(self, job_dir, monkeypatch):
        """`--force` lifts the overwrite gate, not the same-file gate.

        Overwriting your own earlier output is a recoverable choice; writing
        the filtered result over the input it was computed from is not, so
        `check_output_not_input` must keep refusing regardless.
        """
        import Auto3D.SPE as spe_mod
        from Auto3D.exceptions import ConfigurationError

        sdf = job_dir / "mols.sdf"
        _write_sdf(sdf, ["m1"])
        original = sdf.read_bytes()

        def _never(*args, **kwargs):
            raise AssertionError("model construction reached")

        monkeypatch.setattr(spe_mod, "get_device", _never)
        monkeypatch.setattr(spe_mod, "create_model", _never)

        with pytest.raises(ConfigurationError, match="same file"):
            spe_mod.calc_spe(
                str(sdf), "AIMNET", out_path=str(sdf), use_gpu=False,
                overwrite=True,
            )

        assert sdf.read_bytes() == original
