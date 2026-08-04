"""The one shared in-place-rewrite helper, and the three call sites using it.

Three functions in Auto3D rewrite a file they have just read -- ``reorder_sdf``,
``ASE.geometry._annotate_and_rewrite`` and
``utils.stereochemistry.amend_configuration_w`` -- and each grew its own
staging code. Two of them (the ``mkstemp`` pair) copied the target's permission
bits across; ``reorder_sdf`` did not, and it also used a predictable
``<name>.reorder.tmp`` filename. So a 0600 SDF came back 0644 after a reorder:
a permission *loosening*, on exactly the path an ordinary
``auto3d run mols.smi`` takes. That is the divergence
:func:`Auto3D.utils.atomic_io.atomic_write_path` exists to remove.

The table below is the contract, asserted once for the helper and once per call
site, so a site that stops using the helper fails here rather than silently
regressing to its own staging:

* the temp file is a **sibling** of the target -- ``os.replace`` is only atomic
  within one filesystem and raises ``EXDEV`` across them;
* the target's **mode is preserved** -- neither tightened (``mkstemp``'s 0600)
  nor loosened (the process umask);
* **no temp file is left behind** when the body raises;
* the **original file is intact** when the body raises.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.utils.atomic_io import atomic_write_path

# Captured before any test patches Chem.SDWriter (see tests/test_durability.py).
_real_sdwriter = Chem.SDWriter


def _write_sdf(path, names):
    """Write one embedded ethanol conformer per name."""
    with _real_sdwriter(str(path)) as w:
        for name in names:
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol.SetProp("_Name", name)
            mol.SetProp("ID", name)
            w.write(mol)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(str(path)).st_mode)


class TestAtomicWritePathHelper:
    """The contract, on the helper itself."""

    def test_yields_a_sibling_path(self, tmp_path):
        target = tmp_path / "out.sdf"
        target.write_text("original\n")

        with atomic_write_path(str(target), suffix=".sdf") as tmp:
            assert Path(tmp).parent == Path(os.path.realpath(str(tmp_path))), (
                f"temp file {tmp} is not beside its target {target}; "
                "os.replace would raise EXDEV whenever the two differ"
            )
            Path(tmp).write_text("new\n")

        assert target.read_text() == "new\n"

    def test_parent_is_resolved_with_realpath_not_abspath(self, tmp_path):
        """``abspath`` collapses ``..`` lexically and can pick the wrong mount.

        ``/scratch/link/../out.sdf`` where ``link`` points at another
        filesystem: ``abspath`` says ``/scratch``, the replace destination is
        somewhere else, and ``os.replace`` fails with EXDEV *after* a completed
        run. Only the parent is resolved -- ``os.replace`` acts on the final
        component itself, so following a symlinked target would pick the wrong
        directory.
        """
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir, target_is_directory=True)

        target = link / ".." / "link" / "out.sdf"
        Path(real_dir / "out.sdf").write_text("original\n")

        with atomic_write_path(str(target)) as tmp:
            assert Path(tmp).parent == Path(os.path.realpath(str(real_dir)))
            Path(tmp).write_text("new\n")

        assert (real_dir / "out.sdf").read_text() == "new\n"

    @pytest.mark.parametrize("mode", [0o600, 0o644, 0o640])
    def test_target_mode_is_preserved(self, tmp_path, mode):
        """Neither tightened to mkstemp's 0600 nor loosened to the umask."""
        target = tmp_path / "out.sdf"
        target.write_text("original\n")
        os.chmod(target, mode)

        with atomic_write_path(str(target)) as tmp:
            assert _mode(tmp) == mode, (
                "the staged temp file does not carry the target's mode, so "
                "os.replace will change it"
            )
            Path(tmp).write_text("new\n")

        assert _mode(target) == mode

    def test_missing_target_still_works(self, tmp_path):
        """A target that does not exist yet has no mode to copy; do not fail."""
        target = tmp_path / "new.sdf"

        with atomic_write_path(str(target)) as tmp:
            Path(tmp).write_text("new\n")

        assert target.read_text() == "new\n"

    def test_exception_leaves_no_temp_file_and_the_original_intact(self, tmp_path):
        target = tmp_path / "out.sdf"
        target.write_text("original\n")

        with pytest.raises(RuntimeError, match="disk full"):
            with atomic_write_path(str(target)) as tmp:
                Path(tmp).write_text("half a file")
                raise RuntimeError("disk full")

        assert target.read_text() == "original\n", "the original was corrupted"
        leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != "out.sdf")
        assert not leftovers, f"temp files left behind: {leftovers}"

    def test_base_exception_also_cleans_up(self, tmp_path):
        """KeyboardInterrupt mid-write must not leave a stray file either."""
        target = tmp_path / "out.sdf"
        target.write_text("original\n")

        with pytest.raises(KeyboardInterrupt):
            with atomic_write_path(str(target)) as tmp:
                Path(tmp).write_text("half a file")
                raise KeyboardInterrupt

        assert target.read_text() == "original\n"
        leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != "out.sdf")
        assert not leftovers, f"temp files left behind: {leftovers}"


class TestReorderSdfPreservesMode:
    """``reorder_sdf`` is the call site the shared helper was written for.

    It staged through a predictable ``<name>.reorder.tmp`` and never copied the
    target's mode, so a 0600 input came back 0644 -- the process umask's idea of
    a new file, applied to a file the user had deliberately restricted. This is
    the one entry in the tripwire table that failed before the helper existed.
    """

    @pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
    def test_target_mode_survives_a_reorder(self, tmp_path, mode):
        from Auto3D.utils.sdf_io import reorder_sdf

        sdf = tmp_path / "out.sdf"
        smi = tmp_path / "in.smi"
        _write_sdf(sdf, ["a", "b"])
        smi.write_text("CCO b\nCCO a\n")
        os.chmod(sdf, mode)

        reorder_sdf(str(sdf), str(smi))

        assert _mode(sdf) == mode, (
            f"reorder_sdf changed the output file's mode from {mode:04o} to "
            f"{_mode(sdf):04o}"
        )

    def test_the_reordering_still_happens(self, tmp_path):
        """Positive control: the mode test above must not pass vacuously."""
        from Auto3D.utils.sdf_io import reorder_sdf

        sdf = tmp_path / "out.sdf"
        smi = tmp_path / "in.smi"
        _write_sdf(sdf, ["a", "b"])
        smi.write_text("CCO b\nCCO a\n")

        mols = reorder_sdf(str(sdf), str(smi))
        assert [m.GetProp("_Name") for m in mols] == ["b", "a"]
        on_disk = [
            m.GetProp("_Name")
            for m in Chem.SDMolSupplier(str(sdf), removeHs=False)
            if m is not None
        ]
        assert on_disk == ["b", "a"]

    def test_no_predictable_temp_name_is_used(self, tmp_path, monkeypatch):
        """The staged name must come from ``mkstemp``, not from the target's.

        A predictable ``<name>.reorder.tmp`` is guessable by any other process
        sharing the directory, and two concurrent reorders of the same file
        would stage onto each other's temp path.
        """
        from Auto3D.utils.sdf_io import reorder_sdf

        sdf = tmp_path / "out.sdf"
        smi = tmp_path / "in.smi"
        _write_sdf(sdf, ["a"])
        smi.write_text("CCO a\n")

        seen: list[str] = []
        real = Chem.SDWriter

        def spy(path, *a, **k):
            seen.append(str(path))
            return real(path, *a, **k)

        monkeypatch.setattr(Chem, "SDWriter", spy)
        reorder_sdf(str(sdf), str(smi))

        assert seen, "reorder_sdf opened no writer"
        assert not any(p.endswith(".reorder.tmp") for p in seen), (
            f"reorder_sdf still stages through a predictable name: {seen}"
        )


class TestAmendConfigurationPreservesMode:
    """The ``.smi`` call site keeps the property it already had."""

    @pytest.mark.parametrize("mode", [0o600, 0o644])
    def test_target_mode_survives(self, tmp_path, mode):
        from Auto3D.utils.stereochemistry import amend_configuration_w

        smi = tmp_path / "enum.smi"
        smi.write_text("CC[C@H](N)O mol_a_0\n")
        os.chmod(smi, mode)

        amend_configuration_w(str(smi))

        assert _mode(smi) == mode


class TestAnnotateAndRewritePreservesMode:
    """The ``opt_geometry`` rewrite pass keeps the property it already had."""

    @pytest.mark.parametrize("mode", [0o600, 0o644])
    def test_target_mode_survives(self, tmp_path, mode):
        from Auto3D.ASE.geometry import _annotate_and_rewrite

        out = tmp_path / "opt.sdf"
        with _real_sdwriter(str(out)) as w:
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol.SetProp("_Name", "a")
            mol.SetProp("E_tot", "-1.5")
            w.write(mol)
        os.chmod(out, mode)

        _annotate_and_rewrite(str(out))

        assert _mode(out) == mode
        rewritten = [
            m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m is not None
        ]
        assert len(rewritten) == 1
        assert rewritten[0].GetProp("E_tot(Hartree)") == "-1.5"
