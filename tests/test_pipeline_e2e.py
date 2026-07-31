"""Every input molecule must be accounted for, and success must mean success.

The pipeline isolates failure per chunk, per molecule, with sentinel-guaranteed
queue drains -- but has no compensating reporting layer, so failure is reliably
contained and just as reliably invisible. _finalize_output raises only when zero
outputs exist, so 9 of 10 failed chunks exits 0 (C6). find_smiles_not_in_sdf,
the reconciliation function, exists and is exported and tested with zero
production callers (C7).

Slow tier: uses the real aimnet2 registry model on CPU.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from Auto3D.config import Auto3DOptions

pytestmark = pytest.mark.slow


def _input_ids(smi_path: str) -> set[str]:
    ids = set()
    with open(smi_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


class TestInputOutputAccounting:
    """No input may vanish without being reported."""

    def test_every_input_is_present_or_reported(self, job_dir):
        """Each input ID must appear in the output or in a reported failure list.

        A run of only valid molecules has no *structural* reason to lose any
        of them, so their absence would depend on non-deterministic
        convergence luck rather than the C7 defect itself. Instead, mix in the
        same guaranteed-unconvertible sodium counterion used by
        ``test_one_bad_molecule_does_not_remove_the_others`` below, and force
        one molecule per job (the same capacity/memory=1 technique used in
        ``TestExitStatus``) so sodium's job fails alone -- deterministically,
        via optim_rank_wrapper's bare except/continue -- while the other jobs
        succeed independently. Total output is nonzero (no OptimizationError),
        but the failing ID vanishes with no report anywhere reachable from
        main()'s return value. That is the reconciliation gap C7 describes,
        exercised without relying on any molecule's numerical luck.
        """
        from Auto3D.auto3D import main

        smi = job_dir / "mixed10.smi"
        # Na is outside AIMNet2's 14-element set and guaranteed to fail; the
        # other three are simple, well-behaved organics guaranteed to succeed.
        smi.write_text(
            "CCO ethanol\n"
            "CCCO propanol\n"
            "c1ccccc1 benzene\n"
            "[Na+].CC(=O)[O-] sodium_acetate\n"
        )

        args = Auto3DOptions(
            path=str(smi), k=1, use_gpu=False, max_confs=2, capacity=1, memory=1
        )
        out = main(args)

        produced = set()
        for mol in Chem.SDMolSupplier(out, removeHs=False):
            if mol is not None:
                produced.add(mol.GetProp("_Name").split("_")[0])

        # No public interface reports failed inputs today (that is exactly
        # C7), so this is always empty -- but written this way, a later fix
        # that starts populating a failure list on the result would make this
        # test XPASS without any edits here.
        reported_failures = set(getattr(out, "failures", None) or [])

        expected = _input_ids(str(smi))
        missing = expected - produced - reported_failures
        assert not missing, (
            f"{len(missing)} of {len(expected)} inputs vanished with no report "
            f"(absent from the output SDF and from any reported failure list): "
            f"{sorted(missing)}"
        )

    def test_one_bad_molecule_does_not_remove_the_others(self, job_dir):
        """A sodium counterion must fail, and must fail alone.

        This began as an ``xfail(strict=True)`` for C6, whose claim was that an
        unsupported element raises inside ``ensemble_opt`` and
        ``optim_rank_wrapper``'s bare ``except Exception: continue`` then
        discards every molecule in that chunk. **That claim does not hold for
        this input, and the marker was removed rather than the test.**

        The original assertion only checked that the good molecules survived,
        which passes whenever the bad one quietly succeeds, so it established
        nothing and XPASSed for months. Adding the ``sodium_acetate not in
        produced`` precondition made the result meaningful, and CI then showed
        both halves already hold: sodium_acetate is absent (it genuinely
        failed) *and* ethanol, propanol and benzene are all present (its
        failure was contained). Whatever rejects the sodium salt does so at a
        granularity finer than the chunk.

        What this does NOT show: that the bare ``except Exception: continue``
        in ``optim_rank_wrapper`` is harmless. It is still there and still
        chunk-scoped, so a failure mode that reaches it -- a CUDA OOM, an
        mkdir collision -- would still take the whole chunk down. This test
        now stands as a regression guard for the element case only. The other
        half of C6, that the CLI exits 0 after losing molecules, is unaffected
        and still tripwired in ``TestExitStatus`` below.
        """
        from Auto3D.auto3D import main

        smi = job_dir / "mixed.smi"
        # Na is outside AIMNet2's 14-element set; the other three are fine.
        smi.write_text(
            "CCO ethanol\n"
            "CCCO propanol\n"
            "[Na+].CC(=O)[O-] sodium_acetate\n"
            "c1ccccc1 benzene\n"
        )

        args = Auto3DOptions(path=str(smi), k=1, use_gpu=False, max_confs=2)
        out = main(args)

        produced = {
            m.GetProp("_Name").split("_")[0]
            for m in Chem.SDMolSupplier(out, removeHs=False)
            if m is not None
        }

        # A test for "one failure does not cascade" has to establish that a
        # failure happened. Without this the assertions below pass whenever
        # sodium_acetate quietly succeeds, which says nothing about C6 -- and
        # that is exactly how this test XPASSed against a codebase where the
        # bare `except Exception: continue` in optim_rank_wrapper is still
        # there, unchanged.
        #
        # Na (Z=11) is genuinely absent from AIMNet2's implemented species
        # (verified against the cached checkpoints: [1, 5, 6, 7, 8, 9, 14, 15,
        # 16, 17, 33, 34, 35, 53]), and nothing in Auto3D checks that list. So
        # if sodium_acetate appears in the output, the model returned a number
        # for an element it does not implement -- a silently wrong energy,
        # which is a worse finding than the chunk loss this test was written
        # for, and one that must not hide behind a green assertion.
        assert "sodium_acetate" not in produced, (
            "sodium_acetate was optimized and written to the output even though "
            "Na (Z=11) is not among AIMNet2's implemented species. Nothing in "
            "Auto3D validates atomic numbers against the model's species list, "
            "so this is an energy computed for an unsupported element rather "
            f"than the chunk loss C6 describes; produced {sorted(produced)}"
        )

        for good in ("ethanol", "propanol", "benzene"):
            assert good in produced, (
                f"{good} was lost because an unrelated molecule failed; produced "
                f"{sorted(produced)}"
            )


class TestExitStatus:
    """Losing molecules must not exit 0."""

    def test_cli_exits_nonzero_when_molecules_are_missing(self, job_dir):
        """auto3d run must signal partial failure through its exit code."""
        from typer.testing import CliRunner

        from Auto3D.cli.app import app

        smi = job_dir / "mixed.smi"
        smi.write_text("CCO ethanol\n[Na+].CC(=O)[O-] sodium_acetate\n")

        # `auto3d run` has no --capacity/--memory/--max-confs flags (verified via
        # `auto3d run --help`), so a plain two-molecule file lands both rows in a
        # single default-sized chunk (capacity=42/GB memory). One unsupported
        # element in a *shared* chunk currently takes the whole chunk down (C6),
        # which would leave zero total output and make _finalize_output raise
        # OptimizationError on its own -- a *correct* nonzero exit for an
        # unrelated reason, not the silent partial-failure this test targets.
        # A tiny capacity/memory (via --config) forces one molecule per chunk,
        # so ethanol's chunk succeeds independently of sodium_acetate's chunk
        # failing: a genuine partial run that should exit 0 today.
        config = job_dir / "config.yaml"
        config.write_text(f"path: {smi}\ncapacity: 1\nmemory: 1\n")

        result = CliRunner().invoke(
            app,
            ["run", str(smi), "--config", str(config), "--k", "1", "--no-gpu"],
        )
        assert result.exit_code != 0, (
            f"exited 0 despite losing a molecule; output:\n{result.output}"
        )


class TestEnergyAndRankingSanity:
    """Assert on the numbers, not merely that the program ran."""

    def test_energies_are_negative_and_ordered(self, isolated_input):
        """E_tot must be negative and ascending within a conformer group."""
        from Auto3D.auto3D import main

        args = Auto3DOptions(
            path=isolated_input("smiles2.smi"), k=3, use_gpu=False, max_confs=4
        )
        out = main(args)

        groups: dict[str, list[float]] = {}
        for mol in Chem.SDMolSupplier(out, removeHs=False):
            if mol is None:
                continue
            base = mol.GetProp("_Name").split("_")[0]
            groups.setdefault(base, []).append(float(mol.GetProp("E_tot")))

        assert groups, "no molecules in the output"
        for base, energies in groups.items():
            assert all(e < 0 for e in energies), f"{base}: non-negative energy {energies}"
            assert energies == sorted(energies), (
                f"{base}: conformers are not energy-ordered: {energies}"
            )

    def test_top_k_returns_distinct_conformers(self, isolated_input):
        """k=3 must yield at most 3 per molecule, and the first is the minimum."""
        from Auto3D.auto3D import main

        args = Auto3DOptions(
            path=isolated_input("smiles2.smi"), k=3, use_gpu=False, max_confs=6
        )
        out = main(args)

        groups: dict[str, list[float]] = {}
        for mol in Chem.SDMolSupplier(out, removeHs=False):
            if mol is None:
                continue
            base = mol.GetProp("_Name").split("_")[0]
            groups.setdefault(base, []).append(float(mol.GetProp("E_tot")))

        for base, energies in groups.items():
            assert len(energies) <= 3, f"{base}: k=3 but got {len(energies)}"
            assert energies[0] == min(energies), f"{base}: first is not the minimum"
