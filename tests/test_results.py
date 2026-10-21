# tests/test_results.py
"""WorkflowResult is a backwards-compatible str carrying lazy run counts."""

from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.foundation.results import WorkflowResult, count_output


def _write_sdf(path, names):
    with Chem.SDWriter(str(path)) as w:
        for name in names:
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", name)
            w.write(m)


def test_workflow_result_is_str_compatible(tmp_path):
    p = tmp_path / "out.sdf"
    r = WorkflowResult(str(p))
    assert isinstance(r, str)
    assert os.fspath(r) == str(p)
    assert Path(r) == p
    assert r == str(p)


def test_workflow_result_counts(tmp_path):
    p = tmp_path / "out.sdf"
    # two conformers of molecule "a" (a@taut0/@taut1 collapse to one molecule)
    # and one of molecule "b" -> 2 molecules, 3 conformers.
    _write_sdf(p, ["a@taut0", "a@taut1", "b"])
    r = WorkflowResult(str(p))
    assert r.n_molecules == 2
    assert r.n_conformers == 3


def test_eager_counts_survive_the_file_being_moved(tmp_path):
    """When the caller already knows the counts (the orchestrator
    computes them from the record list `reorder_sdf` returns), they must not
    be recomputed lazily from the output path -- which zeros them out the
    moment that path is moved or deleted, as `count_output`'s own "missing
    file -> (0, 0)" fallback does."""
    path = tmp_path / "out.sdf"
    _write_sdf(path, ["a", "a", "b"])  # 3 records, 2 names
    result = WorkflowResult(str(path), n_molecules=2, n_conformers=3)
    path.rename(tmp_path / "elsewhere.sdf")
    assert (result.n_molecules, result.n_conformers) == (2, 3)


def test_lazy_counts_are_still_the_fallback_when_eager_counts_are_omitted(tmp_path):
    """Every existing caller that builds a WorkflowResult from just a path
    (and failures) must keep getting the lazy, computed-from-disk count."""
    p = tmp_path / "out.sdf"
    _write_sdf(p, ["a@taut0", "a@taut1", "b"])
    r = WorkflowResult(str(p))
    assert r.n_molecules == 2
    assert r.n_conformers == 3


def test_lazy_count_strips_whitespace_before_the_taut_suffix(tmp_path):
    """Fix round 2: Chem-M5's end-anchored `strip_taut_suffix` pattern
    (`@taut\\d+(?:_\\d+_\\d+)?$`) no longer matches when whitespace follows
    the suffix, so `strip_taut_suffix(name).strip()` -- strip AFTER the
    regex -- leaves a trailing-whitespace `_Name` like `"mol@taut2 "`
    un-stripped of its `@tautN` suffix entirely. Strip whitespace FIRST
    (the order `id_mapping.decode_ids` already uses), so it still counts
    under its base id `"mol"`."""
    p = tmp_path / "out.sdf"
    _write_sdf(p, ["mol@taut2 ", "mol@taut3", "other"])
    r = WorkflowResult(str(p))
    assert r.n_molecules == 2
    assert r.n_conformers == 3


def test_count_output_missing_file_is_zero(tmp_path):
    assert count_output(str(tmp_path / "nope.sdf")) == (0, 0)
    assert count_output("") == (0, 0)


def test_count_output_preserves_bare_at_in_ids(tmp_path):
    """A legitimate id containing a bare "@" (not the "@tautN" separator)
    must not be truncated by molecule-identity grouping.

    Regression for the drift bug where this module split on bare "@" instead
    of "@taut": three distinct ids "cmpd@1", "cmpd@2", "plain" collapsed to
    two molecules ("cmpd" x2, "plain") instead of three, because "@1"/"@2"
    were mistaken for a tautomer suffix. count_output must now agree with
    strip_taut_suffix's actual separator and report (3, 3).
    """
    p = tmp_path / "out.sdf"
    _write_sdf(p, ["cmpd@1", "cmpd@2", "plain"])
    r = WorkflowResult(str(p))
    assert r.n_molecules == 3
    assert r.n_conformers == 3
    assert count_output(str(p)) == (3, 3)
