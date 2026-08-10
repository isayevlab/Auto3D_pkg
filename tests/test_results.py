# tests/test_results.py
"""WorkflowResult is a backwards-compatible str carrying lazy run counts."""

from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.results import WorkflowResult, count_output


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


def test_count_output_missing_file_is_zero(tmp_path):
    assert count_output(str(tmp_path / "nope.sdf")) == (0, 0)
    assert count_output("") == (0, 0)
