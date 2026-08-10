"""A thermo failure or a saddle point must not survive conformer selection.

``calc_thermo`` writes ``Thermo_failed``: ``""`` for a genuine minimum,
``"transition_state"`` for a confirmed saddle point, ``"not_converged"`` when
the stationary-point gate refused the record, or an exception class name. The
docs tell readers to filter on it -- and no filter did.

So a saddle point, whose G is a 3N-7-mode quantity computed at a structure with
no thermal population, could be selected as a molecule's "most stable
conformer" and published as one. Its electronic energy can genuinely sit below
another conformer's minimum, so this is not hypothetical.

Absence of the property means "not filtered on it", exactly as for
``Converged`` -- an optimizer output has never carried it and must not be
deleted for that.
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.filtering import DROP_REASONS, filter_conformers
from Auto3D.utils.energy import set_e_tot_from_ev


def _rec(e_ev, thermo_failed=None, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol.SetProp("_Name", "m")
    set_e_tot_from_ev(mol, e_ev)
    mol.SetProp("Converged", "True")
    if thermo_failed is not None:
        mol.SetProp("Thermo_failed", thermo_failed)
    return mol


def test_a_transition_state_is_dropped():
    minimum = _rec(-9.0, thermo_failed="", seed=1)
    saddle = _rec(-20.0, thermo_failed="transition_state", seed=7)

    result = filter_conformers([saddle, minimum], rmsd_threshold=0.3)

    assert minimum in result.kept
    assert saddle not in result.kept, (
        "a confirmed saddle point survived selection, and its lower electronic "
        "energy would have made it the reported 'most stable conformer'"
    )
    assert result.dropped.get("thermo_failed") == 1


def test_a_not_converged_record_is_dropped():
    good = _rec(-9.0, thermo_failed="", seed=1)
    bad = _rec(-20.0, thermo_failed="not_converged", seed=7)

    result = filter_conformers([bad, good], rmsd_threshold=0.3)

    assert good in result.kept and bad not in result.kept


def test_an_empty_marker_is_a_success_and_is_kept():
    good = _rec(-9.0, thermo_failed="")
    assert good in filter_conformers([good], rmsd_threshold=0.3).kept


def test_a_record_without_the_marker_is_not_filtered_on_it():
    """An optimizer output has never carried it; it must not be deleted."""
    plain = _rec(-9.0, thermo_failed=None)
    assert plain in filter_conformers([plain], rmsd_threshold=0.3).kept


def test_the_reason_is_in_the_authoritative_vocabulary():
    assert "thermo_failed" in DROP_REASONS


def test_the_k1_fast_path_drops_it_too(tmp_path):
    """``ConformerRanker``'s k==1 shortcut duplicates the predicate chain."""
    from Auto3D.ranking import ConformerRanker

    saddle = _rec(-20.0, thermo_failed="transition_state", seed=7)
    minimum = _rec(-9.0, thermo_failed="", seed=1)
    path = tmp_path / "in.sdf"
    with Chem.SDWriter(str(path)) as w:
        for m in (saddle, minimum):
            w.write(m)

    out = ConformerRanker(
        input_path=str(path),
        out_path=str(tmp_path / "out.sdf"),
        threshold=0.3,
        k=1,
    ).run()

    assert len(out) == 1
    assert out[0].GetProp("Thermo_failed") == "", "the k==1 fast path selected a saddle point"
