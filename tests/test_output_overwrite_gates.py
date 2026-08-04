#!/usr/bin/env python
"""Overwrite gates on the public writers that derive their own output names.

``Chem.SDWriter(path)`` and ``open(path, "w")`` truncate on open, and four
public functions used to do that with no consent gate at all:

* ``Auto3D.tautomer.select_tautomers`` -- ``select_tautomers("/data/results.sdf",
  k=1)`` replaced ``/data/results_top_tautomers.sdf``, a name it invented, with
  this call's selection.
* ``Auto3D.id_mapping.decode_ids`` -- same shape, for ``<base>_out.sdf``.
* ``Auto3D.utils.smi_io.smiles2smi`` -- the caller named the file here, so the
  gate defaults *open*; what matters is that it can be closed.
* ``Auto3D.id_mapping.encode_ids`` -- refused unconditionally, with no way for a
  caller to say yes.

The policy: permissive where the caller named the file, restrictive where Auto3D
invented the name. Every gate is keyword-only, and every one is asserted in both
directions -- a gate that refused legitimate writes would satisfy the "refuses"
half of each pair and break the pipeline.
"""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.exceptions import ConfigurationError
from Auto3D.id_mapping import decode_ids, encode_ids
from Auto3D.tautomer import select_tautomers
from Auto3D.utils.energy import set_e_tot_from_ev
from Auto3D.utils.smi_io import smiles2smi

PRECIOUS = b"IRREPLACEABLE USER DATA\n"


def _tautomer_sdf(path, names=("mol@taut1", "mol@taut2")) -> str:
    """An SDF shaped the way ``select_tautomers`` expects: ``id@tautN`` names."""
    with Chem.SDWriter(str(path)) as writer:
        for i, name in enumerate(names):
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=42 + i)
            mol.SetProp("_Name", name)
            set_e_tot_from_ev(mol, -10.0 - i)
            writer.write(mol)
    return str(path)


def _encoded_sdf(path) -> str:
    """An SDF shaped the way ``decode_ids`` expects: numeric name + ID prop."""
    with Chem.SDWriter(str(path)) as writer:
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "0")
        mol.SetProp("ID", "0_0_0")
        writer.write(mol)
    return str(path)


class TestSelectTautomers:
    """Auto3D invents ``<stem>_top_tautomers.sdf``, so the gate defaults shut."""

    def test_it_refuses_to_replace_an_existing_top_tautomers_file(self, tmp_path):
        """The concrete hazard from the follow-ups list."""
        sdf = _tautomer_sdf(tmp_path / "results.sdf")
        derived = tmp_path / "results_top_tautomers.sdf"
        derived.write_bytes(PRECIOUS)

        with pytest.raises(ConfigurationError, match="already exists"):
            select_tautomers(sdf, k=1)

        assert derived.read_bytes() == PRECIOUS

    def test_it_writes_when_the_derived_name_is_free(self, tmp_path):
        """Inverse: the ordinary case must still go through untouched.

        Every route Auto3D itself takes lands here -- ``get_stable_tautomers``
        passes ``main()``'s output from a job directory created fresh for that
        run -- so a gate that refused unconditionally would break the whole
        tautomer pipeline while passing the test above.
        """
        sdf = _tautomer_sdf(tmp_path / "results.sdf")
        out = select_tautomers(sdf, k=1)

        assert out == str(tmp_path / "results_top_tautomers.sdf")
        assert len(
            [m for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None]
        ) == 1

    def test_overwrite_true_replaces_it(self, tmp_path):
        """Inverse: the gate is a consent gate the caller can lift."""
        sdf = _tautomer_sdf(tmp_path / "results.sdf")
        derived = tmp_path / "results_top_tautomers.sdf"
        derived.write_bytes(PRECIOUS)

        out = select_tautomers(sdf, k=1, overwrite=True)

        assert out == str(derived)
        assert derived.read_bytes() != PRECIOUS

    def test_overwrite_is_keyword_only(self, tmp_path):
        """``select_tautomers(sdf, k, window)`` is the documented positional
        signature; a fourth positional must not silently become ``overwrite``."""
        sdf = _tautomer_sdf(tmp_path / "results.sdf")
        with pytest.raises(TypeError):
            select_tautomers(sdf, 1, None, True)


class TestDecodeIds:
    """Auto3D invents ``<base>_out.sdf``, so the gate defaults shut."""

    def test_it_refuses_to_replace_an_existing_out_file(self, tmp_path):
        sdf = _encoded_sdf(tmp_path / "mols_3d_encoded.sdf")
        derived = tmp_path / "mols_out.sdf"
        derived.write_bytes(PRECIOUS)

        with pytest.raises(ConfigurationError, match="already exists"):
            decode_ids(sdf, {"mol_a": 0})

        assert derived.read_bytes() == PRECIOUS

    def test_it_writes_when_the_derived_name_is_free(self, tmp_path):
        """Inverse: this is the call ``WorkflowOrchestrator`` makes on every run.

        It writes into the job directory it created with a bare ``mkdir()``
        moments earlier, so the derived name is always free there -- but only if
        the gate permits a free name.
        """
        sdf = _encoded_sdf(tmp_path / "mols_3d_encoded.sdf")
        out = decode_ids(sdf, {"mol_a": 0})

        assert out == str(tmp_path / "mols_out.sdf")
        written = [
            m for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None
        ]
        assert [m.GetProp("_Name") for m in written] == ["mol_a"]

    def test_overwrite_true_replaces_it(self, tmp_path):
        sdf = _encoded_sdf(tmp_path / "mols_3d_encoded.sdf")
        derived = tmp_path / "mols_out.sdf"
        derived.write_bytes(PRECIOUS)

        out = decode_ids(sdf, {"mol_a": 0}, overwrite=True)

        assert out == str(derived)
        assert derived.read_bytes() != PRECIOUS

    def test_overwrite_is_keyword_only(self, tmp_path):
        sdf = _encoded_sdf(tmp_path / "mols_3d_encoded.sdf")
        with pytest.raises(TypeError):
            decode_ids(sdf, {"mol_a": 0}, True)


class TestEncodeIds:
    """Already refused unconditionally; now it refuses *consistently*.

    Same keyword, same default, same exception and same message as the other
    three -- and, unlike before, a caller who means it can say so.
    """

    def test_it_still_refuses_by_default(self, tmp_path):
        smi = tmp_path / "mols.smi"
        smi.write_text("CCO a\n")
        derived = tmp_path / "mols_encoded.smi"
        derived.write_bytes(PRECIOUS)

        with pytest.raises(ConfigurationError, match="already exists"):
            encode_ids(str(smi))

        assert derived.read_bytes() == PRECIOUS

    def test_overwrite_true_replaces_it(self, tmp_path):
        """The half that was impossible before: there was no way to consent."""
        smi = tmp_path / "mols.smi"
        smi.write_text("CCO a\n")
        derived = tmp_path / "mols_encoded.smi"
        derived.write_bytes(PRECIOUS)

        new_path, mapping = encode_ids(str(smi), overwrite=True)

        assert new_path == str(derived)
        assert mapping == {"a": 0}
        assert derived.read_text() == "CCO 0\n"

    def test_it_writes_when_the_derived_name_is_free(self, tmp_path):
        """Inverse: the pipeline's own call, which must keep working."""
        smi = tmp_path / "mols.smi"
        smi.write_text("CCO a\nCCC b\n")
        new_path, mapping = encode_ids(str(smi))
        assert mapping == {"a": 0, "b": 1}
        assert new_path == str(tmp_path / "mols_encoded.smi")

    def test_overwrite_is_keyword_only(self, tmp_path):
        smi = tmp_path / "mols.smi"
        smi.write_text("CCO a\n")
        with pytest.raises(TypeError):
            encode_ids(str(smi), None, True)


class TestSmiles2Smi:
    """The caller named this file, so the gate defaults OPEN.

    ``smiles2mols`` writes it into a ``TemporaryDirectory`` on every call, and a
    caller who passes an explicit path has already chosen it. Defaulting shut
    here would be a gate on a decision the caller already made.
    """

    def test_the_default_still_overwrites(self, tmp_path):
        """Inverse first, because this is the back-compat guarantee."""
        out = tmp_path / "mols.smi"
        out.write_bytes(PRECIOUS)

        assert smiles2smi(["CCO"], str(out)) == str(out)
        assert out.read_bytes() != PRECIOUS
        assert out.read_text().startswith("CCO  ")

    def test_overwrite_false_refuses(self, tmp_path):
        out = tmp_path / "mols.smi"
        out.write_bytes(PRECIOUS)

        with pytest.raises(ConfigurationError, match="already exists"):
            smiles2smi(["CCO"], str(out), overwrite=False)

        assert out.read_bytes() == PRECIOUS

    def test_overwrite_false_still_writes_a_free_path(self, tmp_path):
        out = tmp_path / "mols.smi"
        assert smiles2smi(["CCO"], str(out), overwrite=False) == str(out)
        assert out.read_text().startswith("CCO  ")

    def test_overwrite_is_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            smiles2smi(["CCO"], str(tmp_path / "mols.smi"), False)


class TestTheGateRefusesBeforeDoingTheWork:
    """A gate applied just before the write is a gate that wasted the run.

    ``select_tautomers`` reads and groups the whole input before it opens the
    writer; refusing only there means the user waits for the work, then loses
    it. The check must come first.
    """

    def test_select_tautomers_checks_before_reading_the_input(
        self, tmp_path, monkeypatch
    ):
        import Auto3D.tautomer as tautomer

        sdf = _tautomer_sdf(tmp_path / "results.sdf")
        (tmp_path / "results_top_tautomers.sdf").write_bytes(PRECIOUS)

        def _never(*args, **kwargs):
            raise AssertionError(
                "select_tautomers read its input before checking the output path"
            )

        monkeypatch.setattr(tautomer.Chem, "SDMolSupplier", _never)

        with pytest.raises(ConfigurationError, match="already exists"):
            select_tautomers(sdf, k=1)
