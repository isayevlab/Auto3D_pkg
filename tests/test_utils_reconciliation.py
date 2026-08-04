"""Tests for Auto3D.utils.reconciliation module."""
from pathlib import Path

import pytest  # noqa: F401  (used by the __main__ guard below)
from rdkit import Chem

from Auto3D.utils.reconciliation import find_ids_not_in_sdf, find_smiles_not_in_sdf

# Get the test files directory
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"



def _make_mol(name):
    """Build a tiny named RDKit mol for SDF round-trips."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C")
    mol.SetProp("_Name", name)
    return mol



class TestNoneAndMalformedInputHardening:
    """``None`` SDF records and lenient .smi lines must not crash reconciliation."""

    def test_find_smiles_not_in_sdf_skips_none_records(self, tmp_path, monkeypatch):
        """find_smiles_not_in_sdf must skip None SDF records."""
        from rdkit import Chem

        import Auto3D.utils.reconciliation as reconciliation

        valid = Chem.MolFromSmiles("C")
        valid.SetProp("_Name", "mol_a")
        monkeypatch.setattr(
            reconciliation.Chem, "SDMolSupplier", lambda *a, **k: [valid, None]
        )

        smi = tmp_path / "in.smi"
        smi.write_text("C mol_a\nCC mol_b\n")
        sdf = tmp_path / "out.sdf"
        sdf.write_text("placeholder")

        bad = find_smiles_not_in_sdf(str(smi), str(sdf))
        # mol_a is present (valid mol), mol_b is missing -> reported.
        assert ("mol_b", "CC") in bad
        assert all(mol_id != "mol_a" for mol_id, _ in bad)
    def test_find_smiles_not_in_sdf_tolerates_blank_and_3token_lines(
        self, tmp_path, monkeypatch
    ):
        """find_smiles_not_in_sdf tolerates blank and 3-token .smi lines."""
        from rdkit import Chem

        import Auto3D.utils.reconciliation as reconciliation

        valid = Chem.MolFromSmiles("C")
        valid.SetProp("_Name", "mol_a")
        monkeypatch.setattr(
            reconciliation.Chem, "SDMolSupplier", lambda *a, **k: [valid]
        )

        smi = tmp_path / "in.smi"
        # blank line + a 3-token line (first two tokens taken).
        smi.write_text("C mol_a\n\nCC mol_b extra\n")
        sdf = tmp_path / "out.sdf"
        sdf.write_text("placeholder")

        bad = find_smiles_not_in_sdf(str(smi), str(sdf))
        assert ("mol_b", "CC") in bad


class TestFindSmilesNotInSdfTautStripping:
    """C7: a decoded '@tautN' suffix must not cause a false "missing" report."""

    def test_taut_suffixed_output_name_matches_base_smi_id(self, tmp_path):
        """decode_ids keeps 'id@tautN' on tautomer conformers; the .smi only
        has the base id, so find_smiles_not_in_sdf must strip the suffix
        before comparing or every tautomer-derived molecule is misreported."""
        smi = tmp_path / "in.smi"
        smi.write_text("CCO mol_a\n")

        sdf = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(sdf))
        writer.write(_make_mol("mol_a@taut0"))
        writer.close()

        bad = find_smiles_not_in_sdf(str(smi), str(sdf))
        assert bad == [], f"mol_a wrongly reported missing: {bad}"


class TestFindIdsNotInSdf:
    """find_ids_not_in_sdf: the SDF-input counterpart to find_smiles_not_in_sdf."""

    def test_missing_id_is_reported(self, tmp_path):
        """An id present in the source SDF but absent from the output SDF is reported."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        for name in ["mol_a", "mol_b"]:
            writer.write(_make_mol(name))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(_make_mol("mol_a"))  # mol_b never produced a structure
        writer.close()

        bad = find_ids_not_in_sdf(str(source), str(out))
        assert bad == ["mol_b"]

    def test_no_missing_ids_returns_empty_list(self, tmp_path):
        """Every source id present in the output -> nothing reported."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        for name in ["mol_a", "mol_b"]:
            writer.write(_make_mol(name))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        for name in ["mol_b", "mol_a"]:
            writer.write(_make_mol(name))
        writer.close()

        assert find_ids_not_in_sdf(str(source), str(out)) == []

    def test_taut_suffixed_output_name_matches_base_id(self, tmp_path):
        """Same '@tautN' stripping as find_smiles_not_in_sdf, for SDF input."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        writer.write(_make_mol("mol_a"))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(_make_mol("mol_a@taut1"))
        writer.close()

        assert find_ids_not_in_sdf(str(source), str(out)) == []

    def test_an_unreadable_record_is_reported_on_the_input_side_only(
        self, tmp_path, monkeypatch
    ):
        """The two sides of the comparison are not symmetric, and must not be.

        This test used to assert ``== []`` for a source file containing an
        unreadable record, under the heading "must not crash or miscount". The
        empty list *was* the miscount: an input molecule the pipeline never saw
        was reported by nothing, and the run exited 0 claiming completeness.

        The asymmetry is the point:

        * **input side** -- a record that cannot be read is a molecule the user
          supplied and did not get back. It is reported, by position, since it has
          no ``_Name`` to report by.
        * **output side** -- a record that cannot be read yields no name to match
          against, and there is nothing better to do than skip it. Reporting it
          would invent a *missing input* out of an unreadable output.
        """
        import Auto3D.utils.reconciliation as reconciliation

        calls = {"n": 0}

        def fake_supplier(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return [_make_mol("mol_a"), None]  # source: one good, one unreadable
            return [None, _make_mol("mol_a")]      # output: mol_a made it

        monkeypatch.setattr(reconciliation.Chem, "SDMolSupplier", fake_supplier)

        source = tmp_path / "source.sdf"
        source.write_text("placeholder")
        out = tmp_path / "out.sdf"
        out.write_text("placeholder")

        assert find_ids_not_in_sdf(str(source), str(out)) == [
            reconciliation.UNPARSEABLE_RECORD_ID.format(index=1)
        ]


class TestReconciliationSeesUnparseableInputRecords:
    """A record Auto3D could not read must not vanish from the accounting.

    ``encode_ids`` skips an unparseable SDF record with a warning, so it never
    enters the run. ``find_ids_not_in_sdf`` then built its expected-ID list by
    reading **the same source SDF** and skipping the same record -- so it was in
    neither ``source_ids`` nor the output, could not appear in ``failures``, and
    ``_exit_if_incomplete`` saw ``failed_count == 0``. The run printed a success
    summary and exited **0** having processed fewer molecules than the file
    contained, which is precisely what the C7 reconciliation exists to prevent.

    Only the SDF path is affected. ``encode_ids`` reads ``.smi`` input with
    ``on_malformed="raise"``, so a malformed SMILES line aborts the run with
    ``InputValidationError`` long before reconciliation -- the same blindness
    cannot be reached through that door.
    """

    @staticmethod
    def _unparseable_record(name: str) -> str:
        """A molblock RDKit rejects: the counts line is corrupted.

        Built from a real molblock so only the one line under test is invalid;
        a wholly invented block could fail for an unrelated reason.
        """
        mol = _make_mol(name)
        lines = Chem.MolToMolBlock(mol).splitlines()
        lines[3] = "!! corrupted counts line !!"
        return "\n".join(lines)

    def test_an_unparseable_source_record_is_reported_as_a_failure(self, tmp_path):
        source = tmp_path / "source.sdf"
        good = Chem.MolToMolBlock(_make_mol("mol_a"))
        source.write_text(
            good + "$$$$\n" + self._unparseable_record("mol_b") + "\n$$$$\n"
        )
        # Confirm the premise: RDKit reads one molecule and one None.
        parsed = list(Chem.SDMolSupplier(str(source), removeHs=False))
        assert [m is None for m in parsed] == [False, True], "test premise"

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        writer.write(_make_mol("mol_a"))  # the parseable one succeeded
        writer.close()

        missing = find_ids_not_in_sdf(str(source), str(out))

        assert len(missing) == 1, (
            f"a source record the pipeline could not read was left out of the "
            f"accounting entirely, so the run would exit 0 claiming every input "
            f"was processed; got {missing}"
        )
        assert "1" in missing[0], (
            f"the report must say which record could not be read, got {missing[0]!r}"
        )

    def test_a_clean_source_file_still_reports_nothing(self, tmp_path):
        """The new branch must not manufacture failures for a healthy file."""
        source = tmp_path / "source.sdf"
        writer = Chem.SDWriter(str(source))
        for name in ("mol_a", "mol_b"):
            writer.write(_make_mol(name))
        writer.close()

        out = tmp_path / "out.sdf"
        writer = Chem.SDWriter(str(out))
        for name in ("mol_a", "mol_b"):
            writer.write(_make_mol(name))
        writer.close()

        assert find_ids_not_in_sdf(str(source), str(out)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
