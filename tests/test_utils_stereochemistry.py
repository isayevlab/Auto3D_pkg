"""Tests for Auto3D.foundation.utils.stereochemistry module.

This module tests stereochemistry-related utility functions including
enantiomer detection, stereo info extraction, and configuration amendment.
"""

import os
import tempfile

import pytest
from rdkit import Chem

from Auto3D.foundation.exceptions import InputValidationError
from Auto3D.foundation.utils.stereochemistry import (
    amend_configuration,
    amend_configuration_w,
    are_enantiomers,
    check_value,
    create_enantiomer,
    enantiomer,
    enantiomer_helper,
    get_stereo_info,
    no_enantiomer,
    no_enantiomer_helper,
    remove_enantiomers,
)


class TestEnantiomer:
    """Tests for the enantiomer() function."""

    def test_enantiomer_true(self):
        """Test that opposite stereo configurations are detected as enantiomers."""
        # (R, R) vs (S, S) - all inverted -> enantiomers
        l1 = [(1, "R"), (5, "R")]
        l2 = [(1, "S"), (5, "S")]
        assert enantiomer(l1, l2) is True

    def test_enantiomer_false_same(self):
        """Test that same configurations are not enantiomers."""
        # (R, R) vs (R, R) - same -> not enantiomers
        l1 = [(1, "R"), (5, "R")]
        l2 = [(1, "R"), (5, "R")]
        assert enantiomer(l1, l2) is False

    def test_enantiomer_false_partial(self):
        """Test that partially inverted stereo is not enantiomeric."""
        # (R, R) vs (S, R) - only one inverted -> not enantiomers
        l1 = [(1, "R"), (5, "R")]
        l2 = [(1, "S"), (5, "R")]
        assert enantiomer(l1, l2) is False

    def test_enantiomer_single_center(self):
        """Test enantiomer detection with single stereo center."""
        l1 = [(3, "R")]
        l2 = [(3, "S")]
        assert enantiomer(l1, l2) is True

    def test_enantiomer_mismatched_length_raises(self):
        """Test that mismatched list lengths raise ValueError."""
        l1 = [(1, "R"), (5, "R")]
        l2 = [(1, "S")]
        with pytest.raises(ValueError, match="length"):
            enantiomer(l1, l2)


class TestEnantiomerHelper:
    """Tests for the enantiomer_helper() function."""

    def test_enantiomer_helper_filters_pair(self):
        """Test that enantiomeric pairs are filtered to one representative."""
        # Create a pair of enantiomers
        smiles = ["C[C@H](O)F", "C[C@@H](O)F"]
        result = enantiomer_helper(smiles)
        # Should keep only one
        assert len(result) == 1
        assert result[0] in smiles

    def test_enantiomer_helper_keeps_non_chiral(self):
        """Two distinct achiral molecules must both survive enantiomer filtering.

        CCO and CCCO are different compounds. The current implementation drops
        the second because both have empty stereo-center lists and
        ``enantiomer([], [])`` returns True vacuously.
        """
        smiles = ["CCO", "CCCO"]
        result = enantiomer_helper(smiles)
        assert len(result) == 2, f"a distinct achiral molecule was dropped: {result}"

    def test_enantiomer_helper_empty_list(self):
        """Test handling of empty list."""
        result = enantiomer_helper([])
        assert result == []


class TestGetStereoInfo:
    """Tests for the get_stereo_info() function."""

    def test_single_at(self):
        """Test detection of single @ symbol."""
        smi = "C[C@H](O)F"
        result = get_stereo_info(smi)
        assert len(result) == 1
        assert "@" in result.values()

    def test_double_at(self):
        """Test detection of @@ symbol."""
        smi = "C[C@@H](O)F"
        result = get_stereo_info(smi)
        assert len(result) == 1
        assert "@@" in result.values()

    def test_multiple_stereo_centers(self):
        """Test detection of multiple stereo centers."""
        smi = "C[C@H](O)[C@@H](F)Cl"
        result = get_stereo_info(smi)
        assert len(result) == 2
        values = list(result.values())
        assert "@" in values
        assert "@@" in values

    def test_no_stereo(self):
        """Test SMILES with no stereochemistry."""
        smi = "CCO"
        result = get_stereo_info(smi)
        assert len(result) == 0

    def test_ordered_dict_sorted(self):
        """Test that result is sorted by position."""
        smi = "C[C@@H](O)[C@H](F)Cl"
        result = get_stereo_info(smi)
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestNoEnantiomerHelper:
    """Tests for the no_enantiomer_helper() function."""

    def test_all_different_is_enantiomer(self):
        """Test that all-different stereo symbols indicate enantiomers."""
        info1 = ["@", "@@"]
        info2 = ["@@", "@"]
        assert no_enantiomer_helper(info1, info2) is True

    def test_some_same_not_enantiomer(self):
        """Test that matching symbols indicate non-enantiomers."""
        info1 = ["@", "@@"]
        info2 = ["@", "@"]
        assert no_enantiomer_helper(info1, info2) is False

    def test_all_same_not_enantiomer(self):
        """Test identical stereo info is not enantiomeric."""
        info1 = ["@", "@@"]
        info2 = ["@", "@@"]
        assert no_enantiomer_helper(info1, info2) is False


class TestNoEnantiomer:
    """Tests for the no_enantiomer() function."""

    def test_no_enantiomer_present(self):
        """Test when no enantiomer exists in list.

        Note: All molecules in the list must have the same number of
        stereo centers for comparison to work properly.
        """
        smi = "C[C@H](O)F"
        # Use molecules with same stereo center count but different configuration patterns
        smiles = ["C[C@H](O)F", "C[C@H](Cl)Br", "C[C@H](N)O"]
        assert no_enantiomer(smi, smiles) is True

    def test_enantiomer_present(self):
        """Test when enantiomer exists in list."""
        smi = "C[C@H](O)F"
        smiles = ["C[C@H](O)F", "C[C@@H](O)F"]
        assert no_enantiomer(smi, smiles) is False


class TestCreateEnantiomer:
    """Tests for the create_enantiomer() function."""

    def test_single_at_to_double(self):
        """Test conversion of @ to @@."""
        smi = "C[C@H](O)F"
        result = create_enantiomer(smi)
        assert "@@" in result
        assert result != smi

    def test_double_at_to_single(self):
        """Test conversion of @@ to @."""
        smi = "C[C@@H](O)F"
        result = create_enantiomer(smi)
        # The result should have @ but not @@
        info = get_stereo_info(result)
        assert "@" in info.values()

    def test_multiple_centers_all_inverted(self):
        """Test that all stereo centers are inverted."""
        smi = "C[C@H](O)[C@@H](F)Cl"
        result = create_enantiomer(smi)
        # Original has @ then @@, result should have @@ then @
        orig_info = list(get_stereo_info(smi).values())
        result_info = list(get_stereo_info(result).values())
        for orig, res in zip(orig_info, result_info):
            assert orig != res

    def test_three_centers_all_inverted(self):
        """M60 regression: 3+ centers used to be handled by a loop reading a
        variable (key2) set inside the loop's else-branch and read again
        after the loop -- correct only by Python's lack of block scoping.
        A single-pass rewrite must still invert every center in order."""
        smi = "C[C@H](O)[C@@H](F)[C@H](Cl)Br"
        result = create_enantiomer(smi)
        assert result == "C[C@@H](O)[C@H](F)[C@@H](Cl)Br"
        orig_info = list(get_stereo_info(smi).values())
        result_info = list(get_stereo_info(result).values())
        assert len(result_info) == len(orig_info) == 3
        for orig, res in zip(orig_info, result_info):
            assert orig != res

    def test_four_centers_all_inverted(self):
        """Same as above, one more center, to rule out an off-by-one at the
        boundary between the removed len(keys)==1 special case and the
        general multi-key path."""
        smi = "C[C@H](O)[C@@H](F)[C@H](Cl)[C@@H](Br)I"
        result = create_enantiomer(smi)
        assert result == "C[C@@H](O)[C@H](F)[C@@H](Cl)[C@H](Br)I"
        result_info = list(get_stereo_info(result).values())
        assert len(result_info) == 4


class TestCheckValue:
    """Tests for the check_value() function."""

    def test_powers_of_2(self):
        """Test that powers of 2 return True."""
        assert check_value(1) is True
        assert check_value(2) is True
        assert check_value(4) is True
        assert check_value(8) is True
        assert check_value(16) is True

    def test_non_powers_of_2(self):
        """Test that non-powers of 2 return False."""
        assert check_value(3) is False
        assert check_value(5) is False
        assert check_value(6) is False
        assert check_value(7) is False

    def test_fractional_powers(self):
        """Test fractional powers of 2."""
        assert check_value(0.5) is True  # 2^-1
        assert check_value(0.25) is True  # 2^-2


class TestRemoveEnantiomers:
    """Tests for the remove_enantiomers() function."""

    def test_remove_enantiomers_basic(self):
        """Test basic enantiomer removal."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as infile:
            infile.write("C[C@H](O)F mol_1\n")
            infile.write("C[C@@H](O)F mol_2\n")
            inpath = infile.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as outfile:
            outpath = outfile.name

        try:
            result = remove_enantiomers(inpath, outpath)

            # "mol_1"/"mol_2" collapse to the single group "mol" (the
            # trailing isomer index is stripped), and the two input SMILES
            # are a genuine enantiomer pair, so exactly one must survive.
            # `"mol" in result` alone, or `len(lines) >= 1`, would still pass
            # if the enantiomer filter never removed anything at all.
            assert set(result.keys()) == {"mol"}
            assert len(result["mol"]) == 1, f"a genuine enantiomer pair survived: {result['mol']}"

            # Check output file
            with open(outpath) as f:
                lines = f.readlines()
            assert len(lines) == 1
        finally:
            os.unlink(inpath)
            os.unlink(outpath)

    def test_remove_enantiomers_no_stereo(self):
        """Test with molecules having no stereochemistry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as infile:
            infile.write("CCO mol1_1\n")
            infile.write("CCCO mol2_1\n")
            inpath = infile.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as outfile:
            outpath = outfile.name

        try:
            result = remove_enantiomers(inpath, outpath)
            assert "mol1" in result
            assert "mol2" in result
        finally:
            os.unlink(inpath)
            os.unlink(outpath)

    def test_remove_enantiomers_tolerates_blank_lines(self, tmp_path):
        """M59: switched to iter_smi_records, which skips blank/comment lines.

        The old hand-rolled parser did `vals = line.split()` then indexed
        vals[0]/vals[1] unconditionally, so a blank line raised a bare
        IndexError and aborted the whole function.
        """
        inpath = tmp_path / "in.smi"
        inpath.write_text("CCO mol1_1\n\n# a comment\nCCCO mol2_1\n")
        outpath = tmp_path / "out.smi"

        result = remove_enantiomers(str(inpath), str(outpath))

        assert "mol1" in result
        assert "mol2" in result

    def test_remove_enantiomers_rejects_missing_id(self, tmp_path):
        """A non-blank, non-comment line with no ID must still fail loudly,
        just as an InputValidationError rather than a bare IndexError."""
        inpath = tmp_path / "in.smi"
        inpath.write_text("CCO\n")
        outpath = tmp_path / "out.smi"

        with pytest.raises(InputValidationError):
            remove_enantiomers(str(inpath), str(outpath))


class TestAmendConfiguration:
    """Tests for the amend_configuration() function."""

    def test_amend_configuration_complete(self):
        """Test with complete stereo enumeration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as f:
            # Single stereo center with both configurations
            f.write("C[C@H](O)F mol_1\n")
            f.write("C[C@@H](O)F mol_2\n")
            path = f.name

        try:
            result = amend_configuration(path)
            # A single stereocenter (2 possible configurations) with both
            # already present (mol_1, mol_2 -> group "mol") is already
            # complete: amend_configuration must leave it as the same two
            # SMILES, not add a spurious enantiomer or drop one. `"mol" in
            # result` alone would pass even if the value were empty, doubled,
            # or garbage.
            assert set(result.keys()) == {"mol"}
            assert result["mol"] == ["C[C@H](O)F", "C[C@@H](O)F"]
        finally:
            os.unlink(path)

    def test_amend_configuration_w_writes_file(self):
        """Test that amend_configuration_w writes to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as f:
            f.write("C[C@H](O)F mol_1\n")
            path = f.name

        try:
            amend_configuration_w(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 1
        finally:
            os.unlink(path)

    def test_amend_configuration_tolerates_blank_and_comment_lines(self, tmp_path):
        """M59: switched to iter_smi_records. The old parser did
        `tuple(line.strip().split())`, which raises "not enough values to
        unpack" on a blank line -- aborting the whole function -- and "too
        many values to unpack" on a line with a third whitespace column.
        """
        path = tmp_path / "in.smi"
        path.write_text("C[C@H](O)F mol_1\n\n# comment\nC[C@@H](O)F mol_2\n")

        result = amend_configuration(str(path))

        assert "mol" in result

    def test_amend_configuration_tolerates_extra_column(self, tmp_path):
        """A trailing whitespace-separated column beyond SMILES+ID must not
        raise, matching every other consumer of this format."""
        path = tmp_path / "in.smi"
        path.write_text("C[C@H](O)F mol_1 extra_column\n")

        result = amend_configuration(str(path))

        assert "mol" in result

    def test_amend_configuration_rejects_missing_id(self, tmp_path):
        """A non-blank, non-comment line with no ID must still fail loudly."""
        path = tmp_path / "in.smi"
        path.write_text("C[C@H](O)F\n")

        with pytest.raises(InputValidationError):
            amend_configuration(str(path))


class TestIntegration:
    """Integration tests using real molecular examples."""

    def test_enantiomer_detection_rdkit_consistency(self):
        """Test that enantiomer detection is consistent with RDKit."""
        smi1 = "C[C@H](O)F"
        smi2 = "C[C@@H](O)F"

        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        centers1 = Chem.FindMolChiralCenters(mol1, useLegacyImplementation=False)
        centers2 = Chem.FindMolChiralCenters(mol2, useLegacyImplementation=False)

        # These should be detected as enantiomers
        assert enantiomer(centers1, centers2) is True

    def test_create_enantiomer_valid_smiles(self):
        """Test that created enantiomers are valid SMILES."""
        smi = "C[C@H](O)[C@@H](F)Cl"
        result = create_enantiomer(smi)

        # Should produce valid SMILES
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_stereo_info_roundtrip(self):
        """Test that stereo info extraction is consistent."""
        smi = "C[C@H](O)[C@@H](F)[C@H](Cl)Br"
        info = get_stereo_info(smi)

        # Should detect 3 stereo centers
        assert len(info) == 3

        # Create enantiomer and verify all centers inverted
        enantiomer_smi = create_enantiomer(smi)
        enantiomer_info = get_stereo_info(enantiomer_smi)

        assert len(enantiomer_info) == 3
        for orig_val, enan_val in zip(info.values(), enantiomer_info.values()):
            if orig_val == "@":
                assert enan_val == "@@"
            else:
                assert enan_val == "@"


class TestEnantiomerHelperDiastereomers:
    """Regression guards: reflection inverts tetrahedral centers and leaves E/Z alone."""

    def test_enantiomer_pair_with_a_double_bond_is_still_filtered(self):
        """Same E/Z, inverted center: a genuine enantiomeric pair."""
        smiles = ["C/C=C/C[C@H](O)C", "C/C=C/C[C@@H](O)C"]
        result = enantiomer_helper(smiles)
        assert len(result) == 1, f"a genuine enantiomer pair survived: {result}"

    def test_diastereomers_both_survive(self):
        """Different E/Z and inverted center: diastereomers, not enantiomers."""
        smiles = ["C/C=C/C[C@H](O)C", "C/C=C\\C[C@@H](O)C"]
        result = enantiomer_helper(smiles)
        assert len(result) == 2, f"a diastereomer was discarded: {result}"

    def test_two_centers_partially_inverted_both_survive(self):
        """Inverting only one of two centers gives a diastereomer."""
        smiles = ["C[C@H](O)[C@H](F)Cl", "C[C@@H](O)[C@H](F)Cl"]
        result = enantiomer_helper(smiles)
        assert len(result) == 2, f"a diastereomer was discarded: {result}"

    def test_two_centers_fully_inverted_is_filtered(self):
        """Inverting both centers gives the enantiomer."""
        smiles = ["C[C@H](O)[C@H](F)Cl", "C[C@@H](O)[C@@H](F)Cl"]
        result = enantiomer_helper(smiles)
        assert len(result) == 1, f"a genuine enantiomer pair survived: {result}"

    def test_duplicate_smiles_collapse_to_one(self):
        """The same molecule twice is a duplicate, not an enantiomeric pair.

        ``enantiomer_helper`` now filters on ``enantiomer_key``, the sorted
        set of a molecule's own canonical SMILES and its mirror image's.
        Identical SMILES canonicalize to the same string and so share a key
        regardless of the mirror-image half of that set, which is why the
        duplicate is collapsed here -- not because of any special-case
        equality check.
        """
        result = enantiomer_helper(["C[C@H](O)F", "C[C@H](O)F"])
        assert len(result) == 1, result

    def test_meso_duplicate_from_amend_configuration_is_collapsed(self):
        """A meso form and its string-inverted twin are one molecule, not a pair.

        ``amend_configuration_w`` runs before this filter and appends
        ``create_enantiomer(smi)`` for every isomer with no partner in its
        group. For a meso compound that string surgery produces a DIFFERENT
        SMILES for the SAME molecule, which no pairwise enantiomer test can
        catch -- the two are not an enantiomeric pair. Without collapsing it
        the species is embedded, optimized and written twice.
        """
        meso = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"
        meso_inverted = "O[C@@H](C(=O)O)[C@H](O)C(=O)O"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(meso)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(meso_inverted)
        ), "the two strings must denote one molecule or this test means nothing"

        result = enantiomer_helper(
            [
                "O[C@H](C(=O)O)[C@H](O)C(=O)O",
                meso,
                "O[C@@H](C(=O)O)[C@@H](O)C(=O)O",
                meso_inverted,
            ]
        )
        assert len(result) == 2, f"tartaric acid must yield one of L/D plus meso, once: {result}"


class TestAreEnantiomers:
    """Direct tests for are_enantiomers().

    ``enantiomer_helper`` filters via ``enantiomer_key`` instead of this
    function, so nothing else in the pipeline calls it; it is public and
    documented in CHANGELOG.md, so it gets its own coverage here rather than
    being left an advertised-but-untested predicate.
    """

    def test_true_enantiomeric_pair(self):
        """A single inverted tetrahedral center is a genuine enantiomer pair."""
        assert are_enantiomers("C[C@H](O)F", "C[C@@H](O)F") is True

    def test_identical_smiles_are_not_enantiomers(self):
        """A molecule is not its own enantiomeric partner."""
        assert are_enantiomers("CCO", "CCO") is False

    def test_achiral_pair_is_not_enantiomers(self):
        """Two different achiral molecules have no stereocenter to invert."""
        assert are_enantiomers("CCO", "CCCO") is False

    def test_geometric_pair_is_not_enantiomers(self):
        """A reflection cannot change E/Z, so cis/trans isomers are not a pair."""
        assert are_enantiomers("C/C=C/C", "C/C=C\\C") is False

    def test_partially_inverted_two_center_diastereomer_is_not_enantiomers(self):
        """Inverting only one of two centers gives a diastereomer, not a pair."""
        assert are_enantiomers("C[C@H](O)[C@H](F)Cl", "C[C@@H](O)[C@H](F)Cl") is False

    def test_fully_inverted_two_center_pair_is_enantiomers(self):
        """Inverting both centers of a two-center molecule gives its enantiomer."""
        assert are_enantiomers("C[C@H](O)[C@H](F)Cl", "C[C@@H](O)[C@@H](F)Cl") is True

    def test_unparseable_smiles_returns_false_without_raising(self):
        """Unparseable input is handled, not propagated as an exception."""
        assert are_enantiomers("not_a_smiles(((", "C[C@H](O)F") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
