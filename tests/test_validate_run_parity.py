# tests/test_validate_run_parity.py
"""M25 parity: `auto3d validate` must reject exactly what the runner rejects.

Before this fix, cli.commands.validate.validate_smiles_file did not require an
ID column (it took parts[0] with no length check), while file_ops.encode_ids
(via iter_smi_records, on_malformed="raise") always has -- so a SMILES-only
file passed `auto3d validate` and then failed the run, whose own error hint
told the user to run the validator that had just approved it. The two also
disagreed on '#'-prefixed comment lines: validate_smiles_file has always
skipped them; iter_smi_records had no comment handling at all and would try
to parse '#' as data.

These tests check both directions for both defects: a file that passes
`validate_smiles_file` must not fail `encode_ids`, and a file that fails one
must fail the other.
"""
from __future__ import annotations

import pytest

from Auto3D.cli.commands.validate import validate_smiles_file
from Auto3D.exceptions import InputValidationError
from Auto3D.utils.file_ops import encode_ids


def _validator_accepts(path) -> bool:
    return validate_smiles_file(path).valid


def _runner_accepts(path) -> bool:
    """True if encode_ids -- the runner's entry point for a .smi file, called
    from WorkflowOrchestrator._validate_input before any worker is forked --
    accepts `path` without raising."""
    try:
        encode_ids(str(path))
    except InputValidationError:
        return False
    return True


class TestValidateRunnerParity:
    """Every case here must agree in both directions: validator and runner
    both accept, or both reject. Disagreement in either direction is exactly
    M25 (a validator more permissive than the runner is worse than none)."""

    def test_well_formed_file_passes_both(self, tmp_path):
        p = tmp_path / "ok.smi"
        p.write_text("CCO ethanol\nCC(=O)O acetic_acid\n")
        assert _validator_accepts(p)
        assert _runner_accepts(p)

    def test_id_less_line_fails_both(self, tmp_path):
        """M25's headline defect: a SMILES with no ID column. validate used
        to approve this (cli/commands/validate.py:42 took parts[0] with no
        length check); the runner has always rejected it."""
        p = tmp_path / "no_id.smi"
        p.write_text("CCO\n")
        assert not _validator_accepts(p)
        assert not _runner_accepts(p)

    def test_id_less_line_among_valid_lines_fails_both(self, tmp_path):
        """Same defect, but mixed with otherwise-valid data -- the common
        real-world shape (one bad line in an otherwise fine file)."""
        p = tmp_path / "mixed.smi"
        p.write_text("CCO ethanol\nCC(=O)O\n")  # second line has no ID
        assert not _validator_accepts(p)
        assert not _runner_accepts(p)

    def test_comment_line_passes_both(self, tmp_path):
        """A '#'-prefixed line is a comment to both, not data -- even a
        single-token comment with no space, which (absent comment handling)
        would otherwise look exactly like the ID-less defect above."""
        p = tmp_path / "with_comment.smi"
        p.write_text("#comment\nCCO ethanol\n")
        assert _validator_accepts(p)
        assert _runner_accepts(p)

    def test_multi_word_comment_line_passes_both(self, tmp_path):
        p = tmp_path / "with_wordy_comment.smi"
        p.write_text("# this is a comment about the file below\nCCO ethanol\n")
        assert _validator_accepts(p)
        assert _runner_accepts(p)

    def test_comment_only_file_passes_both(self, tmp_path):
        p = tmp_path / "comment_only.smi"
        p.write_text("# nothing but a comment\n")
        assert _validator_accepts(p)
        assert _runner_accepts(p)


class TestValidateCheckSmiFormatParity:
    """check_smi_format (utils/validation.py) is the other runner-side check
    that reads the raw user .smi file directly (via check_input, called from
    both main() and smiles2mols) -- it must agree with the validator too."""

    @staticmethod
    def _check_smi_format_accepts(path) -> bool:
        from types import SimpleNamespace

        from Auto3D.utils.validation import check_smi_format

        args = SimpleNamespace(path=str(path), enumerate_isomer=True)
        try:
            check_smi_format(args)
        except InputValidationError:
            return False
        return True

    def test_id_less_line_fails_both(self, tmp_path):
        p = tmp_path / "no_id.smi"
        p.write_text("CCO\n")
        assert not _validator_accepts(p)
        assert not self._check_smi_format_accepts(p)

    def test_comment_line_passes_both(self, tmp_path):
        p = tmp_path / "with_comment.smi"
        p.write_text("#comment\nCCO ethanol\n")
        assert _validator_accepts(p)
        assert self._check_smi_format_accepts(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
