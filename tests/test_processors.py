"""Tests for Auto3D.processors module."""

import os
import tempfile
from pathlib import Path

import pytest

from Auto3D.config import Auto3DOptions
from Auto3D.processors import TautomerProcessor


class TestTautomerProcessor:
    """Tests for TautomerProcessor class."""

    def test_init_stores_config(self):
        """TautomerProcessor should store config."""
        config = Auto3DOptions(path="test.smi", k=1)
        processor = TautomerProcessor(config)
        assert processor.config is config

    def test_process_returns_input_when_disabled(self, tmp_path):
        """When enumerate_tautomer=False, process() should return input path unchanged."""
        config = Auto3DOptions(
            path=str(tmp_path / "test.smi"),
            k=1,
            enumerate_tautomer=False,
        )
        processor = TautomerProcessor(config)

        input_path = "/some/input/path.smi"
        output_path = "/some/output/path.smi"

        result = processor.process(input_path, output_path)

        # Should return original input path
        assert result == input_path

    def test_process_returns_output_when_enabled(self, tmp_path):
        """When enumerate_tautomer=True, process() should return output path."""
        # Create a temporary input file with valid SMILES
        input_file = tmp_path / "input.smi"
        input_file.write_text("C=C(O)C smi1\n")  # Enol form that can tautomerize

        output_file = tmp_path / "output.smi"

        config = Auto3DOptions(
            path=str(input_file),
            k=1,
            enumerate_tautomer=True,
            tauto_engine="rdkit",
            pKaNorm=False,
        )
        processor = TautomerProcessor(config)

        result = processor.process(str(input_file), str(output_file))

        # Should return output path
        assert result == str(output_file)
        # Output file should exist
        assert output_file.exists()


class TestTautomerProcessorConfigOptions:
    """Tests for TautomerProcessor configuration options."""

    def test_rdkit_engine(self, tmp_path):
        """TautomerProcessor should work with rdkit engine."""
        input_file = tmp_path / "input.smi"
        input_file.write_text("CC=O acetaldehyde\n")

        output_file = tmp_path / "output.smi"

        config = Auto3DOptions(
            path=str(input_file),
            k=1,
            enumerate_tautomer=True,
            tauto_engine="rdkit",
        )
        processor = TautomerProcessor(config)

        result = processor.process(str(input_file), str(output_file))
        # `.exists()` alone is satisfied by an empty file, which is also what
        # the *disabled* branch's contract looks like from the outside if
        # `result` were accidentally the input path. Pin both: the enabled
        # branch must return the OUTPUT path specifically, and that file must
        # actually carry the tautomer engine's records, not be empty.
        assert result == str(output_file)
        written = Path(result).read_text().strip()
        assert written, "tautomer engine produced no records in the output file"


def test_tautomer_processor_uses_facade(monkeypatch, tmp_path):
    from Auto3D.config import Auto3DOptions
    import Auto3D.processors as proc

    calls = {}

    class _Eng:
        def run(self):
            calls["ran"] = True

    def fake_create(engine_type, input_path, output_path, pka_norm=True):
        calls["args"] = (engine_type, input_path, output_path, pka_norm)
        return _Eng()

    monkeypatch.setattr(proc, "create_tautomer_engine", fake_create)
    monkeypatch.setattr(proc, "hash_taut_smi", lambda a, b: None)

    # pKaNorm=False is deliberately non-default (Auto3DOptions defaults it to
    # True): a mutation that hardcodes the forwarded value instead of reading
    # self.config.pKaNorm would otherwise still match a default-valued cfg.
    cfg = Auto3DOptions(
        path="x.smi",
        k=1,
        enumerate_tautomer=True,
        tauto_engine="rdkit",
        pKaNorm=False,
    )
    out = proc.TautomerProcessor(cfg).process("in.smi", "out.smi")
    assert out == "out.smi"
    assert calls["ran"] is True
    # Pin the full argument tuple `create_tautomer_engine` was called with, not
    # just the engine name: `args[1]`/`args[2]` catch input_path/output_path
    # being swapped or dropped, and `args[3]` catches the wrong (or a missing)
    # pKaNorm being forwarded -- none of which the old, name-only assertion
    # could have caught.
    assert calls["args"] == ("rdkit", "in.smi", "out.smi", False)


def test_tautomer_processor_skips_when_disabled():
    from Auto3D.config import Auto3DOptions
    from Auto3D.processors import TautomerProcessor

    cfg = Auto3DOptions(path="x.smi", k=1, enumerate_tautomer=False)
    assert TautomerProcessor(cfg).process("in.smi", "out.smi") == "in.smi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
