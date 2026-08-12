"""YAML config loading must be safe, and must fail cleanly on bad input.

These tests previously defined their own ``load_yaml_config`` helper -- a
local shim that "mimics how the CLI loads YAML configs" -- and imported
nothing from Auto3D. All four exercised the shim, so deleting the real
loader entirely would not have failed one of them, and one asserted a
behavior (``handles empty files gracefully``) that the real function did not
have: it raised ``AttributeError: 'NoneType' object has no attribute
'items'``, which the CLI reported as "Unexpected Error" at exit 1.

Every test here now drives ``Auto3D.cli.config_schema.load_yaml_config``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Auto3D.cli.config_schema import load_yaml_config
from Auto3D.exceptions import ConfigurationError


def _write(tmp_path: Path, text: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(text)
    return cfg


class TestYamlIsLoadedSafely:
    """Python-specific YAML tags must never be constructed."""

    def test_python_name_tag_is_refused(self, tmp_path):
        """``!!python/name:`` must not resolve to a live Python object.

        ``yaml.FullLoader`` would return a reference to ``os.system`` here.
        ``safe_load`` refuses the tag outright. The assertion checks the
        *refusal*, and the negative control below checks that ordinary
        configs still load -- without it, a loader that rejected everything
        would satisfy this test.
        """
        cfg = _write(tmp_path, "k: !!python/name:os.system\n")

        with pytest.raises(ConfigurationError, match="not valid YAML"):
            load_yaml_config(cfg)

    def test_python_object_tag_is_refused(self, tmp_path):
        """``!!python/object/apply:`` is the tag that actually executes."""
        cfg = _write(
            tmp_path,
            "k: !!python/object/apply:os.system ['echo pwned']\n",
        )

        with pytest.raises(ConfigurationError, match="not valid YAML"):
            load_yaml_config(cfg)


class TestOrdinaryConfigsStillLoad:
    """Negative control for the refusals above."""

    def test_a_normal_config_loads(self, tmp_path):
        cfg = _write(
            tmp_path,
            "path: mols.smi\nk: 1\nuse_gpu: false\n",
        )

        config = load_yaml_config(cfg)

        # load_yaml_config returns a validated Auto3DOptions, not a dict -- the
        # shim these tests used to call returned a dict, which is one reason
        # they could not have caught a change in the real function.
        assert config.path == "mols.smi"  # stored as str, not Path
        assert config.k == 1
        assert config.use_gpu is False

    def test_the_string_None_becomes_a_real_None(self, tmp_path):
        """`window: None` in YAML is the string "None", not a null.

        Auto3D treats it as "unset" for the sentinel fields, so it must be
        converted before validation or the bounds check would reject a string.
        """
        cfg = _write(tmp_path, "path: mols.smi\nk: 1\nwindow: None\n")

        config = load_yaml_config(cfg)

        assert config.window is None


class TestMalformedFilesFailCleanly:
    """A bad config is the user's mistake, and must not look like a crash.

    Each of these used to reach ``data.items()`` and surface as
    ``AttributeError`` inside the generic "Unexpected Error" panel at exit 1.
    ``ConfigurationError`` is what the CLI maps to exit 2 with a hint.
    """

    def test_an_empty_file_is_refused(self, tmp_path):
        cfg = _write(tmp_path, "")

        with pytest.raises(ConfigurationError, match="is empty"):
            load_yaml_config(cfg)

    def test_a_whitespace_only_file_is_refused(self, tmp_path):
        """Parses to None exactly as an empty file does."""
        cfg = _write(tmp_path, "\n\n   \n")

        with pytest.raises(ConfigurationError, match="is empty"):
            load_yaml_config(cfg)

    def test_a_top_level_list_is_refused(self, tmp_path):
        """``match=`` anchors on the phrase unique to THIS guard's message
        ("...its top level is a {type}.") -- "must contain a YAML mapping" is
        also a substring of the empty-file message (config_schema.py:341), so
        that alone cannot tell this test apart from the empty-file guard
        firing by mistake (e.g. if the not-a-mapping check were deleted and
        the empty-file check's message merely happened to also match).
        """
        cfg = _write(tmp_path, "- k\n- window\n")

        with pytest.raises(ConfigurationError, match="top level is a list"):
            load_yaml_config(cfg)

    def test_a_top_level_scalar_is_refused(self, tmp_path):
        cfg = _write(tmp_path, "just a bare string\n")

        with pytest.raises(ConfigurationError, match="top level is a str"):
            load_yaml_config(cfg)

    def test_unparseable_yaml_is_refused(self, tmp_path):
        """A syntax error, not a structural one."""
        cfg = _write(tmp_path, "k: [1, 2\nwindow: 3\n")

        with pytest.raises(ConfigurationError, match="not valid YAML"):
            load_yaml_config(cfg)


class TestTheCliReportsThemAsConfigurationProblems:
    """End-to-end: the exception type above must reach the user as exit 2.

    Asserting the exception type alone would not prove this -- the CLI could
    still map it to the generic handler. These drive the real command.
    """

    @pytest.mark.parametrize(
        "content,expected",
        [
            ("", "empty"),
            ("- a\n- b\n", "mapping"),
            ("k: [1, 2\n", "not valid YAML"),
        ],
    )
    def test_bad_config_exits_2_with_a_message(self, tmp_path, content, expected):
        from typer.testing import CliRunner

        from Auto3D.cli.app import app

        smi = tmp_path / "mols.smi"
        smi.write_text("CCO m1\n")
        cfg = _write(tmp_path, content)

        result = CliRunner().invoke(app, ["run", str(smi), "-c", str(cfg)])

        assert result.exit_code == 2, result.output
        assert "Unexpected Error" not in result.output
        assert "AttributeError" not in result.output
