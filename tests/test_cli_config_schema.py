# tests/test_cli_config_schema.py
"""Tests for CLI configuration schema."""

import pytest
from pathlib import Path


def test_config_schema_exists():
    """Config schema class should exist."""
    from Auto3D.cli.config_schema import CLIConfig
    assert CLIConfig is not None


def test_config_defaults():
    """Config should have sensible defaults."""
    from Auto3D.cli.config_schema import CLIConfig

    config = CLIConfig(path=Path("test.smi"))
    assert config.optimizing_engine == "AIMNET"
    assert config.use_gpu is True
    assert config.opt_steps == 2000


def test_config_validation_k_positive():
    """k must be positive if set."""
    from Auto3D.cli.config_schema import CLIConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CLIConfig(path=Path("test.smi"), k=-1)


def test_config_validation_engine():
    """optimizing_engine must be valid."""
    from Auto3D.cli.config_schema import CLIConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CLIConfig(path=Path("test.smi"), optimizing_engine="INVALID")


def test_config_gpu_idx_parsing():
    """gpu_idx should parse string to list."""
    from Auto3D.cli.config_schema import CLIConfig

    config = CLIConfig(path=Path("test.smi"), gpu_idx="0,1,2")
    assert config.gpu_idx == [0, 1, 2]


def test_config_gpu_idx_single():
    """gpu_idx should handle single int."""
    from Auto3D.cli.config_schema import CLIConfig

    config = CLIConfig(path=Path("test.smi"), gpu_idx=0)
    assert config.gpu_idx == 0


def test_config_to_auto3d_options():
    """Config should convert to Auto3DOptions."""
    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import Auto3DOptions

    config = CLIConfig(path=Path("test.smi"), k=5)
    options = config.to_auto3d_options()

    assert isinstance(options, Auto3DOptions)
    assert options.path == "test.smi"
    assert options.k == 5


def test_load_yaml_config(tmp_path):
    """Should load config from YAML file."""
    from Auto3D.cli.config_schema import load_yaml_config

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("""
path: input.smi
k: 10
optimizing_engine: ANI2x
use_gpu: false
""")

    config = load_yaml_config(yaml_file)
    assert config.k == 10
    # Engine strings are preserved verbatim (registry names/paths are
    # case-sensitive); to_auto3d_options resolves built-ins case-insensitively.
    assert config.optimizing_engine == "ANI2x"
    assert config.to_auto3d_options().optimizing_engine == "ANI2x"
    assert config.use_gpu is False


def test_config_accepts_registry_and_path_engines(tmp_path):
    from Auto3D.cli.config_schema import CLIConfig
    for eng in ("AIMNET", "aimnet2-2025", "ANI2x"):
        assert CLIConfig(path="x.smi", optimizing_engine=eng).optimizing_engine == eng
    f = tmp_path / "m.pt"; f.write_text("x")
    assert CLIConfig(path="x.smi", optimizing_engine=str(f)).optimizing_engine == str(f)


def test_config_rejects_garbage_engine():
    import pytest
    from Auto3D.cli.config_schema import CLIConfig
    with pytest.raises(Exception):
        CLIConfig(path="x.smi", optimizing_engine="not-a-model-or-path")


def test_merge_cli_overrides():
    """CLI overrides should take precedence."""
    from Auto3D.cli.config_schema import CLIConfig, merge_configs

    base = CLIConfig(path=Path("test.smi"), k=5, use_gpu=True)
    overrides = {"k": 10, "use_gpu": False}

    merged = merge_configs(base, overrides)
    assert merged.k == 10
    assert merged.use_gpu is False


def test_config_exposes_batchsize_and_tf32():
    """batchsize_atoms and allow_tf32 are accepted by CLIConfig and forwarded to
    Auto3DOptions (so the shipped parameters.yaml loads via `auto3d run -c`)."""
    from Auto3D.cli.config_schema import CLIConfig

    cfg = CLIConfig(path="x.smi", k=1, batchsize_atoms=2048, allow_tf32=True)
    assert cfg.batchsize_atoms == 2048
    assert cfg.allow_tf32 is True
    opts = cfg.to_auto3d_options()
    assert opts.batchsize_atoms == 2048
    assert opts.allow_tf32 is True


def test_shipped_parameters_yaml_loads():
    """The repo-root parameters.yaml must validate against the modern CLI schema."""
    from Auto3D.cli.config_schema import load_yaml_config

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml_config(repo_root / "parameters.yaml")
    # k/window are mutually exclusive; the example sets k and leaves window unset.
    assert cfg.k == 1
    assert cfg.window is None
    cfg.to_auto3d_options()  # must not raise
