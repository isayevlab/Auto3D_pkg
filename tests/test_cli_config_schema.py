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
    assert config.optimizing_engine == "ANI2X"  # Should be normalized to uppercase
    assert config.use_gpu is False


def test_merge_cli_overrides():
    """CLI overrides should take precedence."""
    from Auto3D.cli.config_schema import CLIConfig, merge_configs

    base = CLIConfig(path=Path("test.smi"), k=5, use_gpu=True)
    overrides = {"k": 10, "use_gpu": False}

    merged = merge_configs(base, overrides)
    assert merged.k == 10
    assert merged.use_gpu is False
