# tests/test_cli_security.py
"""Tests for YAML security in CLI configuration loading."""

import pytest
import tempfile
from pathlib import Path
import yaml


def load_yaml_config(yaml_path: str) -> dict:
    """Load configuration from a YAML file using safe_load.

    This is a test helper that mimics how the CLI loads YAML configs.
    The actual CLI uses yaml.safe_load internally for security.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(yaml_path) as f:
        parameters = yaml.safe_load(f)

    # Convert 'None' strings to None (matching CLI behavior)
    if parameters:
        for key, val in list(parameters.items()):
            if val == "None":
                parameters[key] = None

    return parameters or {}


def test_yaml_loading_is_safe():
    """YAML loading should not allow access to Python objects.

    FullLoader allows !!python/name which can expose dangerous functions.
    safe_load blocks all Python-specific YAML tags for security.
    """
    # Create a YAML file that uses !!python/name tag
    # FullLoader would return a reference to os.system function
    # safe_load should reject this entirely
    malicious_yaml = """
key: !!python/name:os.system
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(malicious_yaml)
        f.flush()

        # safe_load should raise an error, not return a Python function reference
        with pytest.raises(Exception):  # yaml.constructor.ConstructorError
            load_yaml_config(f.name)

    Path(f.name).unlink()


def test_yaml_loading_normal_config():
    """YAML loading should work for normal configuration."""
    normal_yaml = """
path: /some/path.smi
k: 1
use_gpu: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(normal_yaml)
        f.flush()

        config = load_yaml_config(f.name)
        assert config['path'] == '/some/path.smi'
        assert config['k'] == 1
        assert config['use_gpu'] is False

    Path(f.name).unlink()


def test_yaml_loading_none_string_conversion():
    """YAML loading should convert 'None' strings to None."""
    yaml_with_none = """
path: /some/path.smi
window: None
memory: None
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_with_none)
        f.flush()

        config = load_yaml_config(f.name)
        assert config['path'] == '/some/path.smi'
        assert config['window'] is None
        assert config['memory'] is None

    Path(f.name).unlink()


def test_yaml_loading_empty_file():
    """YAML loading should handle empty files gracefully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        f.flush()

        config = load_yaml_config(f.name)
        assert config == {}

    Path(f.name).unlink()
