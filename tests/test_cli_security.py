# tests/test_cli_security.py
import pytest
import tempfile
from pathlib import Path
from Auto3D.auto3Dcli import load_yaml_config

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
        assert config['use_gpu'] == False

    Path(f.name).unlink()
