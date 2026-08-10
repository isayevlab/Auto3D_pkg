# tests/test_ase_torch_config.py
"""Tests to verify ASE modules don't override TF32 settings."""

import pytest
import torch


def test_thermo_module_no_hardcoded_tf32():
    """ASE thermo module should not override TF32 settings."""
    # Set TF32 to True before importing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Import should not change these settings
    import importlib

    import Auto3D.ASE.thermo as thermo_module

    importlib.reload(thermo_module)

    # Settings should remain unchanged (controlled by torch_config)
    # Note: This test documents expected behavior after fix
    assert torch.backends.cuda.matmul.allow_tf32 == True
    assert torch.backends.cudnn.allow_tf32 == True


def test_geometry_module_no_hardcoded_tf32():
    """ASE geometry module should not override TF32 settings."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import importlib

    import Auto3D.ASE.geometry as geometry_module

    importlib.reload(geometry_module)

    assert torch.backends.cuda.matmul.allow_tf32 == True
    assert torch.backends.cudnn.allow_tf32 == True
