"""Pytest configuration and shared fixtures for Auto3D tests.

This module provides session-scoped fixtures for expensive resources like
neural network models, avoiding redundant loading across tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Test file paths
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "openeye: marks tests requiring OpenEye license")


# Session-scoped device fixture
@pytest.fixture(scope="session")
def device():
    """Get the test device (CPU for consistency)."""
    return torch.device("cpu")


# Session-scoped model fixtures - load once per test session
@pytest.fixture(scope="session")
def aimnet_model(device):
    """Load AIMNET model once for all tests."""
    from Auto3D.model_factory import create_model
    return create_model("AIMNET", device)


@pytest.fixture(scope="session")
def ani2x_model(device):
    """Load ANI2x model once for all tests."""
    pytest.importorskip("torchani")
    from Auto3D.model_factory import create_model
    return create_model("ANI2x", device)


@pytest.fixture(scope="session")
def ani2xt_model(device):
    """Load ANI2xt model once for all tests."""
    from Auto3D.model_factory import create_model
    return create_model("ANI2xt", device)


# Common test file paths
@pytest.fixture(scope="session")
def smiles2_path():
    """Path to smiles2.smi test file."""
    return str(FILES_DIR / "smiles2.smi")


@pytest.fixture(scope="session")
def smiles10_path():
    """Path to smiles10.smi test file."""
    return str(FILES_DIR / "smiles10.smi")


@pytest.fixture(scope="session")
def cyclooctane_path():
    """Path to cyclooctane.sdf test file."""
    return str(FILES_DIR / "cyclooctane.sdf")


# Skip markers
@pytest.fixture
def skip_without_openeye():
    """Skip test if OpenEye license is not available."""
    if not os.environ.get("OE_LICENSE"):
        pytest.skip("No OE_LICENSE environment variable set")


@pytest.fixture
def skip_without_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
