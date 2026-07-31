"""Pytest configuration and shared fixtures for Auto3D tests.

This module provides session-scoped fixtures for expensive resources like
neural network models, avoiding redundant loading across tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Test file paths
TEST_DIR = Path(__file__).parent
FILES_DIR = TEST_DIR / "files"


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


@pytest.fixture(autouse=True)
def _release_gpu_memory_after_slow_tests(request):
    """Release cached models and GPU memory after each *slow* test.

    The slow suite runs many full GPU pipelines and AIMNet2 Hessian/thermo
    calculations back-to-back in a single process. Without releasing GPU memory
    and cached models between them, memory pressure accumulates and
    non-deterministically corrupts later GPU work -- e.g. ``calc_thermo``'s
    AIMNet2 Hessian yields imaginary frequencies, so the thermochemistry result
    is garbage (or its properties are never written, giving a ``KeyError``).
    This made the slow tests pass individually but fail under combined ordering.

    Scoped to slow tests on purpose: fast tests skip this teardown so the
    session-scoped ``aimnet_model`` fixture stays warm (the fast gate must not
    reload models). It is also a no-op on CPU / CI, where there is no CUDA cache.
    """
    yield
    if request.node.get_closest_marker("slow") is None:
        return
    import gc

    from Auto3D.model_factory import ModelFactory

    ModelFactory.clear_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def job_dir(tmp_path):
    """Give each test its own pipeline output directory.

    The pipeline writes job folders next to its input file. Tests that share an
    input directory therefore collide under combined or randomized ordering,
    which is why the heavy end-to-end modules were excluded from CI. Copying the
    input into a per-test directory removes the shared state (audit M31).
    """
    d = tmp_path / "job"
    d.mkdir()
    return d


@pytest.fixture
def isolated_input(job_dir):
    """Copy a file from tests/files into this test's own directory.

    Returns a callable: ``isolated_input("smiles2.smi") -> str`` (absolute path).
    """
    import shutil

    def _copy(name: str) -> str:
        dest = job_dir / name
        shutil.copy(FILES_DIR / name, dest)
        return str(dest)

    return _copy
