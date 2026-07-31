"""Unit tests for the config module."""
from __future__ import annotations

from dataclasses import replace

import pytest

from Auto3D.config import Auto3DOptions


def test_input_format_is_real_field_surviving_replace():
    """input_format must be a declared field so dataclasses.replace() preserves it.

    It was previously set as a dynamic attribute via __setitem__, which
    replace() silently dropped -- a latent AttributeError for any consumer
    reading it off a replace()'d config copy.
    """
    cfg = Auto3DOptions(path="x.smi", k=1)
    assert cfg.input_format is None          # declared default
    cfg["input_format"] = "smi"              # dict-like write, as workflow.py does
    assert cfg.input_format == "smi"
    assert "input_format" in cfg.keys()      # part of the dict-like contract
    cfg2 = replace(cfg, batchsize_atoms=2048)
    assert cfg2.input_format == "smi"        # survives replace()


class TestAuto3DOptions:
    """Tests for Auto3DOptions dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Auto3DOptions()

        assert config.path is None
        assert config.k is False  # False means not set
        assert config.window is False  # False means not set
        assert config.verbose is False
        assert config.job_name == ""
        assert config.enumerate_tautomer is False
        assert config.tauto_engine == "rdkit"
        assert config.pKaNorm is True
        assert config.isomer_engine == "rdkit"
        assert config.enumerate_isomer is True
        assert config.mode_oe == "classic"
        assert config.mpi_np == 4
        assert config.max_confs is None
        assert config.use_gpu is True
        assert config.gpu_idx == 0
        assert config.optimizing_engine == "AIMNET"
        assert config.patience == 250
        assert config.opt_steps == 2000
        assert config.convergence_threshold == 0.01
        assert config.threshold == 0.3
        assert config.memory is None
        assert config.capacity == 42
        assert config.batchsize_atoms == 1024

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = Auto3DOptions(
            path="/path/to/input.smi",
            k=5,
            window=2.0,
            verbose=True,
            job_name="test_job",
            use_gpu=False,
            optimizing_engine="ANI2x",
        )

        assert config.path == "/path/to/input.smi"
        assert config.k == 5
        assert config.window == 2.0
        assert config.verbose is True
        assert config.job_name == "test_job"
        assert config.use_gpu is False
        assert config.optimizing_engine == "ANI2x"

    def test_dict_access(self):
        """Test that config can be accessed like a dict."""
        config = Auto3DOptions(path="/test/path.smi", k=10)

        assert config["path"] == "/test/path.smi"
        assert config["k"] == 10

    def test_dict_set(self):
        """Test that config values can be set via dict access."""
        config = Auto3DOptions()
        config["path"] = "/new/path.smi"
        config["k"] = 3

        assert config.path == "/new/path.smi"
        assert config.k == 3

    def test_items_method(self):
        """Test that items() returns all config key-value pairs."""
        config = Auto3DOptions(path="/test.smi", k=5)
        items = dict(config.items())

        assert "path" in items
        assert "k" in items
        assert items["path"] == "/test.smi"
        assert items["k"] == 5

    def test_keys_method(self):
        """Test that keys() returns all config keys."""
        config = Auto3DOptions()
        keys = list(config.keys())

        assert "path" in keys
        assert "k" in keys
        assert "optimizing_engine" in keys

    def test_gpu_idx_single_int(self):
        """Test gpu_idx with single integer."""
        config = Auto3DOptions(gpu_idx=2)
        assert config.gpu_idx == 2

    def test_gpu_idx_list(self):
        """Test gpu_idx with list of integers."""
        config = Auto3DOptions(gpu_idx=[0, 1, 2])
        assert config.gpu_idx == [0, 1, 2]

    def test_immutable_default_list(self):
        """Test that default list values are not shared between instances."""
        config1 = Auto3DOptions()
        config2 = Auto3DOptions()

        # If gpu_idx is a list, modifying one shouldn't affect the other
        # But with default int value, this just verifies independence
        config1["k"] = 5
        assert config2.k is False  # Default is False, not None


class TestChunkMeta:
    """Tests for ChunkMeta TypedDict."""

    def test_chunk_meta_structure(self):
        """Test that ChunkMeta can be used as expected."""
        from Auto3D.config import ChunkMeta

        meta: ChunkMeta = {
            "output": "/path/to/output.sdf",
            "optimized_og": "/path/to/optimized.sdf",
            "enumerated_sdf": "/path/to/enumerated.sdf",
            "sorted_sdf": "/path/to/sorted.sdf",
            "housekeeping_folder": "/path/to/housekeeping",
        }

        assert meta["output"] == "/path/to/output.sdf"
        assert meta["housekeeping_folder"] == "/path/to/housekeeping"


def test_energy_tol_above_fp32_noise():
    from Auto3D.constants import DEFAULT_ENERGY_TOL
    # fp32 ULP at typical molecular total energies (~thousands of eV) is ~1e-3 eV;
    # the tolerance must be at or above that to be a live criterion.
    assert DEFAULT_ENERGY_TOL >= 1e-3


def test_capacity_default_matches_across_layers():
    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import Auto3DOptions

    assert Auto3DOptions(path="x.smi").capacity == CLIConfig(path="x.smi").capacity


def test_negative_k_rejected():
    from Auto3D.config import Auto3DOptions
    from Auto3D.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", k=-1)


def test_negative_window_rejected():
    from Auto3D.config import Auto3DOptions
    from Auto3D.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", window=-0.5)


def test_zero_k_rejected():
    """k=0 used to be treated as "not specified" (falsy) and silently
    accepted, but CLIConfig has always rejected it (its k >= 1 bound applies
    to any non-None value, and 0 is not None) -- Auto3DOptions must match,
    per Task 1's "one set of bounds, on every path". Only None/False mean
    "not specified" now; see test_default_and_valid_k_window_accepted for
    those sentinels.
    """
    from Auto3D.config import Auto3DOptions
    from Auto3D.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", k=0)


def test_default_and_valid_k_window_accepted():
    from Auto3D.config import Auto3DOptions
    # defaults (False) and valid positives must NOT raise
    Auto3DOptions(path="x.smi")
    Auto3DOptions(path="x.smi", k=5)
    Auto3DOptions(path="x.smi", window=2.0)
    Auto3DOptions(path="x.smi", k=False)  # False is "not specified", allowed


class TestOptimizerWorkerIndices:
    """One optimizer process per GPU on GPU; a single worker otherwise.

    A CPU run with a list of gpu_idx must collapse to ONE worker (the index is
    unused on CPU) so N processes do not contend for the same cores / load the
    model N times. The spawn site and the isomer worker's sentinel count both
    derive from this, so they cannot drift.
    """

    def test_single_int_index(self):
        from Auto3D.config import optimizer_worker_indices
        assert optimizer_worker_indices(True, 0) == [0]
        assert optimizer_worker_indices(False, 2) == [2]

    def test_gpu_list_fans_out(self):
        from Auto3D.config import optimizer_worker_indices
        assert optimizer_worker_indices(True, [0, 1, 2]) == [0, 1, 2]

    def test_cpu_list_collapses_to_one(self):
        from Auto3D.config import optimizer_worker_indices
        assert optimizer_worker_indices(False, [0, 1]) == [0]

    def test_cpu_empty_list_is_safe(self):
        from Auto3D.config import optimizer_worker_indices
        assert optimizer_worker_indices(False, []) == [0]
