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
            verbose=True,
            job_name="test_job",
            use_gpu=False,
            optimizing_engine="ANI2x",
        )

        assert config.path == "/path/to/input.smi"
        assert config.k == 5
        assert config.verbose is True
        assert config.job_name == "test_job"
        assert config.use_gpu is False
        assert config.optimizing_engine == "ANI2x"

    def test_custom_window_value(self):
        """window (used alone, not alongside k -- see TestMutuallyExclusiveSelectors
        in test_config_parity.py) can be set to a custom value."""
        config = Auto3DOptions(path="/path/to/input.smi", window=2.0)

        assert config.window == 2.0

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


def test_optimization_config_exposes_no_energy_criterion_knobs():
    """The energy-stability knobs are gone, not merely defaulted (audit M1).

    ``test_energy_tol_above_fp32_noise`` stood here until 4.0.0 and asserted
    that ``DEFAULT_ENERGY_TOL >= 1e-3`` so the criterion would be "live". It
    never was: the criterion also required ``fmax < opttol``, which is exactly
    where the force criterion had already converged the structure, so no value
    of the tolerance could have changed an outcome. The knob and the constant
    are removed rather than left inert, and this asserts they stay removed --
    a config field that reaches nothing is worse than no field.
    """
    import Auto3D.constants as constants
    from Auto3D.config import OptimizationConfig

    for name in ("energy_tol", "energy_patience"):
        assert not hasattr(OptimizationConfig(), name)
        assert name not in OptimizationConfig().to_dict()
    for name in ("DEFAULT_ENERGY_TOL", "DEFAULT_ENERGY_PATIENCE"):
        assert not hasattr(constants, name)

    with pytest.raises(TypeError):
        OptimizationConfig(energy_tol=1e-3)


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


def test_false_sentinel_accepted_on_every_bound_field_both_entry_points():
    """False must mean "not specified" for k/window/memory/max_confs on BOTH
    Auto3DOptions and CLIConfig, not just one.

    Before this fix, Pydantic coerced `False` -> `0`/`0.0` (bool is an int
    subclass) *before* CLIConfig's `_check_bounds` model validator ever saw
    the value, so `check_field_bounds`'s `value is False` skip (config.py)
    never fired on the CLIConfig path -- CLIConfig rejected all four fields
    set to `False` (as an out-of-range 0/0.0) while Auto3DOptions (a plain
    dataclass with no such coercion step) silently accepted them. Reproduced
    live before this fix:
    ``CLIConfig(path=Path("x.smi"), k=1, window=False)`` raised
    ValidationError while
    ``Auto3DOptions(path="x.smi", k=1, window=False)`` did not raise. The
    shipped ``docs/legacy-v2/parameters.yaml`` sets exactly this
    (``k: 1`` / ``window: False``), so this divergence broke a real,
    in-repo example, not just a hypothetical one.
    """
    from pathlib import Path

    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import Auto3DOptions

    # Auto3DOptions: all four False, together and individually.
    opts = Auto3DOptions(
        path="x.smi", k=False, window=False, memory=False, max_confs=False
    )
    assert opts.k is False
    assert opts.window is False
    assert opts.memory is False
    assert opts.max_confs is False

    # CLIConfig: same four fields, same False values -- must validate too,
    # and must normalize False to CLIConfig's own "unset" sentinel (None).
    cfg = CLIConfig(
        path=Path("x.smi"), k=False, window=False, memory=False, max_confs=False
    )
    assert cfg.k is None
    assert cfg.window is None
    assert cfg.memory is None
    assert cfg.max_confs is None

    # And individually, mixed with a real value for the other selector, the
    # way the shipped legacy example does (k=1, window=False).
    Auto3DOptions(path="x.smi", k=1, window=False)
    CLIConfig(path=Path("x.smi"), k=1, window=False)


def test_non_numeric_threshold_raises_configuration_error():
    """A non-numeric bound value (e.g. threshold="0.3", a str) must raise
    ConfigurationError, not a bare TypeError.

    `operator.gt`/`operator.ge` (config.py's `_BOUND_OPS`) raise TypeError
    when compared against a str, which used to propagate unhandled from
    `check_field_bounds` -- an untyped exception unlike every range check
    beside it, and one the CLI's `handle_error` shows as an opaque
    "Unexpected Error" (exit 1) instead of a configuration problem with a
    hint (exit 2).
    """
    from Auto3D.config import Auto3DOptions
    from Auto3D.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", k=1, threshold="0.3")


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


class TestSentinelsAreNotSilentlyReinterpreted:
    """`k=True` passed every gate and meant `k=1`.

    bool is a subclass of int, so `operator.ge(True, 1)` is True and the bounds
    check let it through; `top_k`'s `if k == 1` then matched. The effect was
    harmless, but `k: int | bool = False` advertises a bool where only `False` was
    ever meant as a sentinel, so `True` was a value the type called legal and
    nothing gave a meaning to. A caller who wrote it meant something, and it was
    not "one conformer".
    """

    def test_k_true_is_rejected_rather_than_read_as_one(self):
        from Auto3D.config import Auto3DOptions
        from Auto3D.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="got True"):
            Auto3DOptions(path="in.smi", k=True)

    def test_k_false_is_still_the_not_specified_sentinel(self):
        """False must keep working: it is how "use window instead" is spelled."""
        from Auto3D.config import Auto3DOptions

        options = Auto3DOptions(path="in.smi", k=False, window=2.0)
        assert options.k is False
        assert options.window == 2.0

    def test_a_real_k_is_unaffected(self):
        from Auto3D.config import Auto3DOptions

        assert Auto3DOptions(path="in.smi", k=1).k == 1


class TestParallelEmbeddingIsReachable:
    """The option must arrive at the isomer engine, not just exist on the config.

    Until 3.0.0 `use_parallel_embedding` was a constructor argument on the isomer
    engine with no route from `Auto3DOptions`, so no `main()` or `smiles2mols` run
    could reach it and `isomers/parallel_embed.py` was reachable only from tests --
    which is why an audit listed that module as dead code.

    Asserting `Auto3DOptions(use_parallel_embedding=True).use_parallel_embedding
    is True` would pass with the plumbing still missing: it tests the dataclass,
    not the wiring. These assert what the factory is actually called with.
    """

    def test_the_option_reaches_the_isomer_engine_factory(self, monkeypatch, tmp_path):
        from Auto3D import auto3D as auto3D_mod
        from Auto3D.config import Auto3DOptions

        seen = {}

        class _StubEngine:
            def run(self):
                raise RuntimeError("stop here: the factory call is what is asserted")

        def _capture(**kwargs):
            seen.update(kwargs)
            return _StubEngine()

        monkeypatch.setattr(
            auto3D_mod.IsomerEngineFactory, "create", staticmethod(_capture)
        )

        smi = tmp_path / "in.smi"
        smi.write_text("CCO ethanol\n")
        options = Auto3DOptions(
            # use_gpu=False: this box and CI are CPU-only, and check_gpu_requested
            # is fatal for a GPU request with no visible device -- it would fire
            # before the factory call under test.
            path=str(smi), k=1, use_gpu=False,
            use_parallel_embedding=True,
            parallel_workers=3,
            parallel_embedding_threshold=2,
        )

        with pytest.raises(RuntimeError, match="stop here"):
            auto3D_mod.smiles2mols(["CCO"], options)

        assert seen.get("use_parallel_embedding") is True, (
            "use_parallel_embedding never reached the isomer engine: the field "
            f"exists on the config but is not plumbed. Factory got: {sorted(seen)}"
        )
        assert seen.get("parallel_workers") == 3, (
            "parallel_workers stayed at the constructor default, so enabling "
            "parallel embedding could not control its worker count"
        )
        assert seen.get("parallel_embedding_threshold") == 2, (
            "parallel_embedding_threshold stayed at its default, so the batch-size "
            "gate could not be tuned"
        )

    def test_the_default_is_still_serial(self):
        """Off by default: enabling it changes a run's resource profile."""
        from Auto3D.config import Auto3DOptions

        options = Auto3DOptions(path="in.smi", k=1)
        assert options.use_parallel_embedding is False

    @pytest.mark.parametrize(
        "field", ["parallel_workers", "parallel_embedding_threshold"]
    )
    def test_a_count_below_one_is_rejected(self, field):
        """Bounds come from FIELD_BOUNDS, so both entry points share them."""
        from Auto3D.config import Auto3DOptions
        from Auto3D.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match=field):
            Auto3DOptions(path="in.smi", k=1, **{field: 0})
