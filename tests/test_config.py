"""Unit tests for the config module."""

from __future__ import annotations

import pytest

from Auto3D.foundation.config import Auto3DOptions
from Auto3D.foundation.exceptions import ConfigurationError


def test_input_format_is_real_field_surviving_replace():
    """input_format must be a declared field so ``replace()`` preserves it.

    It was previously set as a dynamic attribute via __setitem__, which
    replace() silently dropped -- a latent AttributeError for any consumer
    reading it off a replace()'d config copy. This test is about `replace()`,
    not about the dict-compat shim (that is `TestAuto3DOptions` and
    `test_dict_style_access_warns_once_per_method`'s job), so it reaches its
    subject through plain attribute access and never triggers a
    DeprecationWarning.
    """
    cfg = Auto3DOptions(path="x.smi", k=1)
    assert cfg.input_format is None  # declared default
    cfg.input_format = "smi"
    assert cfg.input_format == "smi"
    assert "input_format" in type(cfg).model_fields  # a declared field
    cfg2 = cfg.replace(batchsize_atoms=2048)
    assert cfg2.input_format == "smi"  # survives replace()


def test_replace_revalidates_rather_than_copying_blindly():
    """``Auto3DOptions.replace`` must run the validators, as ``dataclasses.replace`` did.

    The method exists because the obvious pydantic translation --
    ``model_copy(update=...)`` -- skips validation entirely. Swapping one for
    the other would have left every copied config unchecked, and nothing else in
    the suite would have noticed: the copy is only made on the way into the
    optimizer, where an out-of-range value surfaces as a bad run rather than an
    error.
    """
    cfg = Auto3DOptions(path="x.smi", k=1)
    with pytest.raises(ConfigurationError):
        cfg.replace(batchsize_atoms=0)  # FIELD_BOUNDS: ge 1
    # And the unvalidated escape hatch really is the one that would have passed.
    assert cfg.model_copy(update={"batchsize_atoms": 0}).batchsize_atoms == 0


def test_to_optimization_config_requires_the_absolute_batchsize():
    """The absolute batch size is the caller's to supply, and there is no default.

    ``Auto3DOptions.batchsize_atoms`` is a *per gigabyte* figure: both entry
    points multiply it by the memory they measured (through
    ``chunk_manager.scaled_batchsize_atoms``) before any optimizer sees it,
    while ``OptimizationConfig.batchsize_atoms`` is the absolute count the
    optimizer actually batches on. A no-argument ``to_optimization_config()``
    silently copied the per-gigabyte number into the absolute field, which is
    how ``smiles2mols`` came to run at 1024 atoms where ``main()`` ran at
    16,384 on the same config. The keyword is required so that reading the
    field without scaling it is a TypeError rather than a quiet 16x.
    """
    from Auto3D.foundation.config import Auto3DOptions

    o = Auto3DOptions(path="x.smi", k=1, batchsize_atoms=1024)
    with pytest.raises(TypeError):
        o.to_optimization_config()
    assert o.to_optimization_config(batchsize_atoms=4096).batchsize_atoms == 4096
    # The options object keeps the per-gigabyte value: the scaling happens on
    # the way out, never in place (review findings #35/#36).
    assert o.batchsize_atoms == 1024


@pytest.mark.parametrize("bad", [0, -1, 1024.5, "4096", None, True])
def test_to_optimization_config_validates_the_batchsize_it_is_handed(bad):
    """The keyword passes the same bounds check the field itself passes.

    Until the scaled value became an argument it reached the optimizer inside an
    ``Auto3DOptions`` copy built by ``replace()``, which re-ran the model's
    validation on it. ``OptimizationConfig`` is a plain dataclass that validates
    nothing, so without a check here a zero, a negative, a string or a
    fractional count would travel all the way to the optimizer -- and on
    ``main()``'s path, across a spawn boundary first, surfacing inside a worker
    rather than at the call that chose the number. The seam every scaled value
    now passes through is this method, so the check lives here and raises the
    same ``ConfigurationError`` an out-of-range field on the options object
    raises (exit 2, with the ``auto3d config init`` hint) rather than a bare
    TypeError or ValueError.
    """
    from Auto3D.foundation.config import Auto3DOptions

    o = Auto3DOptions(path="x.smi", k=1, batchsize_atoms=1024)
    with pytest.raises(ConfigurationError):
        o.to_optimization_config(batchsize_atoms=bad)


class TestAuto3DOptions:
    """Tests for Auto3DOptions dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Auto3DOptions()

        assert config.path is None
        assert config.k is None  # None means not set
        assert config.window is None  # None means not set
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
        """Test that config can be accessed like a dict (deprecated shim)."""
        config = Auto3DOptions(path="/test/path.smi", k=10)

        with pytest.warns(DeprecationWarning):
            assert config["path"] == "/test/path.smi"
        with pytest.warns(DeprecationWarning):
            assert config["k"] == 10

    def test_dict_set(self):
        """Test that config values can be set via dict access (deprecated shim)."""
        config = Auto3DOptions()
        with pytest.warns(DeprecationWarning):
            config["path"] = "/new/path.smi"
        with pytest.warns(DeprecationWarning):
            config["k"] = 3

        assert config.path == "/new/path.smi"
        assert config.k == 3

    def test_items_method(self):
        """Test that items() returns all config key-value pairs (deprecated shim)."""
        config = Auto3DOptions(path="/test.smi", k=5)
        with pytest.warns(DeprecationWarning):
            items = dict(config.items())

        assert "path" in items
        assert "k" in items
        assert items["path"] == "/test.smi"
        assert items["k"] == 5

    def test_keys_method(self):
        """Test that keys() returns all config keys (deprecated shim)."""
        config = Auto3DOptions()
        with pytest.warns(DeprecationWarning):
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


def test_optimization_config_exposes_no_energy_criterion_knobs():
    """The energy-stability knobs are gone, not merely defaulted (audit M1).

    ``test_energy_tol_above_fp32_noise`` stood here until 3.0.0 and asserted
    that ``DEFAULT_ENERGY_TOL >= 1e-3`` so the criterion would be "live". It
    never was: the criterion also required ``fmax < opttol``, which is exactly
    where the force criterion had already converged the structure, so no value
    of the tolerance could have changed an outcome. The knob and the constant
    are removed rather than left inert, and this asserts they stay removed --
    a config field that reaches nothing is worse than no field.
    """
    import Auto3D.foundation.constants as constants
    from Auto3D.foundation.config import OptimizationConfig

    for name in ("energy_tol", "energy_patience"):
        assert not hasattr(OptimizationConfig(), name)
        assert name not in OptimizationConfig().to_dict()
    for name in ("DEFAULT_ENERGY_TOL", "DEFAULT_ENERGY_PATIENCE"):
        assert not hasattr(constants, name)

    with pytest.raises(TypeError):
        OptimizationConfig(energy_tol=1e-3)


def test_capacity_default_comes_from_the_shared_constant():
    """There is one class now, so "the layers agree" is not a thing to check.

    What remains worth pinning is that the default is the shared constant rather
    than a literal re-typed on the field -- which is what made the two-class
    version drift-prone in the first place.
    """
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.constants import DEFAULT_CAPACITY

    assert Auto3DOptions(path="x.smi").capacity == DEFAULT_CAPACITY


def test_dict_style_access_warns_once_per_method():
    """Each of the five dict-compat shim methods must warn, exactly once per call.

    The shim stays for one release (CHANGELOG.md, Unreleased), but every call
    site using it should see a DeprecationWarning telling it to switch to
    attribute access before the shim is removed in 4.0.
    """
    import warnings

    config = Auto3DOptions(path="in.smi")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config["k"]
        config["k"] = 2
        config.get("k")
        config.keys()
        config.items()
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 5
    assert all("Auto3DOptions is not a dict" in m for m in messages)


def test_field_items_is_the_non_deprecated_iteration_accessor():
    """``field_items()`` is the supported replacement for the dict-like
    shim's ``keys()``/``items()``: every declared field, exactly once, in
    ``model_fields`` declaration order, with no warning at all.
    """
    import warnings

    config = Auto3DOptions(path="in.smi")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = config.field_items()
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []

    field_names = [k for k, _ in result]
    assert field_names == list(type(config).model_fields)  # declaration order
    assert len(field_names) == len(set(field_names))  # each field exactly once
    assert result == [(k, getattr(config, k)) for k in type(config).model_fields]


def test_to_dict_names_every_field_by_its_own_name():
    """``OptimizationConfig.to_dict()`` must use each field's own name.

    ``convergence_threshold`` used to come back as the legacy ``opttol`` key
    -- its one reader, ``ensemble_opt``'s ``param["opttol"]``, is the only
    thing that cared about the spelling, so there was no backward-compat
    reason left to keep the mismatch between the field and its dict key.
    """
    from Auto3D.foundation.config import OptimizationConfig

    cfg = OptimizationConfig(
        opt_steps=10, convergence_threshold=0.02, patience=5, batchsize_atoms=1024
    )
    assert cfg.to_dict() == {
        "opt_steps": 10,
        "convergence_threshold": 0.02,
        "patience": 5,
        "batchsize_atoms": 1024,
    }
