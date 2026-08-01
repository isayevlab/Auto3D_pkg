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


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("ani2x", "ANI2x"),
        ("ANI2X", "ANI2x"),
        ("ani2xt", "ANI2xt"),
        ("ANI2XT", "ANI2xt"),
        ("Aimnet2", "Aimnet2"),  # not a named engine: passes through verbatim
        ("AIMNET", "AIMNET"),
        ("aimnet", "AIMNET"),
    ],
)
def test_config_accepts_case_insensitive_engine_names(raw, canonical):
    """Regression: `ani2x`/`ANI2X`/`ani2xt`/`ANI2XT`/`Aimnet2` must all be
    accepted by CLIConfig -- they were all rejected with "Unknown
    optimizing_engine" once `_validate_engine` started delegating to
    `resolve_engine_name`, which (before this fix) compared engine names with
    exact, case-sensitive equality. `auto3d run in.smi --engine ani2x` and
    any YAML with `optimizing_engine: ani2x` died on this.

    The CLIConfig field itself preserves the caller's casing verbatim (see
    test_config_accepts_registry_and_path_engines); `to_auto3d_options()` is
    what normalizes the three named engines to their canonical mixed-case
    spelling via `engine_map`, checked here.
    """
    from Auto3D.cli.config_schema import CLIConfig

    config = CLIConfig(path="x.smi", optimizing_engine=raw)
    assert config.optimizing_engine == raw
    assert config.to_auto3d_options().optimizing_engine == canonical


def test_config_rejects_garbage_engine():
    import pytest
    from Auto3D.cli.config_schema import CLIConfig
    with pytest.raises(Exception):
        CLIConfig(path="x.smi", optimizing_engine="not-a-model-or-path")


def test_config_rejects_registry_name_typo():
    """M21: 'aimnet2-2025x' shares the 'aimnet2' prefix with a real registry
    name, so the old prefix-match validator (`v.lower().startswith("aimnet2")`)
    accepted it. Verified live before this fix:
    CLIConfig(path=Path("x.smi"), optimizing_engine="aimnet2-2025x") did not
    raise. The validator now delegates to resolve_engine_name, which performs
    a real registry lookup instead of a prefix match."""
    from pydantic import ValidationError

    from Auto3D.cli.config_schema import CLIConfig

    with pytest.raises(ValidationError, match="aimnet2-2025x"):
        CLIConfig(path=Path("x.smi"), optimizing_engine="aimnet2-2025x")


def test_merge_cli_overrides():
    """CLI overrides should take precedence."""
    from Auto3D.cli.config_schema import CLIConfig, merge_configs

    base = CLIConfig(path=Path("test.smi"), k=5, use_gpu=True)
    overrides = {"k": 10, "use_gpu": False}

    merged = merge_configs(base, overrides)
    assert merged.k == 10
    assert merged.use_gpu is False


def test_merge_configs_cli_k_overrides_file_window():
    """`--k` on the CLI must substitute the config file's `window`, not
    accumulate alongside it.

    `auto3d run in.smi -c cfg.yaml --k 1` with `cfg.yaml` setting
    `window: 5.0` used to hard-fail the mutual-exclusion rule (M28) because
    the override was added to the base dict instead of substituting for the
    file's other selector -- reproduced directly here via `merge_configs`.
    """
    from Auto3D.cli.config_schema import CLIConfig, merge_configs

    base = CLIConfig(path=Path("test.smi"), window=5.0)
    merged = merge_configs(base, {"k": 1})

    assert merged.k == 1
    assert merged.window is None


def test_merge_configs_cli_window_overrides_file_k():
    """Same substitution, the other direction: `--window` must clear the
    file's `k`."""
    from Auto3D.cli.config_schema import CLIConfig, merge_configs

    base = CLIConfig(path=Path("test.smi"), k=10)
    merged = merge_configs(base, {"window": 2.5})

    assert merged.window == 2.5
    assert merged.k is None


def test_merge_configs_explicit_cli_conflict_still_raises():
    """Two explicit, genuinely conflicting CLI selectors (`--k` AND
    `--window` both passed) must still be rejected -- the substitution
    added by this fix only clears the *other* source's selector, not a
    selector the same override dict explicitly sets."""
    from Auto3D.cli.config_schema import CLIConfig, merge_configs
    from Auto3D.exceptions import ConfigurationError

    base = CLIConfig(path=Path("test.smi"))
    with pytest.raises(ConfigurationError):
        merge_configs(base, {"k": 1, "window": 2.0})


def test_merge_configs_validation_failure_is_configuration_error():
    """A CLIConfig validation failure surfacing from merge_configs must be a
    ConfigurationError (exit 2, with a hint), not a raw pydantic
    ValidationError (which cli/commands/run.py's `except Auto3DError`
    clause does not catch, so it fell through to the generic "Unexpected
    Error" exit-1 path instead).
    """
    from Auto3D.cli.config_schema import CLIConfig, merge_configs
    from Auto3D.exceptions import ConfigurationError

    base = CLIConfig(path=Path("test.smi"), k=1)
    with pytest.raises(ConfigurationError):
        merge_configs(base, {"threshold": -1})


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


def test_shipped_legacy_v2_parameters_yaml_loads():
    """docs/legacy-v2/parameters.yaml (``k: 1`` / ``window: False``) must
    validate through the exact construction ``auto3Dcli._run_legacy_yaml``
    uses -- ``yaml.safe_load`` + the "None"-string-to-None conversion +
    ``CLIConfig(**parameters)`` (auto3Dcli.py, around the ``CLIConfig(
    **parameters)`` call) -- not the pipeline itself. Before this fix,
    ``window: False`` was coerced by Pydantic to ``0.0`` ahead of
    ``CLIConfig``'s bound-check model validator, which then rejected it as
    a non-positive window: this exact file, run through this exact CLI
    entry point, raised ``ValidationError`` and exited 1 on this branch
    while working unmodified on `main`.
    """
    import yaml as yaml_mod

    from Auto3D.cli.config_schema import CLIConfig

    repo_root = Path(__file__).resolve().parent.parent
    yaml_path = repo_root / "docs" / "legacy-v2" / "parameters.yaml"

    with open(yaml_path) as f:
        parameters = yaml_mod.safe_load(f)
    for key, val in list(parameters.items()):
        if val == "None":
            parameters[key] = None

    config = CLIConfig(**parameters)  # must not raise
    assert config.k == 1
    assert config.window is None  # False normalized to CLIConfig's own sentinel
    assert config.memory is None
    assert config.max_confs is None
    config.to_auto3d_options()  # must not raise either


def test_cliconfig_covers_all_auto3doptions_fields():
    """Guard against config-layer drift: every user-facing Auto3DOptions field
    must be reachable from the CLI/YAML via CLIConfig. ``input_format`` is set
    internally by the workflow, so it is the only allowed exclusion."""
    import dataclasses

    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import Auto3DOptions

    excluded = {"input_format"}
    opt_fields = {f.name for f in dataclasses.fields(Auto3DOptions)} - excluded
    cli_fields = set(CLIConfig.model_fields)
    missing = opt_fields - cli_fields
    assert not missing, f"CLIConfig is missing Auto3DOptions fields: {missing}"
