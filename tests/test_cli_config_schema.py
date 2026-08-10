# tests/test_cli_config_schema.py
"""Tests for CLI configuration schema."""

from pathlib import Path

import pytest


def test_config_defaults():
    """Config should have sensible defaults."""
    from Auto3D.cli.config_schema import CLIConfig

    config = CLIConfig(path=Path("test.smi"))
    assert config.optimizing_engine == "AIMNET"
    assert config.use_gpu is True
    assert config.opt_steps == 2000


def test_config_validation_k_positive():
    """k must be positive if set."""
    from pydantic import ValidationError

    from Auto3D.cli.config_schema import CLIConfig

    with pytest.raises(ValidationError):
        CLIConfig(path=Path("test.smi"), k=-1)


def test_config_validation_engine():
    """optimizing_engine must be valid."""
    from pydantic import ValidationError

    from Auto3D.cli.config_schema import CLIConfig

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


def test_load_yaml_config_validation_failure_is_configuration_error(tmp_path):
    """A bad value in the YAML file itself (not a CLI override) must raise
    ConfigurationError from load_yaml_config, not a raw pydantic
    ValidationError.

    Before this fix, `merge_configs` (the sibling construction site) already
    translated ValidationError -> ConfigurationError, but `load_yaml_config`
    did not: `auto3d run in.smi -c cfg.yaml` with `cfg.yaml` setting `k: 0`
    exited 1 under the generic "Unexpected Error" panel instead of exit 2
    with the "run auto3d config init" hint, because `execute_run`
    (cli/commands/run.py) only special-cases `Auto3DError` and an
    untranslated `ValidationError` fell through to its `except Exception`
    clause. Both `load_yaml_config` and `merge_configs` now go through the
    shared `build_cli_config` helper.
    """
    from pydantic import ValidationError

    from Auto3D.cli.config_schema import load_yaml_config
    from Auto3D.exceptions import ConfigurationError

    yaml_file = tmp_path / "bad_config.yaml"
    yaml_file.write_text("path: input.smi\nk: 0\n")

    with pytest.raises(ConfigurationError) as exc_info:
        load_yaml_config(yaml_file)
    # Must be the translated ConfigurationError, not the raw pydantic error.
    assert not isinstance(exc_info.value, ValidationError)


def test_config_accepts_registry_and_path_engines(tmp_path):
    from Auto3D.cli.config_schema import CLIConfig

    for eng in ("AIMNET", "aimnet2-2025", "ANI2x"):
        assert CLIConfig(path="x.smi", optimizing_engine=eng).optimizing_engine == eng
    f = tmp_path / "m.pt"
    f.write_text("x")
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


def test_shipped_parameters_yaml_is_complete():
    """The shipped example must show every option, or it teaches a subset.

    ``parameters.yaml`` is an *instance*, not a fourth schema -- but an instance
    is how most users discover which options exist, and three
    (``use_parallel_embedding``, ``parallel_workers``,
    ``parallel_embedding_threshold``) were simply absent with nothing noticing.
    Anything deliberately omitted goes on the allowlist below *with a reason*,
    so "missing" and "intentionally not shown" stop being the same state.
    """
    import yaml as yaml_mod

    from Auto3D.cli.config_schema import OPTIONS_ONLY_FIELDS

    omitted = {
        # k is set instead; check_selectors_mutually_exclusive rejects both at
        # once, so an example cannot demonstrate the two together. The key is
        # still present as `window: None` for discoverability.
        "window": "mutually exclusive with k, which the example sets",
    }

    repo_root = Path(__file__).resolve().parent.parent
    data = yaml_mod.safe_load((repo_root / "parameters.yaml").read_text())

    expected = set(_auto3d_option_fields()) - set(OPTIONS_ONLY_FIELDS)
    missing = expected - set(data) - set(omitted)
    assert not missing, (
        f"parameters.yaml does not mention these options: {sorted(missing)}. "
        f"Add them, or add them to this test's `omitted` allowlist with a reason."
    )
    unknown = set(data) - expected
    assert not unknown, f"parameters.yaml sets options that do not exist: {unknown}"


def test_shipped_legacy_v2_parameters_yaml_loads():
    """docs/legacy-v2/parameters.yaml (``k: 1`` / ``window: False``) must
    validate through ``load_yaml_config`` -- the function ``auto3Dcli.
    _run_legacy_yaml`` and ``cli/commands/run.py`` both call, and now the
    only YAML ingestion path there is -- not through the pipeline itself.
    Before this fix, ``window: False`` was coerced by Pydantic to ``0.0``
    ahead of ``CLIConfig``'s bound-check model validator, which then rejected
    it as a non-positive window: this exact file, run through this exact CLI
    entry point, raised ``ValidationError`` and exited 1 on this branch while
    working unmodified on `main`.

    This used to re-implement the legacy path inline (``yaml.safe_load`` + the
    "None"-string-to-None loop + ``CLIConfig(**parameters)``) while claiming to
    use "the exact construction ``_run_legacy_yaml`` uses". That claim was what
    made the duplicate ingestion layer invisible: the copy under test could
    only ever agree with the copy in ``auto3Dcli.py``, so the three shape
    guards the legacy path was missing (empty file, non-mapping top level, YAML
    syntax error) were untested from either side. Calling the real function is
    strictly better -- it tests the path instead of a replica of it.
    """
    from Auto3D.cli.config_schema import load_yaml_config

    repo_root = Path(__file__).resolve().parent.parent
    yaml_path = repo_root / "docs" / "legacy-v2" / "parameters.yaml"

    config = load_yaml_config(yaml_path)  # must not raise
    assert config.k == 1
    assert config.window is None  # False normalized to CLIConfig's own sentinel
    assert config.memory is None
    assert config.max_confs is None
    config.to_auto3d_options()  # must not raise either


def test_shipped_legacy_v2_tauto_yaml_raises_configuration_error():
    """``docs/legacy-v2/tauto.yaml`` carries keys from a removed feature and
    must be rejected as an ``Auto3D.exceptions.ConfigurationError``, naming
    them -- exactly what CHANGELOG.md and docs/source/migration-3.0.rst now
    tell 4.0 users to catch.

    Both documents previously said this raised a field-named
    ``pydantic.ValidationError``, which stopped being true when the legacy
    YAML path moved onto ``build_cli_config``: a reader who wrote
    ``except pydantic.ValidationError`` around it would catch nothing and let
    the error escape. This pins the type the migration guide promises, at the
    exact file the guide names.
    """
    from pydantic import ValidationError

    from Auto3D.cli.config_schema import load_yaml_config
    from Auto3D.exceptions import Auto3DError, ConfigurationError

    repo_root = Path(__file__).resolve().parent.parent
    yaml_path = repo_root / "docs" / "legacy-v2" / "tauto.yaml"
    assert yaml_path.exists(), "the migration guide names this file"

    with pytest.raises(ConfigurationError) as exc_info:
        load_yaml_config(yaml_path)

    # ConfigurationError is an Auto3DError, so `except Auto3DError` in
    # cli/commands/run.py catches it -- exit 2 with a hint, not exit 1 under
    # "Unexpected Error".
    assert isinstance(exc_info.value, Auto3DError)
    # ...and it is NOT the raw pydantic error the docs used to name.
    assert not isinstance(exc_info.value, ValidationError)
    # The field names survive the translation; that is the whole reason the
    # message is worth showing.
    assert "tauto_k" in str(exc_info.value)


def test_build_cli_config_translates_non_pydantic_validator_errors():
    """A value a field validator cannot coerce must still surface as
    ``ConfigurationError``, not as whatever exception the coercion raised.

    ``build_cli_config`` translated only ``ValidationError``. Pydantic turns a
    ``ValueError``/``AssertionError`` raised inside a validator into a
    field-named ``ValidationError``, but re-raises anything else untouched --
    and ``CLIConfig.parse_gpu_idx`` calls ``int(v)``, which raises
    ``TypeError`` for a mapping. So ``auto3d run in.smi -c cfg.yaml`` with
    ``gpu_idx: {a: 1}`` leaked a bare ``TypeError`` past
    ``cli/commands/run.py``'s ``except Auto3DError`` clause into its
    ``except Exception`` fallback: "Unexpected Error" at exit code 1, for a
    plain bad value in a config file -- the exact outcome
    ``build_cli_config`` exists to eliminate.
    """
    from Auto3D.cli.config_schema import build_cli_config
    from Auto3D.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        build_cli_config(path=Path("in.smi"), k=1, gpu_idx={"a": 1})

    # A list element is coerced the same way, so the same leak was reachable
    # through `gpu_idx: [0, {a: 1}]` too.
    with pytest.raises(ConfigurationError):
        build_cli_config(path=Path("in.smi"), k=1, gpu_idx=[0, {"a": 1}])

    # Still valid input must still be accepted -- the new except clause must
    # not swallow anything that used to work.
    assert build_cli_config(path=Path("in.smi"), k=1, gpu_idx="0,1").gpu_idx == [0, 1]


# =============================================================================
# CLIConfig <-> Auto3DOptions parity
#
# `Auto3DOptions` (Auto3D/config.py) is the authoritative configuration schema:
# the Python API's `main()`/`smiles2mols` take it, so it cannot depend on the
# CLI layer, which makes it the only candidate for "source". `CLIConfig` stays
# hand-written -- generating it with `pydantic.create_model` would hide the
# fields from mypy and IDEs and couple the config layer to a metaprogramming API
# -- and the four tests below make drift impossible to *merge* instead of
# impossible to *write*. Adding an option is two edits (dataclass field,
# CLIConfig field); the second is named by a failing test until it is made.
#
# Four legs, because each catches a failure the others cannot:
#   1. names, both directions
#   2. defaults
#   3. sentinels (None here, False there)
#   4. round-trip through to_auto3d_options()   <-- nothing checked this before
# =============================================================================


def _auto3d_option_fields():
    import dataclasses

    from Auto3D.config import Auto3DOptions

    return {f.name: f for f in dataclasses.fields(Auto3DOptions)}


def test_cliconfig_covers_all_auto3doptions_fields():
    """Leg 1: name parity, in BOTH directions.

    Every user-facing ``Auto3DOptions`` field must be reachable from the
    CLI/YAML via ``CLIConfig``, and ``CLIConfig`` must not carry a field the
    core schema has never heard of -- an option a user can set in a YAML file
    and that ``to_auto3d_options`` then has nowhere to put. Only the reachability
    direction was checked before.

    The exclusion list is ``config_schema.OPTIONS_ONLY_FIELDS``, which states
    the reason per field, rather than a set literal repeated here.
    """
    from Auto3D.cli.config_schema import OPTIONS_ONLY_FIELDS, CLIConfig

    opt_fields = set(_auto3d_option_fields()) - set(OPTIONS_ONLY_FIELDS)
    cli_fields = set(CLIConfig.model_fields)

    assert not opt_fields - cli_fields, (
        f"CLIConfig is missing Auto3DOptions fields: {opt_fields - cli_fields}"
    )
    assert not cli_fields - opt_fields, (
        f"CLIConfig declares fields Auto3DOptions does not have: "
        f"{cli_fields - opt_fields}. A user could set them and nothing would "
        f"read them."
    )
    # An excluded field must actually exist on Auto3DOptions, or the exclusion
    # list is silently hiding a typo instead of an internal field.
    assert set(OPTIONS_ONLY_FIELDS) <= set(_auto3d_option_fields())


def test_cliconfig_defaults_match_auto3doptions():
    """Leg 2: default parity.

    A default written twice is a default that can differ by entry point --
    ``auto3d run in.smi`` and ``main(Auto3DOptions(path=...))`` would then be
    different runs. Sentinel fields are excluded here and checked by leg 3,
    which is about the deliberate ``None``/``False`` difference.
    """
    from Auto3D.cli.config_schema import OPTIONS_ONLY_FIELDS, CLIConfig
    from Auto3D.config import SENTINEL_FIELDS

    skip = set(OPTIONS_ONLY_FIELDS) | set(SENTINEL_FIELDS)
    mismatched = {}
    for name, spec in _auto3d_option_fields().items():
        if name in skip:
            continue
        cli_default = CLIConfig.model_fields[name].default
        if cli_default != spec.default:
            mismatched[name] = (spec.default, cli_default)
    assert not mismatched, (
        f"default drift between Auto3DOptions and CLIConfig "
        f"(field: (Auto3DOptions, CLIConfig)): {mismatched}"
    )


def test_sentinel_fields_use_the_expected_sentinel_on_each_side():
    """Leg 3: sentinel parity.

    The two classes spell "not specified" differently on purpose -- ``None`` on
    ``CLIConfig`` (pydantic coerces ``False`` to ``0`` before any bound check;
    see ``_false_means_unset``) and ``False``/``None`` on ``Auto3DOptions``. That
    difference is bridged, and a bridge is only safe while both ends are known,
    so both ends are asserted rather than assumed.
    """
    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import SENTINEL_FIELDS

    for name in sorted(SENTINEL_FIELDS):
        assert CLIConfig.model_fields[name].default is None, name
        assert _auto3d_option_fields()[name].default in (False, None), name


# One non-default value per CLIConfig field, for the round-trip leg. Every value
# differs from the field's default, so a field that fails to cross shows up as
# the default rather than as an equal value. `optimizing_engine` is 'ANI2xt', a
# built-in name that short-circuits registry resolution without importing the
# optional `aimnet` package (the same reason tests/test_cli.py picks it).
_ROUND_TRIP_VALUES: dict[str, object] = {
    "path": Path("elsewhere/other.smi"),
    "verbose": True,
    "job_name": "round-trip",
    "enumerate_tautomer": True,
    "tauto_engine": "oechem",
    "pKaNorm": False,
    "isomer_engine": "omega",
    "enumerate_isomer": False,
    "mode_oe": "dense",
    "max_confs": 11,
    "mpi_np": 3,
    "optimizing_engine": "ANI2xt",
    "use_gpu": False,
    "gpu_idx": 2,
    "opt_steps": 777,
    "convergence_threshold": 0.02,
    "patience": 111,
    "threshold": 0.44,
    "batchsize_atoms": 2048,
    "use_parallel_embedding": True,
    "parallel_workers": 5,
    "parallel_embedding_threshold": 11,
    "memory": 12,
    "capacity": 43,
    "allow_tf32": True,
}

# k and window are mutually exclusive, so they cannot both be set in one object;
# the round-trip runs once per selector instead.
_ROUND_TRIP_SELECTORS: dict[str, object] = {"k": 7, "window": 4.5}


def test_round_trip_values_table_covers_every_field():
    """The round-trip table must name every field, or leg 4 silently shrinks.

    A new option whose value is missing here would be round-tripped at its
    default, which the assertion below could never distinguish from "did not
    arrive". This is the test that fails first when an option is added.
    """
    from Auto3D.cli.config_schema import CLIConfig

    covered = set(_ROUND_TRIP_VALUES) | set(_ROUND_TRIP_SELECTORS)
    assert covered == set(CLIConfig.model_fields), (
        f"missing from the round-trip table: {set(CLIConfig.model_fields) - covered}; "
        f"unknown fields in it: {covered - set(CLIConfig.model_fields)}"
    )
    # Every listed value must really differ from the default, or the assertion
    # in the round-trip test degenerates into "the default equals the default".
    same_as_default = {
        name: value
        for name, value in _ROUND_TRIP_VALUES.items()
        if value == CLIConfig.model_fields[name].default
    }
    assert not same_as_default, (
        f"these round-trip values equal the field default, so they cannot "
        f"detect a dropped field: {same_as_default}"
    )


@pytest.mark.parametrize("selector", sorted(_ROUND_TRIP_SELECTORS))
def test_to_auto3d_options_forwards_every_field(selector):
    """Leg 4: round-trip parity -- the leg nothing checked before this.

    ``to_auto3d_options`` used to be 27 hand-written ``field=self.field``
    assignments, and the only parity test compared field-*name* sets. Deleting
    one assignment therefore passed every test while silently dropping a user's
    setting: ``auto3d run in.smi -c cfg.yaml`` would read ``patience: 50`` from
    the file, validate it, print it, and then run with 250.

    It is now a ``dataclasses.fields(Auto3DOptions)`` loop with an explicit
    transform table for the four fields whose value genuinely differs across the
    boundary, so forwarding is structural. This test is what keeps it structural.
    """
    from Auto3D.cli.config_schema import CLIConfig

    other = next(s for s in _ROUND_TRIP_SELECTORS if s != selector)
    config = CLIConfig(**_ROUND_TRIP_VALUES, **{selector: _ROUND_TRIP_SELECTORS[selector]})
    options = config.to_auto3d_options()

    # The chosen selector crosses; the other stays "not specified", spelled
    # False on the Auto3DOptions side.
    assert options[selector] == _ROUND_TRIP_SELECTORS[selector]
    assert options[other] is False

    dropped = {}
    for name, value in _ROUND_TRIP_VALUES.items():
        arrived = options[name]
        # `path` is the one type change: Path -> str (str(None) would be the
        # literal "None", so the transform is not just `str`).
        expected = str(value) if name == "path" else value
        if arrived != expected:
            dropped[name] = (expected, arrived)
    assert not dropped, (
        f"to_auto3d_options did not forward these fields (field: (sent, arrived)): {dropped}"
    )


def test_engine_choices_table_matches_cliconfig_literals():
    """``Auto3D.config.ENGINE_CHOICES`` is the isomer/tautomer whitelist.

    ``CLIConfig`` keeps ``Literal`` annotations because a ``Literal`` is a type
    -- mypy and pydantic's error messages both use it -- but it is the one
    remaining hand-written copy of that information, so it is pinned here. This
    is deliberately NOT the ``Field(ge=)`` duplication ``FIELD_BOUNDS``'s
    docstring forbids: a ``Literal`` cannot be derived from the table without
    losing static typing, whereas a numeric bound in a ``Field`` buys nothing a
    type can use.
    """
    import typing

    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import ENGINE_CHOICES

    for name, choices in ENGINE_CHOICES.items():
        annotation = CLIConfig.model_fields[name].annotation
        assert typing.get_args(annotation) == choices, (
            f"{name}: CLIConfig Literal{typing.get_args(annotation)} disagrees "
            f"with Auto3D.config.ENGINE_CHOICES{choices}"
        )


def test_config_init_tables_only_name_real_options():
    """``auto3d config init``'s three tables are instances, and must stay so.

    ``cli/commands/config.py`` holds ``DEFAULT_CONFIG`` (the template), ``PRESETS``
    (quick/balanced/thorough) and ``generate_commented_yaml``'s ``comments`` --
    hand-written key lists, like ``parameters.yaml``. They are not schemas and are
    not consolidated away: a template is a curated subset by design, and each
    comment is CLI-facing prose, not the field documentation ``Auto3DOptions``
    already carries in its per-field docstrings. What they must never do is drift
    into naming an option that does not exist (silently ignored, or rejected by
    ``extra="forbid"`` the moment a user runs the file they were just given) or
    emit a template key with no explanation.
    """
    from Auto3D.cli.commands.config import DEFAULT_CONFIG, PRESETS, generate_commented_yaml
    from Auto3D.cli.config_schema import OPTIONS_ONLY_FIELDS

    real = set(_auto3d_option_fields()) - set(OPTIONS_ONLY_FIELDS)

    assert not set(DEFAULT_CONFIG) - real, (
        f"DEFAULT_CONFIG names options that do not exist: {set(DEFAULT_CONFIG) - real}"
    )
    for name, preset in PRESETS.items():
        assert not set(preset) - real, f"preset {name!r}: {set(preset) - real}"

    # Every template key must come out commented, so `config init` never hands a
    # user a bare key they have to look up elsewhere.
    lines = generate_commented_yaml(dict(DEFAULT_CONFIG)).splitlines()
    uncommented = []
    for key in DEFAULT_CONFIG:
        index = next((i for i, line in enumerate(lines) if line.startswith(f"{key}:")), None)
        if index is None or index == 0 or not lines[index - 1].startswith("#"):
            uncommented.append(key)
    assert not uncommented, (
        f"`auto3d config init` emits these keys with no explanatory comment: {uncommented}"
    )

    # And the template must itself be a runnable configuration.
    from Auto3D.cli.config_schema import build_cli_config

    build_cli_config(**DEFAULT_CONFIG)


def test_no_field_bounds_field_declares_a_second_pydantic_constraint():
    """``FIELD_BOUNDS`` is the only place a numeric bound may be declared.

    Its docstring says so, ``_check_bounds`` enforces it for every field on both
    entry points, and a previous change that added ``Field(ge=1)`` to fields
    already in the table had to be reverted -- two constraint sets for one
    option is precisely the defect this module's parity tests exist to prevent,
    and a pydantic constraint also fails with a different exception path than
    ``check_field_bounds``'s ``ConfigurationError``.

    Checked against pydantic's own metadata rather than the source text, so it
    holds however the constraint is spelled (``Field(ge=...)``,
    ``Annotated[int, Ge(...)]``, ``conint``, ...).
    """
    from Auto3D.cli.config_schema import CLIConfig
    from Auto3D.config import FIELD_BOUNDS

    forbidden = ("ge", "gt", "le", "lt", "multiple_of", "allow_inf_nan")
    offenders = {}
    for name in FIELD_BOUNDS:
        constraints = [
            f"{attr}={getattr(meta, attr)}"
            for meta in CLIConfig.model_fields[name].metadata
            for attr in forbidden
            if getattr(meta, attr, None) is not None
        ]
        if constraints:
            offenders[name] = constraints
    assert not offenders, (
        f"these CLIConfig fields declare a numeric constraint that already "
        f"lives in Auto3D.config.FIELD_BOUNDS: {offenders}. Bounds go in that "
        f"table only; _check_bounds applies them to every entry point."
    )
