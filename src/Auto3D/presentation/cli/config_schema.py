# src/Auto3D/cli/config_schema.py
"""Pydantic configuration schema for Auto3D CLI.

This module provides validated configuration using Pydantic, supporting
YAML file loading, CLI overrides, and conversion to Auto3DOptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from Auto3D.engines.models.preflight import resolve_engine_name
from Auto3D.foundation.config import (
    SELECTOR_FIELDS,
    Auto3DOptions,
)
from Auto3D.foundation.exceptions import ConfigurationError

#: ``Auto3DOptions`` fields ``CLIConfig`` deliberately does not carry, with the
#: reason each is excluded. Used by ``to_auto3d_options`` below (which forwards
#: every *other* field mechanically) and by
#: ``tests/test_cli_config_schema.py``'s field-parity test, so "not exposed to
#: the CLI" is stated once instead of once per consumer.
OPTIONS_ONLY_FIELDS: dict[str, str] = {
    "input_format": (
        "derived by the workflow from the input file's suffix, not something a user sets"
    ),
}

# Built-in engine names mapped back to the exact spelling Auto3DOptions and the
# model factory compare against. `_validate_engine` accepts any case (
# `resolve_engine_name` case-folds) but returns the value unchanged, so
# `self.optimizing_engine` still carries whatever the caller typed, e.g.
# "ani2x". Registry names and custom paths are deliberately absent and pass
# through verbatim -- see test_config_accepts_registry_and_path_engines.
_ENGINE_CANONICAL_CASE: dict[str, str] = {
    "ANI2X": "ANI2x",
    "ANI2XT": "ANI2xt",
    "AIMNET": "AIMNET",
}


def _to_options_path(value: Path | None) -> str | None:
    """``Path`` -> ``str``, keeping an absence an absence.

    ``str(None)`` would be the literal ``"None"``: a path that looks real and
    names nothing.
    """
    return str(value) if value is not None else None


def _to_options_engine(value: str) -> str:
    return _ENGINE_CANONICAL_CASE.get(value.upper(), value)


#: The only fields whose value differs between the two classes. Everything else
#: is forwarded verbatim by ``to_auto3d_options``, so a field added to both
#: classes cannot be silently dropped on the way across (which the previous
#: hand-written 27-assignment ``return Auto3DOptions(...)`` allowed -- nothing
#: checked that the mapper actually forwarded every field).
_TO_OPTIONS_TRANSFORMS: dict[str, Any] = {
    "path": _to_options_path,
    "optimizing_engine": _to_options_engine,
}


def build_cli_config(**kwargs: Any) -> Auto3DOptions:
    """Construct an ``Auto3DOptions``, translating any failure into ``ConfigurationError``.

    This is the one construction path every site that builds a ``CLIConfig``
    from external data (a YAML file, merged CLI overrides, a bare CLI
    invocation) should call instead of ``CLIConfig(...)`` directly. Before
    this helper existed, that translation lived only inside `merge_configs`
    -- so a bad value reaching CLIConfig through `load_yaml_config` (a
    config-file value, never merged with any CLI override) raised a raw
    pydantic `ValidationError` that `cli/commands/run.py`'s `except
    Auto3DError` clause does not catch. It fell through to the generic
    "Unexpected Error" panel at exit code 1 instead of the ConfigurationError
    panel (exit 2, "run auto3d config init" hint) every other invalid-
    configuration path produces -- e.g. `auto3d run in.smi -c cfg.yaml` with
    `cfg.yaml` setting `k: 0`. Concentrating the translation here, rather than
    duplicating the try/except at each construction site, is what keeps every
    site in sync with no second edit.

    ``ValidationError`` is not the only way ``CLIConfig(...)`` can fail.
    Pydantic converts a ``ValueError``/``AssertionError`` raised inside a
    validator into a field-named ``ValidationError``, but it re-raises any
    other exception unchanged -- and ``parse_gpu_idx`` above calls ``int(x)``,
    which raises ``TypeError`` on a value that is neither a string, a number,
    nor a list (``auto3d run in.smi -c cfg.yaml`` with ``gpu_idx: {a: 1}``).
    Translating only ``ValidationError`` let that ``TypeError`` escape to
    ``cli/commands/run.py``'s ``except Exception`` clause, producing the
    generic "Unexpected Error" panel at exit code 1 -- precisely the outcome
    this helper exists to eliminate, and for the same class of input (a bad
    value in a config file) it already handles correctly everywhere else. A
    malformed configuration value is a configuration error whichever layer
    notices it, so it exits 2 with the "run auto3d config init" hint like
    every other one.
    """
    try:
        config = Auto3DOptions(**kwargs)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        # Raised out of a field validator's own coercion (see above) rather
        # than by pydantic, so it carries no field name -- say which layer
        # rejected it instead of surfacing a bare "int() argument must be...".
        raise ConfigurationError(f"Invalid configuration value: {exc}") from exc

    # Engine-name resolution happens HERE, not on the model, and the reason is
    # not only layering. `resolve_engine_name` lives in `Auto3D.engines.models`, so a
    # validator on `Auto3DOptions` (foundation layer) would point upward at the
    # engine layer -- the edge PR #159 removed. It would also run on *every*
    # construction, including the pickled reconstruction inside each spawned
    # worker, and resolving a registry name can reach the model registry. A
    # config object validates values; whether a name resolves against a registry
    # is a question for the layer that owns the registry.
    #
    # Doing it at this boundary keeps what CLIConfig's validator bought: a typo
    # in `--engine` or a config file is refused while the user is still looking
    # at their terminal, rather than inside a worker where the error is
    # swallowed. `check_valid_configuration` checks it again on the run path.
    try:
        resolve_engine_name(config.optimizing_engine)
    except (ConfigurationError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    return config


def require_input_path(config: Auto3DOptions) -> Auto3DOptions:
    """Refuse a settings-only config where a runnable one is needed.

    ``path`` is optional so a reusable config file -- settings only, input named
    per run -- is valid; ``auto3d run in.smi -c cfg.yaml`` supplies the input on
    the command line. But a config with no input is not runnable, and that
    obligation used to be discharged inside ``CLIConfig.to_auto3d_options``.
    With one class there is no conversion step to hang it on, so it is a
    function the entry points that need a runnable config call by name.

    Returns ``config`` unchanged, so it reads as a gate at the call site.

    Raises:
        ConfigurationError: ``path`` is unset.
    """
    if config.path is None:
        raise ConfigurationError(
            "No input path: this configuration sets options only.",
            hint=(
                "Add a 'path:' key to the config file, or supply the input on "
                "the command line, e.g. 'auto3d run mols.smi -c config.yaml'."
            ),
        )
    return config


def load_yaml_config(yaml_path: Path) -> Auto3DOptions:
    """Load and validate configuration from a YAML file.

    Uses ``yaml.safe_load``, so Python-specific tags such as
    ``!!python/name:os.system`` are rejected rather than constructed.

    Raises:
        ConfigurationError: the file is empty, is not a YAML mapping, or is
            not parseable as YAML; or the mapping fails ``Auto3DOptions``
            validation (via ``build_cli_config``).
    """
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        # A syntax error in the user's file is a configuration problem, not an
        # internal fault. Raised as ConfigurationError so the CLI reports it at
        # exit 2 with a hint rather than as "Unexpected Error" at exit 1.
        raise ConfigurationError(f"{yaml_path} is not valid YAML: {exc}") from exc

    # An empty file parses to None, and a top-level list or scalar parses to a
    # non-mapping. Both used to reach `data.items()` and surface as
    # `AttributeError: 'NoneType' object has no attribute 'items'` under the
    # generic "Unexpected Error" panel at exit 1 -- an internal-looking crash
    # for an ordinary mistake, and precisely the outcome build_cli_config
    # exists to prevent.
    if data is None:
        raise ConfigurationError(
            f"{yaml_path} is empty. A config file must contain a YAML mapping "
            "of option names to values, for example 'k: 1'."
        )
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{yaml_path} must contain a YAML mapping of option names to "
            f"values, but its top level is a {type(data).__name__}."
        )

    # Convert 'None' strings to actual None
    for key, val in list(data.items()):
        if val == "None":
            data[key] = None

    return build_cli_config(**data)


def merge_configs(base: Auto3DOptions, overrides: dict[str, Any]) -> Auto3DOptions:
    """Merge CLI overrides into base configuration.

    An override replaces the base's value for that field -- it does not
    accumulate alongside it. This matters most for the mutually-exclusive
    conformer-selection strategies in `Auto3D.foundation.config.SELECTOR_FIELDS` (`k`/
    `window`; see `Auto3D.foundation.config.check_selectors_mutually_exclusive`): an
    explicit `--k`/`--window` on the CLI is the user choosing *which*
    strategy to use, substituting for whatever the config file set -- not
    requesting both at once. Before this fix, `auto3d run in.smi -c cfg.yaml
    --k 1` with `cfg.yaml`'s `window: 5.0` left both `k=1` (the override) and
    `window=5.0` (the file's, still sitting in `base_dict`) in the merged
    result, which the mutual-exclusion guard then rejected -- even though
    substituting one selector for the other is exactly what a CLI override is
    for. When the CLI explicitly sets one selector and leaves the rest alone,
    every other selector is cleared here so only the explicit one survives;
    if the CLI explicitly sets more than one selector at once (a genuine
    conflict, not a merge artifact), none are cleared and the mutual-exclusion
    guard below still fires. Iterating `SELECTOR_FIELDS` -- rather than
    hardcoding `k`/`window` a second time here -- means a third selector added
    to that one tuple picks up this substitution behavior automatically.
    """
    base_dict = base.model_dump()

    explicit = [f for f in SELECTOR_FIELDS if overrides.get(f) is not None]
    if len(explicit) == 1:
        for f in SELECTOR_FIELDS:
            if f != explicit[0]:
                base_dict[f] = None

    # Only apply non-None overrides
    for key, value in overrides.items():
        if value is not None:
            base_dict[key] = value

    return build_cli_config(**base_dict)
