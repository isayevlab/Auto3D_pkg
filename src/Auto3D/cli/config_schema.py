# src/Auto3D/cli/config_schema.py
"""Pydantic configuration schema for Auto3D CLI.

This module provides validated configuration using Pydantic, supporting
YAML file loading, CLI overrides, and conversion to Auto3DOptions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from Auto3D.config import (
    SELECTOR_FIELDS,
    SENTINEL_FIELDS,
    Auto3DOptions,
    check_field_bounds,
)
from Auto3D.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CAPACITY,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RMSD_THRESHOLD,
)
from Auto3D.exceptions import ConfigurationError
from Auto3D.models.preflight import resolve_engine_name

#: ``Auto3DOptions`` fields ``CLIConfig`` deliberately does not carry, with the
#: reason each is excluded. Used by ``to_auto3d_options`` below (which forwards
#: every *other* field mechanically) and by
#: ``tests/test_cli_config_schema.py``'s field-parity test, so "not exposed to
#: the CLI" is stated once instead of once per consumer.
OPTIONS_ONLY_FIELDS: dict[str, str] = {
    "input_format": (
        "derived by the workflow from the input file's suffix, not something a "
        "user sets"
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


def _to_options_selector(value: Any) -> Any:
    """``CLIConfig``'s ``None`` "unset" sentinel -> ``Auto3DOptions``'s ``False``.

    Only ``k``/``window`` need this. The other two ``SENTINEL_FIELDS``
    (``memory``, ``max_confs``) are typed ``int | None`` on *both* classes, so
    ``None`` carries across unchanged.
    """
    return value if value else False


def _to_options_engine(value: str) -> str:
    return _ENGINE_CANONICAL_CASE.get(value.upper(), value)


#: The only fields whose value differs between the two classes. Everything else
#: is forwarded verbatim by ``to_auto3d_options``, so a field added to both
#: classes cannot be silently dropped on the way across (which the previous
#: hand-written 27-assignment ``return Auto3DOptions(...)`` allowed -- nothing
#: checked that the mapper actually forwarded every field).
_TO_OPTIONS_TRANSFORMS: dict[str, Any] = {
    "path": _to_options_path,
    "k": _to_options_selector,
    "window": _to_options_selector,
    "optimizing_engine": _to_options_engine,
}


class CLIConfig(BaseModel):
    """Validated configuration for Auto3D CLI."""

    # Input
    path: Path | None = None
    """Path to input .smi or .sdf file, or None for a settings-only config.

    Optional, because the input is supplied on the command line by the modern
    entry point: ``auto3d run INPUT -c cfg.yaml`` overrides whatever ``path``
    the file carries (``run`` explicitly excludes the key -- see
    ``cli/commands/run.py``). Requiring it
    here made the natural reusable config -- settings only, input per run --
    the one shape the CLI refused: ``auto3d config validate cfg.yaml`` and
    ``auto3d run in.smi -c cfg.yaml`` both died on ``path / Field required``
    for a file that describes a perfectly runnable set of options.

    What "optional" must NOT mean is "runnable without an input", so the
    obligation moves to :meth:`to_auto3d_options`, which refuses to build an
    ``Auto3DOptions`` with no path. The deprecated ``auto3d config.yaml``
    form -- the one entry point with no other source of an input path --
    therefore still fails, as a ``ConfigurationError`` at exit 2 exactly as
    before, and now says which key is missing instead of quoting pydantic.
    """

    # Output control
    k: int | None = Field(None, description="Top-k conformers per molecule")
    window: float | None = Field(None, description="Energy window in kcal/mol")

    # Engine settings
    optimizing_engine: str = "AIMNET"
    use_gpu: bool = True
    gpu_idx: int | list[int] = 0

    # Isomer settings.
    #
    # The two Literals below are the *typed* view of Auto3D.config.ENGINE_CHOICES,
    # which is where those whitelists are declared and where
    # Auto3DOptions.__post_init__ enforces them. They stay written out here
    # because a Literal is what mypy and pydantic's error messages can use and
    # neither can read a dict at type-check time --
    # test_engine_choices_table_matches_cliconfig_literals asserts the two agree,
    # so this copy cannot drift. (Contrast the numeric bounds, which are in
    # FIELD_BOUNDS *only*: a Field(ge=) buys nothing a type can use.)
    enumerate_tautomer: bool = False
    tauto_engine: Literal["rdkit", "oechem"] = "rdkit"
    pKaNorm: bool = True
    enumerate_isomer: bool = True
    isomer_engine: Literal["rdkit", "omega"] = "rdkit"
    mode_oe: str = "classic"
    max_confs: int | None = None
    mpi_np: int = 4

    # Optimization settings
    opt_steps: int = DEFAULT_OPT_STEPS
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    patience: int = DEFAULT_PATIENCE
    threshold: float = DEFAULT_RMSD_THRESHOLD
    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS

    # Parallel conformer embedding. Mirrors Auto3DOptions, which
    # test_cliconfig_covers_all_auto3doptions_fields requires -- and which is what
    # makes these reachable from a YAML config rather than Python only.
    # No Field(ge=1) here: the bounds live in Auto3D.config.FIELD_BOUNDS and are
    # enforced by _check_bounds below. A second constraint declared here is the
    # drift that validator's own docstring warns against.
    use_parallel_embedding: bool = False
    parallel_workers: int = 4
    parallel_embedding_threshold: int = 10

    # Resource settings
    memory: int | None = None
    capacity: int = DEFAULT_CAPACITY
    allow_tf32: bool = False

    # Output settings
    verbose: bool = False
    job_name: str = ""

    model_config = {"extra": "forbid"}

    # The field list is `Auto3D.config.SENTINEL_FIELDS` itself, not a second
    # hand-maintained copy of the same four names: a fifth sentinel field
    # added to that constant would otherwise keep `check_field_bounds`'s
    # None/False skip but miss this False->None interception, silently
    # reopening the exact entry-point divergence this phase closed (accepted
    # by Auto3DOptions, rejected by CLIConfig). `sorted` only pins a
    # deterministic argument order -- SENTINEL_FIELDS is a frozenset.
    @field_validator(*sorted(SENTINEL_FIELDS), mode="before")
    @classmethod
    def _false_means_unset(cls, v: Any) -> Any:
        """Map the legacy ``False`` "not specified" sentinel to ``None``.

        Pydantic coerces ``bool`` to ``int``/``float`` (``bool`` is an
        ``int`` subclass) as part of its own type validation, which runs
        *before* the ``mode="after"`` model validator below
        (``_check_bounds``) ever sees the value. So by the time
        ``check_field_bounds``'s ``value is False`` skip runs, ``False`` has
        already become ``0``/``0.0`` and fails the ``k``/``memory``/
        ``max_confs`` >=1 or ``window`` >0 bound -- even though
        ``Auto3DOptions`` (which has no such coercion step) accepts the same
        input. This ``mode="before"`` validator runs first and intercepts
        ``False`` ahead of that coercion, so both classes agree that
        ``k=False``/``window=False``/``memory=False``/``max_confs=False``
        mean "not specified", exactly like the shipped
        ``docs/legacy-v2/parameters.yaml`` example (``window: False``).
        """
        return None if v is False else v

    @field_validator("gpu_idx", mode="before")
    @classmethod
    def parse_gpu_idx(cls, v: Any) -> int | list[int]:
        """Parse gpu_idx from string, int, or list."""
        if isinstance(v, str):
            if "," in v:
                return [int(x.strip()) for x in v.split(",")]
            return int(v)
        if isinstance(v, list):
            return [int(x) for x in v]
        return int(v) if v is not None else 0

    @field_validator("optimizing_engine")
    @classmethod
    def _validate_engine(cls, v: str) -> str:
        """Reject engine names the registry doesn't recognize.

        Delegates to ``resolve_engine_name`` -- the same single source of
        truth used by ``check_valid_configuration`` and (after this fix) the
        auxiliary ``energy``/``optimize``/``thermo`` CLI commands -- instead
        of re-implementing a prefix match here. The prefix match this
        replaced (``v.lower().startswith("aimnet2")``) accepted any typo
        sharing that prefix, e.g. ``aimnet2-2025x``, which then survived
        config parsing and failed later inside a spawned worker where the
        error is swallowed.
        """
        try:
            resolve_engine_name(v)
        except ConfigurationError as exc:
            raise ValueError(str(exc)) from exc
        return v

    @field_validator("tauto_engine", "isomer_engine", mode="before")
    @classmethod
    def normalize_lowercase(cls, v: str) -> str:
        """Normalize to lowercase."""
        return v.lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_bounds(self) -> CLIConfig:
        """Enforce Auto3D.config.FIELD_BOUNDS -- the same table Auto3DOptions
        uses -- instead of a second, hand-maintained set of Field(ge=/gt=)
        constraints that could silently drift from it.
        """
        try:
            check_field_bounds(self.__dict__)
        except ConfigurationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def to_auto3d_options(self, allow_missing_path: bool = False) -> Auto3DOptions:
        """Convert to Auto3DOptions for core workflow.

        Args:
            allow_missing_path: Permit ``path=None``. No caller in the package
                currently sets it; it exists for a caller that supplies its
                molecules by some route other than a file on disk (as
                ``smiles2mols`` does, writing them to a temporary ``.smi`` and
                assigning ``args.path`` itself before validating anything).
                Supplying a placeholder path instead would put a path in the
                options object that names no file the user ever mentioned,
                which is worse than declaring the absence.

        Raises:
            ConfigurationError: ``path`` is unset and ``allow_missing_path``
                is False. ``path`` is optional on this model so a
                settings-only config file is valid (see the field's
                docstring), but an ``Auto3DOptions`` with no input is not
                runnable through ``main()`` -- it would fail inside
                ``check_valid_configuration`` on ``path=None``. This is the
                single place that obligation is discharged, so every consumer
                (``run``/``tautomers``, the legacy YAML form, a direct caller)
                gets the same refusal.
        """
        if self.path is None and not allow_missing_path:
            raise ConfigurationError(
                "No input path: this configuration sets options only.",
                hint=(
                    "Add a 'path:' key to the config file, or supply the "
                    "input on the command line, e.g. 'auto3d run mols.smi "
                    "-c config.yaml'."
                ),
            )
        # Driven off `dataclasses.fields(Auto3DOptions)` -- the authoritative
        # schema -- rather than 27 hand-written `field=self.field` assignments.
        # Those assignments were the one place drift could not be caught:
        # `test_cliconfig_covers_all_auto3doptions_fields` compared field *name*
        # sets, so deleting a line here silently dropped a user's setting on the
        # floor and every test still passed. Iterating the dataclass means a new
        # field is forwarded the moment it exists on both classes, and a field
        # that exists on only one is a KeyError/TypeError here rather than a
        # silent default.
        values: dict[str, Any] = {}
        for spec in dataclasses.fields(Auto3DOptions):
            if spec.name in OPTIONS_ONLY_FIELDS:
                continue
            value = getattr(self, spec.name)
            transform = _TO_OPTIONS_TRANSFORMS.get(spec.name)
            values[spec.name] = transform(value) if transform else value

        return Auto3DOptions(**values)


def build_cli_config(**kwargs: Any) -> CLIConfig:
    """Construct a ``CLIConfig``, translating any construction failure into
    Auto3D's own ``ConfigurationError``.

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
        return CLIConfig(**kwargs)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        # Raised out of a field validator's own coercion (see above) rather
        # than by pydantic, so it carries no field name -- say which layer
        # rejected it instead of surfacing a bare "int() argument must be...".
        raise ConfigurationError(f"Invalid configuration value: {exc}") from exc


def load_yaml_config(yaml_path: Path) -> CLIConfig:
    """Load and validate configuration from a YAML file.

    Uses ``yaml.safe_load``, so Python-specific tags such as
    ``!!python/name:os.system`` are rejected rather than constructed.

    Raises:
        ConfigurationError: the file is empty, is not a YAML mapping, or is
            not parseable as YAML; or the mapping fails ``CLIConfig``
            validation (via ``build_cli_config``).
    """
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        # A syntax error in the user's file is a configuration problem, not an
        # internal fault. Raised as ConfigurationError so the CLI reports it at
        # exit 2 with a hint rather than as "Unexpected Error" at exit 1.
        raise ConfigurationError(
            f"{yaml_path} is not valid YAML: {exc}"
        ) from exc

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


def merge_configs(base: CLIConfig, overrides: dict[str, Any]) -> CLIConfig:
    """Merge CLI overrides into base configuration.

    An override replaces the base's value for that field -- it does not
    accumulate alongside it. This matters most for the mutually-exclusive
    conformer-selection strategies in `Auto3D.config.SELECTOR_FIELDS` (`k`/
    `window`; see `Auto3D.config.check_selectors_mutually_exclusive`): an
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
