# src/Auto3D/cli/config_schema.py
"""Pydantic configuration schema for Auto3D CLI.

This module provides validated configuration using Pydantic, supporting
YAML file loading, CLI overrides, and conversion to Auto3DOptions.
"""

from __future__ import annotations

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


class CLIConfig(BaseModel):
    """Validated configuration for Auto3D CLI."""

    # Input
    path: Path | None = None
    """Path to input .smi or .sdf file, or None for a settings-only config.

    Optional, because the input is supplied on the command line by every
    modern entry point: ``auto3d run INPUT -c cfg.yaml``,
    ``auto3d tautomers INPUT -c cfg.yaml`` and ``auto3d smiles ... -c
    cfg.yaml`` all override whatever ``path`` the file carries (``run``
    explicitly excludes the key -- see ``cli/commands/run.py``). Requiring it
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

    # Isomer settings
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
            allow_missing_path: Permit ``path=None``. Exactly one caller sets
                it: ``auto3d smiles`` (``cli/commands/smiles.py``), which
                takes its molecules from the command line and hands the
                options to ``smiles2mols`` -- and ``smiles2mols`` writes the
                SMILES to a temporary ``.smi`` and assigns ``args.path``
                itself before validating anything, so any path supplied here
                is discarded unread. Supplying a placeholder instead would put
                a path in the options object that names no file the user ever
                mentioned, which is worse than declaring the absence.

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
        # Map built-in engine names back to the canonical form expected by
        # Auto3DOptions; registry names and custom paths pass through verbatim.
        #
        # This table is live, not dead: `_validate_engine` (above) now accepts
        # any case of these three names -- `resolve_engine_name` case-folds
        # them -- but it validates and returns `v` unchanged, so
        # `self.optimizing_engine` still carries whatever case the caller
        # typed (e.g. "ani2x"). This map is what normalizes that back to the
        # exact mixed-case spelling (`ANI2x`/`ANI2xt`) that `MODEL_ANI2X`/
        # `MODEL_ANI2XT` and their downstream exact-match comparisons expect.
        # Registry names/aliases are deliberately left out of this map and
        # pass through as typed -- see test_config_accepts_registry_and_path_engines.
        engine_map = {"ANI2X": "ANI2x", "ANI2XT": "ANI2xt", "AIMNET": "AIMNET"}
        engine = engine_map.get(self.optimizing_engine.upper(), self.optimizing_engine)

        return Auto3DOptions(
            # `str(None)` would be the literal "None", a path that looks real
            # and names nothing; keep the absence an absence.
            path=str(self.path) if self.path is not None else None,
            k=self.k if self.k else False,
            window=self.window if self.window else False,
            enumerate_tautomer=self.enumerate_tautomer,
            tauto_engine=self.tauto_engine,
            pKaNorm=self.pKaNorm,
            isomer_engine=self.isomer_engine,
            enumerate_isomer=self.enumerate_isomer,
            mode_oe=self.mode_oe,
            max_confs=self.max_confs,
            mpi_np=self.mpi_np,
            optimizing_engine=engine,
            use_gpu=self.use_gpu,
            gpu_idx=self.gpu_idx,
            opt_steps=self.opt_steps,
            convergence_threshold=self.convergence_threshold,
            patience=self.patience,
            threshold=self.threshold,
            batchsize_atoms=self.batchsize_atoms,
            memory=self.memory,
            capacity=self.capacity,
            allow_tf32=self.allow_tf32,
            verbose=self.verbose,
            job_name=self.job_name,
        )


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
