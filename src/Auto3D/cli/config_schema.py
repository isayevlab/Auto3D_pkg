# src/Auto3D/cli/config_schema.py
"""Pydantic configuration schema for Auto3D CLI.

This module provides validated configuration using Pydantic, supporting
YAML file loading, CLI overrides, and conversion to Auto3DOptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from Auto3D.config import Auto3DOptions, check_field_bounds
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

    # Required
    path: Path
    """Path to input .smi or .sdf file."""

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

    def to_auto3d_options(self) -> Auto3DOptions:
        """Convert to Auto3DOptions for core workflow."""
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
            path=str(self.path),
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


def load_yaml_config(yaml_path: Path) -> CLIConfig:
    """Load and validate configuration from YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Convert 'None' strings to actual None
    for key, val in list(data.items()):
        if val == "None":
            data[key] = None

    return CLIConfig(**data)


def merge_configs(base: CLIConfig, overrides: dict[str, Any]) -> CLIConfig:
    """Merge CLI overrides into base configuration."""
    base_dict = base.model_dump()

    # Only apply non-None overrides
    for key, value in overrides.items():
        if value is not None:
            base_dict[key] = value

    return CLIConfig(**base_dict)
