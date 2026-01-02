# src/Auto3D/cli/config_schema.py
"""Pydantic configuration schema for Auto3D CLI.

This module provides validated configuration using Pydantic, supporting
YAML file loading, CLI overrides, and conversion to Auto3DOptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from Auto3D.config import Auto3DOptions
from Auto3D.constants import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RMSD_THRESHOLD,
)


class CLIConfig(BaseModel):
    """Validated configuration for Auto3D CLI."""

    # Required
    path: Path
    """Path to input .smi or .sdf file."""

    # Output control
    k: int | None = Field(None, ge=1, description="Top-k conformers per molecule")
    window: float | None = Field(None, gt=0, description="Energy window in kcal/mol")

    # Engine settings
    optimizing_engine: Literal["AIMNET", "ANI2X", "ANI2XT"] = "AIMNET"
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
    mpi_np: int = Field(4, ge=1)

    # Optimization settings
    opt_steps: int = Field(DEFAULT_OPT_STEPS, ge=1)
    convergence_threshold: float = Field(DEFAULT_CONVERGENCE_THRESHOLD, gt=0)
    patience: int = Field(DEFAULT_PATIENCE, ge=1)
    threshold: float = Field(DEFAULT_RMSD_THRESHOLD, gt=0)

    # Resource settings
    memory: int | None = Field(None, ge=1)
    capacity: int = Field(40, ge=1)

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
        return v

    @field_validator("optimizing_engine", mode="before")
    @classmethod
    def normalize_engine(cls, v: str) -> str:
        """Normalize engine name to uppercase."""
        return v.upper() if isinstance(v, str) else v

    @field_validator("tauto_engine", "isomer_engine", mode="before")
    @classmethod
    def normalize_lowercase(cls, v: str) -> str:
        """Normalize to lowercase."""
        return v.lower() if isinstance(v, str) else v

    def to_auto3d_options(self) -> Auto3DOptions:
        """Convert to Auto3DOptions for core workflow."""
        # Map optimizing_engine back to expected format for Auto3DOptions
        engine_map = {"ANI2X": "ANI2x", "ANI2XT": "ANI2xt", "AIMNET": "AIMNET"}
        engine = engine_map.get(self.optimizing_engine, self.optimizing_engine)

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
            memory=self.memory,
            capacity=self.capacity,
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
