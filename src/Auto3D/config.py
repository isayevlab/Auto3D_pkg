"""
Configuration classes for Auto3D.

This module provides typed configuration using dataclasses and Protocols
for better type safety and IDE support.
"""

import operator
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

import torch

from Auto3D.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CAPACITY,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_ENERGY_PATIENCE,
    DEFAULT_ENERGY_TOL,
    DEFAULT_OPT_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RMSD_THRESHOLD,
)
from Auto3D.exceptions import ConfigurationError

# Single source of truth for the numeric bounds every entry point must
# enforce (Auto3DOptions.__post_init__ below and CLIConfig's model
# validator in cli/config_schema.py). Keeping one table -- rather than a
# hand-maintained bound list in each place -- is the point of this phase:
# a bound added/changed here takes effect on every path with no second edit.
#
# name -> (comparison, limit). A field's value must satisfy
# ``value <comparison> limit``; see _BOUND_OPS for the supported set.
FIELD_BOUNDS: dict[str, tuple[str, float]] = {
    "k": ("ge", 1),
    "window": ("gt", 0),
    "mpi_np": ("ge", 1),
    "opt_steps": ("ge", 1),
    "convergence_threshold": ("gt", 0),
    "patience": ("ge", 1),
    "threshold": ("gt", 0),
    "batchsize_atoms": ("ge", 1),
    "memory": ("ge", 1),
    "capacity": ("ge", 1),
    "max_confs": ("ge", 1),
}

_BOUND_OPS: dict[str, tuple[object, str]] = {
    "ge": (operator.ge, ">="),
    "gt": (operator.gt, ">"),
}


def check_field_bounds(values: dict) -> None:
    """Validate ``values`` (field name -> value) against ``FIELD_BOUNDS``.

    Shared by Auto3DOptions.__post_init__ and CLIConfig's model validator so
    both entry points reject the same out-of-range values with the same
    message -- this is what closes C10/M27 on every path instead of just one.

    A value of ``None`` or ``False`` means "not specified" (dynamic/default
    behavior) on the optional numeric fields (k, window, max_confs, memory)
    and is skipped, matching both classes' existing sentinel conventions.
    Fields missing from ``values`` are skipped too, so callers may pass a
    partial mapping.

    Raises:
        ConfigurationError: naming the field and the received value.
    """
    for name, (kind, limit) in FIELD_BOUNDS.items():
        if name not in values:
            continue
        value = values[name]
        if value is None or value is False:
            continue
        cmp, symbol = _BOUND_OPS[kind]
        try:
            in_bounds = cmp(value, limit)
        except TypeError as exc:
            # A non-numeric value (e.g. threshold="0.3", a str) makes the
            # comparison itself raise -- a bare TypeError here is exactly
            # the kind of untyped raise this phase is closing everywhere
            # else (see the mutual-exclusion and range checks below/above),
            # so it must be a ConfigurationError too, not a fall-through
            # exception the CLI's `handle_error` treats as an "Unexpected
            # Error" (exit code 1, no hint) instead of a configuration
            # problem (exit code 2, "run auto3d config init").
            raise ConfigurationError(
                f"{name} must be a number, got {value!r}"
            ) from exc
        if not in_bounds:
            raise ConfigurationError(
                f"{name} must be {symbol} {limit}, got {value!r}"
            )
    _check_selectors_mutually_exclusive(values)


def _check_selectors_mutually_exclusive(values: dict) -> None:
    """Reject ``k`` and ``window`` when both are specified (M28).

    They are alternative conformer-selection strategies -- top-k vs. an
    energy window -- and ``ConformerRanker.run`` (ranking.py) only consults
    one of them, so specifying both means one is silently inert. This is
    what the shipped ``thorough`` preset (cli/commands/config.py) did before
    this fix: ``k=10`` and ``window=5.0`` together, with ``k`` always
    winning.

    Called from inside ``check_field_bounds`` (rather than as a second call
    each caller must remember to make) so both ``Auto3DOptions.__post_init__``
    and ``CLIConfig``'s model validator inherit it automatically -- neither
    needed a code change to pick this up.

    ``select_tautomers`` (Auto3D/tautomer.py) already rejects the equivalent
    combination for tautomer selection, but with a bare ``ValueError`` (one
    of the un-typed raises M29 tracks) -- not an ``Auto3DError`` subclass.
    This raises ``ConfigurationError`` instead, matching this module's own
    convention (every other bound above does the same) and its docstring's
    "incompatible parameter combinations" case, so it can be caught the same
    way as any other configuration-shape violation. The message deliberately
    echoes select_tautomers's wording ("Only k OR window needs to be
    specified") rather than inventing new phrasing.

    ``None``/``False`` mean "not specified", the same convention used above:
    by the time this runs, an out-of-range k/window (e.g. ``k=0``) has
    already raised in the loop above, so a value reaching here is either
    unset or a valid, deliberately-specified one.
    """
    k = values.get("k")
    window = values.get("window")
    if k and window:
        raise ConfigurationError(
            "Only one of k or window may be specified, got "
            f"k={k!r} and window={window!r}"
        )


def optimizer_worker_indices(
    use_gpu: bool, gpu_idx: "int | list[int]"
) -> list[int]:
    """Return the per-process indices for the optimizer worker pool.

    One worker per GPU when running on GPU with a list of indices; a single
    worker otherwise. On CPU the index is unused (the worker runs on
    ``torch.device('cpu')``), so a list of indices must NOT fan out into N
    processes that all contend for the same cores -- that wastes memory (N model
    loads) and risks OOM on a small box. The isomer worker uses this same count
    to emit exactly one "Done" sentinel per optimizer, so both call sites must
    agree to avoid deadlock.
    """
    if isinstance(gpu_idx, int):
        return [gpu_idx]
    if use_gpu:
        return list(gpu_idx)
    # CPU with a list of indices: collapse to a single worker (index unused).
    return [gpu_idx[0] if gpu_idx else 0]


@dataclass
class Auto3DOptions:
    """Configuration options for Auto3D conformer generation.

    This dataclass provides proper type hints, validation, and IDE support
    for Auto3D configuration.

    Example:
        >>> from Auto3D import Auto3DOptions, main
        >>> config = Auto3DOptions(path="input.smi", k=5)
        >>> result = main(config)
    """

    # Input/Output
    path: str | None = None
    """Path to input .smi or .sdf file containing SMILES/molecules."""

    k: int | bool = False
    """Output top-k structures for each SMILES. Set to int or False."""

    window: float | bool = False
    """Output structures within x kcal/mol of lowest energy. Set to float or False."""

    verbose: bool = False
    """When True, save all metadata while running."""

    job_name: str = ""
    """Folder name to save all metadata."""

    # Tautomer settings
    enumerate_tautomer: bool = False
    """When True, enumerate tautomers for the input."""

    tauto_engine: str = "rdkit"
    """Program to enumerate tautomers: 'rdkit' or 'oechem'."""

    pKaNorm: bool = True
    """Normalize ionization state to pH ~7.4 (only with tauto_engine='oechem')."""

    # Isomer settings
    isomer_engine: str = "rdkit"
    """Program for generating 3D isomers: 'rdkit' or 'omega'."""

    enumerate_isomer: bool = True
    """When True, enumerate cis/trans and R/S isomers."""

    mode_oe: str = "classic"
    """Omega mode: 'classic', 'macrocycle', 'dense', 'pose', 'rocs', or 'fast_rocs'."""

    mpi_np: int = 4
    """Number of CPU cores for isomer generation."""

    max_confs: int | None = None
    """Maximum conformers per SMILES. None uses dynamic number (num_heavy_atoms - 1)."""

    # GPU settings
    use_gpu: bool = True
    """Use GPU when available."""

    gpu_idx: int | list[int] = 0
    """GPU device index or list of indices."""

    capacity: int = DEFAULT_CAPACITY
    """Number of SMILES handled per 1GB memory."""

    # Optimization settings
    optimizing_engine: str = "AIMNET"
    """Engine: 'AIMNET' (=aimnet2), any aimnet registry name (aimnet2-2025, aimnet2-nse, ...), 'ANI2x', 'ANI2xt', or a path to a custom model."""

    patience: int = DEFAULT_PATIENCE
    """Drop conformer from optimization if force doesn't decrease for this many steps."""

    opt_steps: int = DEFAULT_OPT_STEPS
    """Maximum optimization steps per structure."""

    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    """Optimization converges when max force is below this (eV/Å)."""

    threshold: float = DEFAULT_RMSD_THRESHOLD
    """RMSD threshold for duplicate removal (Å)."""

    memory: int | None = None
    """RAM size assigned to Auto3D in GB. None for automatic detection."""

    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS
    """Number of atoms per optimization batch per GB."""

    # Performance options
    allow_tf32: bool = False
    """Enable TF32 for faster matmul on Ampere+ GPUs (less precise). Default False."""

    # Derived/internal
    input_format: str | None = None
    """Input file format ('smi' or 'sdf'), inferred from the input suffix during
    setup. Declared as a real field (rather than a dynamic attribute) so it
    survives dataclasses.replace()/pickling and stays in the dict-like API."""

    def __post_init__(self):
        """Normalize string values to lowercase and validate ranges."""
        self.tauto_engine = self.tauto_engine.lower()
        self.isomer_engine = self.isomer_engine.lower()
        self.mode_oe = self.mode_oe.lower()
        check_field_bounds({name: getattr(self, name) for name in FIELD_BOUNDS})

    def __getitem__(self, key: str):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow dict-like assignment for backward compatibility."""
        setattr(self, key, value)

    def get(self, key: str, default=None):
        """Dict-like get method for backward compatibility."""
        return getattr(self, key, default)

    def keys(self):
        """Return field names for backward compatibility."""
        return [f.name for f in self.__dataclass_fields__.values()]

    def items(self):
        """Return field name-value pairs for backward compatibility."""
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_optimization_config(self) -> "OptimizationConfig":
        """Create an OptimizationConfig from these options.

        Returns:
            OptimizationConfig with values from this Auto3DOptions instance.
        """
        return OptimizationConfig(
            opt_steps=self.opt_steps,
            convergence_threshold=self.convergence_threshold,
            patience=self.patience,
            batchsize_atoms=self.batchsize_atoms,
        )


@dataclass
class OptimizationConfig:
    """Configuration for geometry optimization.

    This dataclass encapsulates all parameters related to the FIRE optimizer
    and convergence criteria, replacing the previous untyped dict approach.

    Example:
        >>> config = OptimizationConfig(opt_steps=1000, convergence_threshold=0.005)
        >>> optimizer = optimizing(in_f, out_f, model, device, config)
    """

    opt_steps: int = DEFAULT_OPT_STEPS
    """Maximum number of optimization steps per structure."""

    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    """Force convergence threshold in eV/Angstrom. Structure converges when
    maximum force falls below this value."""

    patience: int = DEFAULT_PATIENCE
    """Number of steps without force decrease before dropping a conformer
    as oscillating."""

    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS
    """Number of atoms per optimization batch. Larger values use more GPU
    memory but may be faster."""

    energy_tol: float = DEFAULT_ENERGY_TOL
    """Energy convergence threshold in eV. Used for early termination when
    energy stabilizes."""

    energy_patience: int = DEFAULT_ENERGY_PATIENCE
    """Number of steps energy must be stable before considering converged."""

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with existing code.

        Returns:
            Dictionary with optimization parameters using legacy key names.
        """
        return {
            "opt_steps": self.opt_steps,
            "opttol": self.convergence_threshold,  # Legacy key name
            "patience": self.patience,
            "batchsize_atoms": self.batchsize_atoms,
            "energy_tol": self.energy_tol,
            "energy_patience": self.energy_patience,
        }


class ChunkMeta(TypedDict):
    """Metadata for a processing chunk."""

    output: str
    """Path to final output SDF."""

    optimized_og: str
    """Path to optimized SDF before ranking."""

    output_taut: str
    """Path to tautomer output SMI."""

    smiles_enumerated: str
    """Path to enumerated SMILES file."""

    smiles_reduced: str
    """Path to reduced SMILES file."""

    smiles_hashed: str
    """Path to hashed SMILES file."""

    enumerated_sdf: str
    """Path to enumerated SDF file."""

    sorted_sdf: str
    """Path to sorted SDF file."""

    housekeeping_folder: str
    """Path to housekeeping folder."""

    path: str
    """Input path for this chunk."""

    dir: str
    """Working directory for this chunk."""


@runtime_checkable
class NNPModel(Protocol):
    """Protocol for Neural Network Potential models.

    Custom NNP models must implement this interface to be compatible
    with Auto3D's optimization engine.

    Attributes:
        coord_pad: Padding value for coordinates (typically 0).
        species_pad: Padding value for species (typically -1).

    Example:
        >>> class MyNNP(torch.nn.Module):
        ...     coord_pad = 0
        ...     species_pad = -1
        ...
        ...     def forward(self, species, coords, charges):
        ...         # Return energies tensor of shape (batch_size,)
        ...         return energies
    """

    coord_pad: int
    """Padding value for coordinates in batched tensors."""

    species_pad: int
    """Padding value for atomic species in batched tensors."""

    def forward(
        self,
        species: torch.Tensor,
        coords: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate energies for a batch of molecules.

        Args:
            species: Atomic numbers, shape (batch_size, max_atoms).
            coords: Atomic coordinates, shape (batch_size, max_atoms, 3).
            charges: Molecular charges, shape (batch_size,).

        Returns:
            Energies tensor of shape (batch_size,) in eV.
        """
        ...


# Type alias for backward compatibility
OptionsDict = Auto3DOptions
