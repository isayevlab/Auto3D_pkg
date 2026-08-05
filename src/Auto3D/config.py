"""
Configuration classes for Auto3D.

This module provides typed configuration using dataclasses and TypedDicts
for better type safety and IDE support.
"""

import operator
from dataclasses import dataclass
from typing import TypedDict

from Auto3D.constants import (
    DEFAULT_BATCHSIZE_ATOMS,
    DEFAULT_CAPACITY,
    DEFAULT_CONVERGENCE_THRESHOLD,
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
    # 10, not 1. This table declared ("ge", 1) while utils/validation.py
    # hand-wrote `< 10` twice -- in check_input and again in
    # check_valid_configuration -- so one option had two different minimums and
    # opt_steps=5 was accepted by Auto3DOptions/CLIConfig, printed a banner, and
    # only then failed at run start. 10 is the surviving number because it is
    # the one the optimizer is actually built around, not merely the incumbent:
    # batch_opt/optimization_engine.py's n_steps checks "have all structures
    # converged?" only on `istep % 10 == 0`, emits progress events on the same
    # cadence, and guards its stats print with an explicit `n >= 10` to avoid
    # `n // 10 == 0`. Below 10 steps none of those fire, so the loop has no
    # early exit, no progress, and no reporting -- a regime its own code
    # special-cases. Physically, a FIRE run needs several steps just to build
    # up velocity and timestep, so fewer than 10 cannot converge a real
    # geometry and would hand back an unconverged structure labeled as
    # optimized. Loosening to 1 would have accepted exactly that; the two local
    # checks are deleted instead and this is now the sole declaration.
    "opt_steps": ("ge", 10),
    "convergence_threshold": ("gt", 0),
    "patience": ("ge", 1),
    "threshold": ("gt", 0),
    "batchsize_atoms": ("ge", 1),
    "parallel_workers": ("ge", 1),
    "parallel_embedding_threshold": ("ge", 1),
    "memory": ("ge", 1),
    "capacity": ("ge", 1),
    "max_confs": ("ge", 1),
}

# The subset of FIELD_BOUNDS where None/False mean "not specified" (dynamic/
# default behavior) rather than an actual value to bounds-check -- k/window
# are alternative selection strategies that default to "unset", memory/
# max_confs both have a documented "None means auto-detect/dynamic" meaning
# (see their Auto3DOptions docstrings and CLIConfig's ``int | None`` typing).
#
# The other seven FIELD_BOUNDS entries (mpi_np, opt_steps,
# convergence_threshold, patience, threshold, batchsize_atoms, capacity) have
# no such "unset" meaning -- they always have a concrete default already, so
# there is nothing for None/False to opt out of -- and CLIConfig types them as
# plain `int`/`float` (not `| None`), so pydantic already rejects None there on
# construction. Before this constant existed, the loop below skipped None/
# False for *all eleven* fields, so passing e.g. ``threshold=None`` straight to
# Auto3DOptions (a dataclass with no type-coercion step) was silently accepted
# while ``CLIConfig(threshold=None)`` rejected it -- the same
# entry-point-dependent divergence this phase closed for k/window/memory/
# max_confs, just left open for the other seven. Scoping the skip to exactly
# this set, rather than every key in FIELD_BOUNDS, is what closes it: an
# explicit None/False on any of the seven now falls through to the same
# comparison (and the same ConfigurationError) on both entry points instead of
# being silently waved through on the Auto3DOptions side only.
SENTINEL_FIELDS: frozenset[str] = frozenset({"k", "window", "memory", "max_confs"})

# Mutually-exclusive conformer-selection strategies (see
# check_selectors_mutually_exclusive below). Exposed as a shared tuple --
# rather than left implicit in that function's body -- so other call sites
# that need to know which fields are alternative selectors, not just whether
# a given combination is invalid, use the same list instead of hardcoding
# "k"/"window" a second time. cli/config_schema.py's merge_configs is exactly
# such a site: an explicit CLI override of one selector must clear every
# *other* selector from the base config (so an override substitutes for the
# file's selector instead of accumulating alongside it), which requires
# knowing the full set of selector field names, not just how to reject an
# invalid combination of them. Adding a third selector needs only one edit,
# here, to take effect on both the rejection (below) and the substitution
# (merge_configs).
SELECTOR_FIELDS: tuple[str, ...] = ("k", "window")

# The permitted values for the two enumerable engine fields, in one table, for
# the same reason FIELD_BOUNDS holds the numeric bounds: the alternative is the
# same whitelist hand-written once per validator, drifting silently. It was
# written three times before this -- CLIConfig's two ``Literal``s
# (cli/config_schema.py) and two local ``valid_isomer_engines`` /
# ``valid_tauto_engines`` sets inside check_valid_configuration
# (utils/validation.py), the latter now deleted.
#
# Checked by Auto3DOptions.__post_init__ below, which already lowercases both
# fields, so the check belongs there rather than in a downstream validator that
# can only be reached by some entry points. CLIConfig keeps its ``Literal``s --
# a Literal is a *type*, visible to mypy and to pydantic's error messages, not
# a runtime constraint of the kind FIELD_BOUNDS's docstring forbids duplicating
# -- and tests/test_cli_config_schema.py asserts their arguments equal this
# table, so the one remaining hand-written copy cannot drift.
#
# ``mode_oe`` is deliberately absent: its documented values are omega modes
# nothing validates today, and adding a constraint here would be a new
# restriction rather than the consolidation of an existing one.
ENGINE_CHOICES: dict[str, tuple[str, ...]] = {
    "isomer_engine": ("rdkit", "omega"),
    "tauto_engine": ("rdkit", "oechem"),
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
    behavior) only for the fields in ``SENTINEL_FIELDS`` (k, window,
    max_confs, memory) and is skipped there, matching both classes' existing
    sentinel conventions. Every other bounded field has no "unset" meaning and
    must reject ``None``/``False`` just like any other out-of-range value (see
    ``SENTINEL_FIELDS``'s docstring). Fields missing from ``values`` are
    skipped too, so callers may pass a partial mapping.

    Raises:
        ConfigurationError: naming the field and the received value.
    """
    for name, (kind, limit) in FIELD_BOUNDS.items():
        if name not in values:
            continue
        value = values[name]
        if name in SENTINEL_FIELDS and (value is None or value is False):
            continue
        # `k=True` used to pass every gate and mean k=1: bool is a subclass of
        # int, so operator.ge(True, 1) is True, and `top_k`'s `if k == 1` then
        # matched. Harmless in effect, but `k: int | bool = False` advertises a
        # bool where only `False` was ever meant as a sentinel, so `True` is a
        # value the type says is legal and nothing gives a meaning to. Rejected
        # rather than silently reinterpreted -- a caller who wrote it meant
        # something, and it was not "one conformer".
        if value is True:
            raise ConfigurationError(
                f"{name} must be a number, got True. Only False is a sentinel "
                f"here (meaning 'not specified'); write {name}=1 for one."
            )
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
    check_selectors_mutually_exclusive(values)


def check_selectors_mutually_exclusive(values: dict) -> None:
    """Reject more than one of ``SELECTOR_FIELDS`` (currently ``k``/
    ``window``) being specified at once (M28).

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
    provided = {f: values.get(f) for f in SELECTOR_FIELDS if values.get(f)}
    if len(provided) > 1:
        got = ", ".join(f"{name}={value!r}" for name, value in provided.items())
        raise ConfigurationError(
            f"Only one of {' or '.join(SELECTOR_FIELDS)} may be specified, got {got}"
        )


def check_engine_choices(values: dict) -> None:
    """Validate ``values`` (field name -> value) against ``ENGINE_CHOICES``.

    Fields missing from ``values`` are skipped, so callers may pass a partial
    mapping (the same convention ``check_field_bounds`` uses). Values are
    compared case-insensitively; ``Auto3DOptions.__post_init__`` has already
    lowercased both fields by the time it calls this, so the fold only matters
    for a direct caller.

    Unconditional, unlike the check it replaces: ``check_valid_configuration``
    validated ``tauto_engine`` only when ``enumerate_tautomer`` was true, while
    ``CLIConfig``'s ``Literal["rdkit", "oechem"]`` has always rejected a bad
    value regardless. That was an entry-point divergence -- ``Auto3DOptions(
    tauto_engine="bogus")`` was accepted from Python and refused from the CLI --
    so the stricter of the two is what survives.

    Raises:
        ConfigurationError: naming the field, the received value, and the
            permitted set.
    """
    for name, choices in ENGINE_CHOICES.items():
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, str) and value.lower() in choices:
            continue
        raise ConfigurationError(
            f"{name} must be one of {', '.join(choices)}, got {value!r}"
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
    """Maximum conformers per SMILES.

    ``None`` derives the count per molecule via
    :func:`Auto3D.utils.molprops.calculate_conformer_count`, which is
    ``min(max(1, num_heavy, 2 * 8.481 * num_rotatable ** 1.642), 1000)``
    (https://doi.org/10.1021/acs.jctc.0c01213). The rotatable-bond term dominates
    for anything flexible: glycerol gets **238**, not 5. This docstring used to
    say ``num_heavy_atoms - 1``, which is neither the formula nor the right order
    of magnitude, so a user sizing a run off it underestimated the conformer
    budget by one to two orders of magnitude.
    """

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

    use_parallel_embedding: bool = False
    """Embed conformers in parallel worker processes instead of serially.

    Off by default: parallel embedding spawns processes, so enabling it changes
    a run's resource profile, and that should be the caller's choice rather than
    something they discover.

    Until 3.0.0 this existed only as a constructor argument on the isomer engine
    with no route from here, so no ``main()``/``smiles2mols`` run could reach it
    and the code behind it was reachable only from tests.
    """

    parallel_workers: int = 4
    """Worker processes used when ``use_parallel_embedding`` is on."""

    parallel_embedding_threshold: int = 10
    """Fewest molecules worth embedding in parallel.

    Below this count a run stays serial even with ``use_parallel_embedding`` on,
    since spawning processes for a handful of molecules costs more than it saves.
    """

    batchsize_atoms: int = DEFAULT_BATCHSIZE_ATOMS
    """Atoms per optimization batch, **per gigabyte** of available memory.

    ``ChunkManager`` multiplies this by ``memory`` when you set it, otherwise by
    the *free* GPU memory (or total RAM on CPU), then clamps the product to
    ``_MAX_SCALED_BATCHSIZE_ATOMS`` (16,384). So the default 1024 means 1024
    atoms per batch on a 1 GB card and 16,384 from 16 GB upward -- an 80 GB card
    gets the same 16,384 as a 16 GB one.

    ``ASE.geometry.opt_geometry`` takes the same parameter name **absolutely** --
    1024 means 1024 there whatever the card. The two entry points are up to 16x
    apart on the same value; each docstring says which it is rather than
    pointing at the other.
    """

    # Performance options
    allow_tf32: bool = False
    """Enable TF32 for faster matmul on Ampere+ GPUs (less precise). Default False."""

    # Derived/internal
    input_format: str | None = None
    """Input file format ('smi' or 'sdf'), inferred from the input suffix during
    setup. Declared as a real field (rather than a dynamic attribute) so it
    survives dataclasses.replace()/pickling and stays in the dict-like API."""

    def __post_init__(self):
        """Normalize string values to lowercase, then validate choices and ranges."""
        self.tauto_engine = self.tauto_engine.lower()
        self.isomer_engine = self.isomer_engine.lower()
        self.mode_oe = self.mode_oe.lower()
        check_engine_choices({name: getattr(self, name) for name in ENGINE_CHOICES})
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
        >>> optimizer = optimizing(in_f, out_f, adapter=adapter, device=device, config=config)
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

    # There is deliberately no energy_tol/energy_patience here. Both existed
    # until 3.0.0 and reached an optimizer criterion that could never fire
    # (audit M1), so they were knobs that changed nothing. Removed rather than
    # kept as inert configuration.

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


# Type alias for backward compatibility
OptionsDict = Auto3DOptions
