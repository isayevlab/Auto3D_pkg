"""
Exception hierarchy for Auto3D package.

This module defines a structured exception hierarchy for better error handling
and more informative error messages throughout the package.
"""


class Auto3DError(Exception):
    """Base exception for all Auto3D errors.

    All Auto3D-specific exceptions inherit from this class, allowing users
    to catch all Auto3D errors with a single except clause if desired.

    Attributes:
        hint: Per-raise override for the actionable next step
            ``cli.errors.get_error_hint`` prints under the message, in the
            same spirit as ``DependencyError.dependency_name``. Three states,
            and the difference between the last two matters:

            - ``None`` (the default) -- unset; ``hint_key`` (if it names an
              entry) or the hint for this exception *class* is used instead.
            - a non-empty string -- shown instead of the class hint.
            - ``""`` -- shown instead of the class hint, and that is nothing.
              Used where the class hint would be a non-sequitur and the
              message already says what to do (the end-of-run
              ``auto3d validate`` failure is an ``InputValidationError``, but
              "run ``auto3d validate <file>``" is absurd advice to print
              right after that command already ran and showed what is
              wrong).

            Not preserved across pickling: ``BaseException.__reduce__``
            replays only ``args``. Hints are a CLI presentation concern and
            the CLI catches in-process, so nothing depends on it surviving.
        hint_key: A short, per-raise-site reason (e.g. ``"gpu_unavailable"``,
            ``"output_exists"``) that ``cli.errors.get_error_hint`` looks up
            as ``(type(error), hint_key)`` in its own table, before falling
            back to ``hint`` and the class hint. Lets more than one raise
            site of the same exception class carry its own CLI-flag-specific
            hint (a bare ``GPUError`` message never spells ``--no-gpu`` or
            ``--gpu-idx``; the table does) without that flag text leaking
            into the message or ``hint`` a Python-API caller sees.
            ``None`` (the default) skips the table. Not preserved across
            pickling, for the same reason ``hint`` is not.
    """

    def __init__(self, *args: object, hint: str | None = None, hint_key: str | None = None) -> None:
        super().__init__(*args)
        self.hint = hint
        self.hint_key = hint_key


class ConfigurationError(Auto3DError):
    """Raised when there's an invalid configuration.

    This includes missing required parameters, invalid parameter values,
    or incompatible parameter combinations.
    """

    pass


class InputValidationError(Auto3DError):
    """Raised when input file validation fails.

    This includes invalid SMILES strings, malformed SDF files,
    or files with missing required fields.
    """

    pass


class ModelError(Auto3DError):
    """Base exception for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load.

    This includes corrupted model files or incompatible model formats.
    """

    pass


class NumericalError(ModelError):
    """Raised when numerical instability is detected.

    This includes NaN or Inf values in energies or forces during
    neural network potential calculations. Usually indicates
    problematic molecular geometries or model limitations.
    """

    pass


class OptimizationError(Auto3DError):
    """Base exception for optimization-related errors.

    Raised directly (not through a subclass) when no 3D structure converges
    for a molecule; see WorkflowOrchestrator._run_pipeline and
    batch_opt/model_wrapper.py.
    """

    pass


class FileFormatError(Auto3DError):
    """Raised for unsupported or invalid file formats.

    Auto3D supports .smi (SMILES) and .sdf (SDF) file formats.
    """

    pass


class DependencyError(Auto3DError):
    """Raised when a required dependency is not available.

    Some features require optional dependencies like OpenEye toolkits,
    TorchANI, or ASE. This error is raised when these are needed but not
    installed.

    Attributes:
        dependency_name: A short key identifying the missing dependency
            (e.g. ``"openeye"``, ``"torchani"``, ``"ase"``). ``cli.errors.
            get_error_hint`` looks this up in its own hints map to show an
            install command, read via
            ``getattr(error, "dependency_name", "unknown")`` on every raise
            site. Defaults to
            ``"unknown"`` so a caller that raises ``DependencyError(msg)``
            without naming a dependency still gets a (generic) hint rather
            than a crash in the hint lookup.
    """

    def __init__(
        self,
        message: str,
        dependency_name: str | None = None,
        *,
        hint: str | None = None,
        hint_key: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint, hint_key=hint_key)
        self.dependency_name = dependency_name or "unknown"


class GPUError(Auto3DError):
    """Raised for GPU-related errors.

    This includes missing CUDA devices when GPU computation is requested,
    or GPU memory issues.
    """

    pass
