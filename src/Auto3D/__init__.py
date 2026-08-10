"""
Auto3D - Automatic generation of low-energy 3D molecular conformers.

This package provides tools for generating 3D conformers from SMILES/SDF files
using neural network potentials (AIMNet2, ANI2x, ANI2xt).
"""


def _detect_version() -> str:
    """Read the installed distribution version, or "unknown" if not installed.

    The ``importlib.metadata`` import lives inside this function on purpose: at
    module level it would put ``version`` and ``PackageNotFoundError`` into the
    package namespace, where they are reachable as ``Auto3D.version`` and
    ``Auto3D.PackageNotFoundError`` -- two names this package never meant to
    export. See ``tests/test_import_boundaries.py``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(__name__)
    except PackageNotFoundError:
        return "unknown"


__version__ = _detect_version()

# NOTE: this module imports nothing but the standard library, and that is a
# tested property, not an accident (tests/test_import_boundaries.py). Importing
# the package root is the cost every consumer pays unconditionally -- `auto3d
# --help`, a build script reading `__version__`, a tool that merely lists
# installed distributions -- and none of them need torch or rdkit.
#
# This file used to end with three eager optional-dependency probes (openeye,
# torchani, and `from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt`) wrapped in
# `warnings.catch_warnings()`. Nothing consumed any of them, and the third one
# defeated the `_LAZY_API` mechanism below outright: it reached ANI2xt_no_rep ->
# the Auto3D.utils barrel -> utils.validation -> Auto3D.models.* + torch, which
# turned `import Auto3D` into 1175 modules and 1.35 s and eagerly loaded 20
# Auto3D submodules. Every probe is already duplicated where it is load-bearing
# and where its result is actually used:
#   * openeye  -> isomer_engine.py (names used at call time) and
#                 utils/validation.py (raises DependencyError with a fix hint)
#   * torchani -> utils/validation.py (same) and batch_opt/ANI2xt_no_rep.py
#   * ANI2xt   -> constructed only through model_factory / models.adapter
# Do not reintroduce a probe here. A dependency is checked where it is needed.

__all__ = [
    "__version__",
    # Core API
    "main",
    "generate_conformers",  # canonical alias for main()
    "smiles2mols",
    # Configuration
    "Auto3DOptions",
    "OptimizationConfig",
    # Model creation
    "create_model",
    "ModelFactory",
    # Property calculators
    "calc_spe",
    "opt_geometry",
    "calc_thermo",
    # Tautomers
    "get_stable_tautomers",
    "select_tautomers",
]

# name -> (module, attribute) for lazy public-API imports. generate_conformers is
# the canonical, self-describing alias for main(); main stays for back-compat.
_LAZY_API: dict[str, tuple[str, str]] = {
    "main": ("Auto3D.auto3D", "main"),
    "generate_conformers": ("Auto3D.auto3D", "main"),
    "smiles2mols": ("Auto3D.auto3D", "smiles2mols"),
    "Auto3DOptions": ("Auto3D.config", "Auto3DOptions"),
    "OptimizationConfig": ("Auto3D.config", "OptimizationConfig"),
    "create_model": ("Auto3D.model_factory", "create_model"),
    "ModelFactory": ("Auto3D.model_factory", "ModelFactory"),
    "calc_spe": ("Auto3D.SPE", "calc_spe"),
    "opt_geometry": ("Auto3D.ASE.geometry", "opt_geometry"),
    "calc_thermo": ("Auto3D.ASE.thermo", "calc_thermo"),
    "get_stable_tautomers": ("Auto3D.tautomer", "get_stable_tautomers"),
    "select_tautomers": ("Auto3D.tautomer", "select_tautomers"),
}


# Lazy imports for public API
def __getattr__(name: str):
    """Lazy import for public API functions (see _LAZY_API).

    Design constraint, not a missed optimization: this **must not** cache the
    resolved object into ``globals()``. Caching would turn every access after
    the first into a snapshot, i.e. exactly the import-time binding that makes
    ``from X import y`` capture a stub -- a test that touches ``Auto3D.main``
    and then patches ``Auto3D.auto3D.main`` would patch nothing, report
    success, and fail somewhere else entirely (the mechanism written up at
    length in ``tests/test_lazy_torchani_import.py``). After the first access
    ``import_module`` is a ``sys.modules`` dict lookup, so a cache buys nothing
    measurable. Pinned by
    ``test_import_boundaries.py::test_getattr_does_not_cache_resolved_attributes``.
    """
    if name in _LAZY_API:
        import importlib

        module_name, attr = _LAZY_API[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Report the public API alongside whatever is genuinely present.

    PEP 562 requires this to accompany ``__getattr__``: a module-level
    ``__getattr__`` resolves names that are not in ``globals()``, and the
    default ``dir()`` only sees ``globals()`` -- so without this, ``"main" in
    dir(Auto3D)`` was False and neither tab-completion nor introspection could
    find the public API.

    The union (rather than ``sorted(__all__)``) matters because ``__dir__``
    replaces the default entirely: imported submodules become real attributes
    of the package, so ``Auto3D.cli`` after ``import Auto3D.cli`` must stay
    visible.
    """
    return sorted(set(globals()) | set(__all__))
