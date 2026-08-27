"""Aimnet availability check, shared by every ``import aimnet`` site.

The ``aimnet`` package is a required pip dependency but is deliberately
absent from the conda-forge package (see conda-recipe/meta.yaml and
docs/source/howto/conda_build.rst: aimnet's own dependency
nvalchemi-toolkit-ops is pip-only, so aimnet cannot be packaged for
conda-forge). Every ``import aimnet`` site in Auto3D calls
:func:`require_aimnet` first, so a missing aimnet surfaces as one
consistent, actionable ``DependencyError`` (CLI exit 3, same convention as
missing torchani) instead of a raw ``ModuleNotFoundError`` traceback.

A sibling of ``preflight.py`` and ``adapter.py`` rather than part of either,
so both can use it without importing each other.
"""

from __future__ import annotations

from Auto3D.foundation.exceptions import DependencyError

_MESSAGE = (
    "AIMNet optimizing engines require the 'aimnet' package, which is not "
    "installed. Install it with 'pip install aimnet' (this works inside "
    "conda environments), or choose optimizing_engine='ANI2x' or 'ANI2xt'."
)


def require_aimnet() -> None:
    """Raise ``DependencyError`` if the ``aimnet`` package is not installed.

    Only the clean not-installed case is translated: a
    ``ModuleNotFoundError`` whose ``name`` is ``aimnet`` itself (or one of
    its submodules). A ``ModuleNotFoundError`` naming any *other* module
    means aimnet is installed but its own environment is broken (a missing
    transitive dependency, say); that re-raises unchanged, because "install
    aimnet" would be wrong advice for it.

    Raises:
        DependencyError: aimnet is not installed
            (``dependency_name="aimnet"``, CLI exit 3).
    """
    try:
        import aimnet  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "aimnet" or missing.startswith("aimnet."):
            raise DependencyError(_MESSAGE, dependency_name="aimnet") from exc
        raise
