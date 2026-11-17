"""Aimnet availability check, shared by every ``import aimnet`` site.

The ``aimnet`` package is a required pip dependency (below Python 3.15; see
``AIMNET_PYTHON_CEILING``) but is deliberately absent from the conda-forge package (see conda-recipe/meta.yaml and
docs/source/howto/conda_build.rst: aimnet's own dependency
nvalchemi-toolkit-ops is pip-only, so aimnet cannot be packaged for
conda-forge). Every ``import aimnet`` site in Auto3D calls
:func:`require_aimnet` first, so a missing aimnet surfaces as one
consistent, actionable ``DependencyError`` (CLI exit 3, same convention as
missing torchani) instead of a raw ``ModuleNotFoundError`` traceback.

A sibling of ``preflight.py`` and the ``adapter`` package rather than part of
either, so both can use it without importing each other.
"""

from __future__ import annotations

import sys

from Auto3D.foundation.exceptions import DependencyError

#: The first Python version ``aimnet`` cannot be installed on: its dependency
#: nvalchemi-toolkit-ops declares ``requires-python <3.15``. ``pyproject.toml``
#: makes the ``aimnet`` requirement conditional on the same bound
#: (``python_version < "3.15"``), so that pip still installs current Auto3D on
#: 3.15 rather than falling back to 2.3.1, the newest release that needs no
#: aimnet. A test keeps the two in step. Raise both when upstream lifts its cap.
AIMNET_PYTHON_CEILING = (3, 15)

_UNSUPPORTED_PYTHON_MESSAGE = (
    "AIMNet optimizing engines require the 'aimnet' package, which cannot be "
    "installed on Python {version}: its dependency nvalchemi-toolkit-ops does "
    "not support it yet."
)
_UNSUPPORTED_PYTHON_HINT = (
    "Use a Python 3.11-{last} environment for AIMNet2, or choose "
    "--engine ANI2x or ANI2xt (optimizing_engine='ANI2x' or 'ANI2xt')."
)

_MESSAGE = (
    "AIMNet optimizing engines require the 'aimnet' package, which is not "
    "installed. Install it with 'pip install aimnet' (this works inside "
    "conda environments), or choose optimizing_engine='ANI2x' or 'ANI2xt'."
)


def _python_version() -> tuple[int, int]:
    """The running interpreter's (major, minor); a seam for tests."""
    return sys.version_info.major, sys.version_info.minor


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
            running = _python_version()
            if running >= AIMNET_PYTHON_CEILING:
                # "pip install aimnet" would fail here, so say why instead.
                major, minor = AIMNET_PYTHON_CEILING
                raise DependencyError(
                    _UNSUPPORTED_PYTHON_MESSAGE.format(version=f"{running[0]}.{running[1]}"),
                    dependency_name="aimnet",
                    hint=_UNSUPPORTED_PYTHON_HINT.format(last=f"{major}.{minor - 1}"),
                ) from exc
            raise DependencyError(_MESSAGE, dependency_name="aimnet") from exc
        raise
