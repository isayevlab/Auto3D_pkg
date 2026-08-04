"""Import boundaries for the ``Auto3D`` package root: what ``import Auto3D``
is allowed to cost, and what the package root is allowed to expose.

Two properties are locked here.

**Import cost.** ``import Auto3D`` must reach nothing but the standard library.
The package root is the entry point every consumer pays for -- ``auto3d
--help``, ``from Auto3D import __version__`` in a build script, an unrelated
library that merely lists installed packages -- and none of those need torch,
rdkit, or a neural network potential. The cost is asserted as a **module count**
rather than a wall-clock time because seconds are a property of the machine
(CPU, warm page cache, NFS-mounted site-packages) while ``len(sys.modules)`` is
a property of the code. Eager optional-dependency probes in ``__init__.py`` once
made this 1175 modules / 1.35 s; the cap below leaves generous headroom over the
stdlib-only floor so it tracks a regression in kind, not a fluctuation in
degree.

**Public surface.** ``__getattr__`` without ``__dir__`` violates PEP 562:
``dir()`` stops reporting the lazily resolved names (``"main" in dir(Auto3D)``
was False) while the module namespace still exposes whatever the module body
happened to import. Both halves are asserted -- the public names are visible,
and the import-machinery names are not.

Every cost measurement runs in a **subprocess**. It cannot be done in-process:
``conftest.py`` deliberately imports every ``Auto3D`` submodule before the first
test runs (see ``_import_every_auto3d_module_before_any_test``), so by the time
any assertion here executes, ``sys.modules`` already holds torch, rdkit, and all
of Auto3D. An in-process version of these tests would pass unconditionally.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

# Generous cap over the stdlib-only floor (~140 modules: interpreter startup
# plus importlib.metadata's own imports). Set well above the floor and well
# below the pre-fix 1175 so it survives a stdlib growing an internal import or
# a site-packages .pth adding one, while still going red the moment
# ``Auto3D/__init__.py`` reaches for torch, rdkit, or a domain submodule.
MAX_MODULES_AFTER_BARE_IMPORT = 250

# Names that leaked out of the package root as attributes without ever being
# part of the public API: the ``import warnings`` and ``from
# importlib.metadata import PackageNotFoundError, version`` used to compute
# ``__version__``, and the optional-dependency probes' bindings.
LEAK_CANDIDATES = (
    "warnings",
    "version",
    "PackageNotFoundError",
    "ANI2xt",
    "oechem",
    "oeomega",
    "oequacpac",
    "torchani",
)

# Runs in a fresh interpreter; reports what a bare ``import Auto3D`` cost and
# what it left reachable. Emits one JSON line on stdout.
_PROBE_SOURCE = """
import json, sys, time

t0 = time.perf_counter()
import Auto3D
elapsed = time.perf_counter() - t0

leak_candidates = %(leaks)r
print(json.dumps({
    "elapsed": elapsed,
    "n_modules": len(sys.modules),
    "auto3d_modules": sorted(
        m for m in sys.modules if m == "Auto3D" or m.startswith("Auto3D.")
    ),
    "torch": any(m == "torch" or m.startswith("torch.") for m in sys.modules),
    "rdkit": any(m == "rdkit" or m.startswith("rdkit.") for m in sys.modules),
    "version": Auto3D.__version__,
    "dir": sorted(dir(Auto3D)),
    "all": list(Auto3D.__all__),
    "reachable_leaks": sorted(n for n in leak_candidates if hasattr(Auto3D, n)),
}))
""" % {"leaks": LEAK_CANDIDATES}


@pytest.fixture(scope="module")
def bare_import():
    """Measure a cold ``import Auto3D`` in a fresh interpreter, once."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SOURCE],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Import cost
# --------------------------------------------------------------------------- #

def test_bare_import_does_not_load_torch_or_rdkit(bare_import):
    """The two heavyweight third-party dependencies stay unimported.

    They are what makes the difference between 0.02 s and 1.35 s. Anything that
    needs them imports them itself, at the point of need.
    """
    assert not bare_import["torch"], "import Auto3D pulled in torch"
    assert not bare_import["rdkit"], "import Auto3D pulled in rdkit"


def test_bare_import_loads_no_auto3d_submodule(bare_import):
    """Only the package root itself is imported.

    This is the sharp form of the cost assertion: it names the defect (the
    package root executing a domain submodule's module body) rather than a
    number. ``_LAZY_API`` exists precisely so that no submodule loads until a
    public name is used.
    """
    assert bare_import["auto3d_modules"] == ["Auto3D"], (
        "import Auto3D eagerly imported submodules: "
        f"{[m for m in bare_import['auto3d_modules'] if m != 'Auto3D']}"
    )


def test_bare_import_module_count_stays_under_cap(bare_import):
    """``len(sys.modules)`` after a bare import stays near the stdlib floor."""
    assert bare_import["n_modules"] < MAX_MODULES_AFTER_BARE_IMPORT, (
        f"import Auto3D loaded {bare_import['n_modules']} modules "
        f"(cap {MAX_MODULES_AFTER_BARE_IMPORT}, took "
        f"{bare_import['elapsed']:.3f}s)"
    )


def test_bare_import_still_reports_a_version(bare_import):
    """Cutting the cost must not cost ``__version__``.

    ``__version__`` is computed by a private helper so that ``version`` and
    ``PackageNotFoundError`` do not land in the module namespace; this checks
    the helper still runs, in a fresh interpreter, outside pytest.
    """
    assert isinstance(bare_import["version"], str)
    assert bare_import["version"]


# --------------------------------------------------------------------------- #
# Public surface: __dir__ (PEP 562)
# --------------------------------------------------------------------------- #

def test_dir_reports_the_public_api():
    """Every ``__all__`` name is visible to ``dir()`` and to tab-completion."""
    import Auto3D

    listed = set(dir(Auto3D))
    assert "main" in listed
    missing = sorted(set(Auto3D.__all__) - listed)
    assert not missing, f"__all__ names absent from dir(Auto3D): {missing}"


def test_dir_does_not_leak_import_machinery(bare_import):
    """The package root exposes no name it merely needed in order to load.

    Checked in both directions and in both processes: not in ``dir()`` here,
    and not reachable via ``hasattr`` in a fresh interpreter. ``dir()`` alone is
    not enough -- a name can be absent from ``dir()`` and still resolve -- and
    ``hasattr`` alone is not enough in this process, because ``conftest``'s
    eager submodule import legitimately adds submodule attributes.
    """
    import Auto3D

    listed = set(dir(Auto3D))
    leaked_here = sorted(n for n in LEAK_CANDIDATES if n in listed)
    assert not leaked_here, f"dir(Auto3D) leaks import machinery: {leaked_here}"
    assert not bare_import["reachable_leaks"], (
        "package root exposes non-public attributes: "
        f"{bare_import['reachable_leaks']}"
    )


# --------------------------------------------------------------------------- #
# Public surface: _LAZY_API
# --------------------------------------------------------------------------- #

def test_lazy_api_is_a_bijection_with_all():
    """``_LAZY_API`` and ``__all__`` describe the same surface.

    Without this, a name can be added to ``__all__`` and be unreachable, or
    added to ``_LAZY_API`` and be undocumented.
    """
    import Auto3D

    assert set(Auto3D._LAZY_API) == set(Auto3D.__all__) - {"__version__"}


def test_every_lazy_api_name_resolves_to_its_target():
    """Each lazy name resolves to the exact object at its declared location.

    Stronger than "is not None": it pins the target, so moving a function
    without updating ``_LAZY_API`` fails here instead of silently exporting
    something else of the same name.
    """
    import Auto3D

    for name, (module_name, attr) in Auto3D._LAZY_API.items():
        target = getattr(importlib.import_module(module_name), attr)
        assert getattr(Auto3D, name) is target, name


def test_getattr_does_not_cache_resolved_attributes():
    """Repeated access must re-read the source module, not return a snapshot.

    Caching into ``globals()`` looks like a free optimization and is not: it
    turns ``__getattr__`` into an import-time binding, so a test that touches
    ``Auto3D.main`` and *then* patches ``Auto3D.auto3D.main`` would patch
    nothing and pass for the wrong reason -- the failure mode
    ``tests/test_lazy_torchani_import.py`` documents, which surfaced 182 tests
    downstream of its cause. After the first access ``import_module`` is a
    ``sys.modules`` dict hit, so there is nothing to buy.

    Verified behaviorally (a post-access change to the source module is seen)
    and structurally (the name never lands in the package namespace).
    """
    import Auto3D

    first = Auto3D.Auto3DOptions
    assert first is not None
    assert "Auto3DOptions" not in vars(Auto3D), (
        "__getattr__ cached into the module namespace"
    )

    sentinel = object()
    module = importlib.import_module("Auto3D.config")
    original = module.Auto3DOptions
    try:
        module.Auto3DOptions = sentinel
        assert Auto3D.Auto3DOptions is sentinel, (
            "second access returned a cached value instead of re-reading "
            "Auto3D.config"
        )
    finally:
        module.Auto3DOptions = original

    assert Auto3D.Auto3DOptions is original


def test_unknown_attribute_raises_attribute_error():
    """``__getattr__`` must not turn a typo into an ImportError or a hang."""
    import Auto3D

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        getattr(Auto3D, "nope")  # noqa: B009  (the lookup is the assertion)
