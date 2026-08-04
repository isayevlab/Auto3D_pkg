"""Import boundaries: what ``import Auto3D`` is allowed to cost, what is public,
and which packages are allowed to re-export.

Four properties are locked here. The last three are static (AST) checks over
``src/Auto3D/**`` plus one subprocess probe; only the first needs the package
imported.

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

**The public-surface rule.** ``docs/source/api.rst`` is the definition of what
is public, at the dotted paths it lists; ``Auto3D.__all__`` is the top-level
convenience barrel and may export only what api.rst documents. Both directions
are checked, so the rule is mechanical rather than a preference. See the section
comment above ``test_api_rst_entries_resolve_at_their_documented_path`` for why
the implication runs one way only.

**No package re-exports anything.** ``Auto3D/utils/__init__.py`` was a 41-name
barrel assembled from five of its eight submodules, so three of them had no way
in, and the same function was imported through the barrel in one module and
through its defining module in a sibling. ``batch_opt/batchopt.py`` was a
narrower version of the same thing. Deleting a barrel needs both halves asserted
-- the names are gone, *and* nobody imports through it -- because removing only
the consumers leaves a live barrel for the next contributor, and removing only
the barrel is caught by the type checker at best.

Every cost measurement runs in a **subprocess**. It cannot be done in-process:
``conftest.py`` deliberately imports every ``Auto3D`` submodule before the first
test runs (see ``_import_every_auto3d_module_before_any_test``), so by the time
any assertion here executes, ``sys.modules`` already holds torch, rdkit, and all
of Auto3D. An in-process version of these tests would pass unconditionally.
"""
from __future__ import annotations

import ast
import importlib
import json
import pathlib
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


# --------------------------------------------------------------------------- #
# The public-surface rule
# --------------------------------------------------------------------------- #
#
# A name is public iff ``docs/source/api.rst`` documents it, at the dotted path
# api.rst gives, and that path resolves. ``Auto3D.__all__`` is a *second*,
# narrower thing: the top-level convenience barrel, and the only barrel in the
# package. Every name in it must also be documented in api.rst, so nothing is
# exported that is not documented.
#
# The rule is deliberately **not** "api.rst and ``__all__`` hold the same
# names". api.rst documents 24 entries, ten of which are the exception classes
# (``Auto3D.exceptions.*``); none of those is in ``__all__`` and none should be.
# api.rst documents *dotted paths*, and the rest of the docs consistently
# reference these names that way -- ``Auto3D.model_factory.get_device``,
# ``Auto3D.models.contract.CustomNNP`` (migration-3.0.rst calls that "the
# surviving one", against the removed ``from Auto3D.models import CustomNNP``),
# ``Auto3D.isomers.IsomerEngineFactory``. Promoting those into ``__all__`` would
# mint a *second* supported path for each and contradict the migration guide, so
# the implication runs one way only: exported implies documented.
#
# Corollary, checked by ``test_only_documented_subpackages_define_all``: no
# subpackage ``__init__.py`` re-exports anything, unless its api.rst-documented
# dotted path *is* the package path.

API_RST = pathlib.Path(__file__).resolve().parents[1] / "docs" / "source" / "api.rst"

# ``__version__`` is a string produced by the package itself, not an API object
# autosummary can document.
UNDOCUMENTED_BY_DESIGN = frozenset({"__version__"})


def _api_rst_entries() -> list[str]:
    """Dotted paths listed under api.rst's ``autosummary`` directives."""
    entries = []
    in_block = False
    for raw in API_RST.read_text().splitlines():
        line = raw.strip()
        if line.startswith(".. autosummary::"):
            in_block = True
            continue
        if not in_block:
            continue
        if not line:
            continue
        if line.startswith(":"):  # directive option, e.g. :toctree:
            continue
        if line.startswith("Auto3D."):
            entries.append(line)
        else:  # any other non-blank, non-indented content ends the block
            in_block = False
    return entries


def _resolve_dotted_path(path: str):
    """Import the longest importable prefix of ``path``, then getattr the rest.

    Written this way so a package-path entry (``Auto3D.isomers.IsomerEngineFactory``,
    ``Auto3D.generate_conformers``) resolves by the same rule as a module-path
    entry (``Auto3D.auto3D.main``), instead of the test hard-coding which is which.
    """
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise AssertionError(f"no importable module prefix in {path!r}")


def test_api_rst_entries_resolve_at_their_documented_path():
    """Every documented dotted path is real.

    This is what makes the rule checkable rather than aspirational: api.rst is
    the definition of the public surface, so a documented path that no longer
    resolves is a broken promise, and a sphinx build is too slow to be the only
    place that notices.
    """
    entries = _api_rst_entries()
    assert len(entries) > 20, f"api.rst parse looks wrong: {entries}"
    broken = []
    for path in entries:
        try:
            _resolve_dotted_path(path)
        except (AssertionError, AttributeError, ImportError) as exc:
            broken.append(f"{path}: {type(exc).__name__}: {exc}")
    assert not broken, "api.rst documents paths that do not resolve:\n" + "\n".join(broken)


def test_every_exported_name_is_documented_in_api_rst():
    """``Auto3D.__all__`` may not export an undocumented name.

    The top-level barrel is the one place a name can be reached without knowing
    which module defines it, so it is the one place an undocumented export is
    invisible. Note the implication is one-way -- api.rst documents plenty that
    ``__all__`` does not export (see the section comment above).
    """
    import Auto3D

    documented_leaves = {path.rsplit(".", 1)[1] for path in _api_rst_entries()}
    undocumented = sorted(
        set(Auto3D.__all__) - documented_leaves - UNDOCUMENTED_BY_DESIGN
    )
    assert not undocumented, (
        "names exported from Auto3D.__all__ but absent from docs/source/api.rst: "
        f"{undocumented}"
    )


# --------------------------------------------------------------------------- #
# Package barrels: static (AST) enforcement
# --------------------------------------------------------------------------- #
#
# Static, and specifically AST rather than a regex over the text. A regex
# matches the historical ``from Auto3D.utils import ...`` snippets that live on
# purpose in ``docs/plans/**`` and in ``docs/source/migration-3.0.rst`` (where
# the barrel import is the deliberate *before* half of a before/after pair), so
# a regex-based check could be "satisfied" by falsifying the migration guide.
# Everything below is scoped to ``src/Auto3D/**`` only.

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "Auto3D"

# ``from Auto3D.utils import energy`` -- naming a submodule -- stays legal;
# it is a module reference, not a re-export. ``from Auto3D.utils import
# hartree2ev`` does not.
UTILS_SUBMODULES = frozenset(
    p.stem for p in (SRC_ROOT / "utils").glob("*.py") if p.stem != "__init__"
)

# Names ``batch_opt/batchopt.py`` re-exported for backward compatibility. Two
# of them (``EnForce_ANI``, ``n_steps``) the module genuinely uses, so they
# still resolve there; ``batchopt`` is nonetheless not their home and no
# first-party module may reach them through it.
BATCHOPT_REEXPORTS = frozenset({"EnForce_ANI", "n_steps", "print_stats"})

# The only subpackage whose api.rst-documented dotted path *is* the package
# path, and therefore the only one allowed to re-export. ``cli`` and ``models``
# still carry an ``__all__`` and are known remaining barrels: neither is
# documented at its package path (api.rst documents
# ``Auto3D.models.contract.CustomNNP``, and no ``Auto3D.cli`` name at all), so
# both are debt this set records rather than blesses. Removing one means
# deleting it from here, not adding it.
SUBPACKAGES_WITH_ALL = frozenset({"isomers", "cli", "models"})


def _source_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _absolute_module(node: ast.ImportFrom, path: pathlib.Path) -> str | None:
    """Absolute module name an ``ImportFrom`` names, resolving relative levels."""
    if not node.level:
        return node.module
    package = path.parent.relative_to(SRC_ROOT.parent).parts
    if node.level > 1:
        package = package[: -(node.level - 1)]
    return ".".join((*package, node.module)) if node.module else ".".join(package)


def test_no_src_module_imports_through_the_utils_barrel():
    """No first-party module pulls a *name* out of ``Auto3D.utils``.

    The barrel was never a coherent surface -- three of its eight submodules
    (``energy``, ``convergence``, ``stereo_check``) had no presence in it at all
    -- and ``check_connectivity`` was reached through the barrel in
    ``filtering.py`` and through ``utils.chemistry`` in ``ranking.py`` (both now
    name ``utils.connectivity``), two
    sibling modules disagreeing about the same function with nothing saying
    which was right.
    """
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if _absolute_module(node, path) != "Auto3D.utils":
                continue
            names = [a.name for a in node.names if a.name not in UTILS_SUBMODULES]
            if names:
                offenders.append(f"{path.relative_to(SRC_ROOT.parent)}:{node.lineno}: {names}")
    assert not offenders, (
        "imports through the Auto3D.utils barrel (import from the defining "
        "module instead):\n" + "\n".join(offenders)
    )


def test_utils_init_imports_nothing():
    """``utils/__init__.py`` is docstring-only.

    Frozen deliberately, and not merely as the tail of the demolition: a later
    cluster split ``utils/file_ops.py`` and moved modules underneath this
    package. Its own proposal was to "keep ``file_ops.py`` as a re-export shim
    so ``utils/__init__.py`` is untouched", which would rebuild the barrel one
    directory down. This test makes that fail here instead of being noticed
    later, or not at all.
    """
    path = SRC_ROOT / "utils" / "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = [
        f"line {node.lineno}"
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not imports, f"utils/__init__.py imports something: {imports}"
    non_docstring = [
        type(node).__name__ for node in tree.body if not _is_docstring(node)
    ]
    assert not non_docstring, (
        f"utils/__init__.py has statements beyond its docstring: {non_docstring}"
    )


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
        and isinstance(node.value.value, str)


def test_utils_package_exposes_no_names():
    """The demolition's other half: the names are actually gone at runtime.

    The AST test proves nobody in ``src/`` reaches through the barrel; this
    proves the barrel is not there to reach through. Both halves are needed --
    deleting only the consumers leaves a live barrel for the next contributor
    to use.
    """
    utils = importlib.import_module("Auto3D.utils")
    assert not hasattr(utils, "__all__"), (
        f"Auto3D.utils still declares __all__: {getattr(utils, '__all__', None)}"
    )
    # The ``from ... import name`` form, because that is what consumers wrote
    # and it is the form that raises ImportError (a bare getattr would raise
    # AttributeError, a weaker and differently-worded failure).
    for name in ("check_input", "check_connectivity", "hartree2ev", "reorder_sdf"):
        with pytest.raises(ImportError):
            exec(f"from Auto3D.utils import {name}", {})  # noqa: S102


def test_utils_submodule_imports_still_work():
    """Emptying the barrel must not break ``from Auto3D.utils import energy``.

    A submodule reference is not a re-export, several tests use this form, and
    ``__init__.py`` importing nothing is exactly the condition under which it is
    easy to assume otherwise.
    """
    from Auto3D.utils import energy, validation

    assert energy.hartree2ev > 0  # a float constant, not a function
    assert callable(validation.check_input)


def test_isomer_engine_does_not_import_the_isomers_package():
    """``Auto3D.isomers`` wraps ``isomer_engine``; the arrow may not point back.

    ``isomers.factory`` imports ``Auto3D.isomer_engine`` at module scope, so a
    single import in the other direction closes a cycle. Until 4.0 that cycle
    existed: the two adapter modules and ``factory.create_tautomer_engine``
    reached into ``isomer_engine``, and ``isomer_engine._run_parallel_embedding``
    reached back into ``isomers.parallel_embed``. It stayed latent only because
    every edge was a function-scope import -- which is exactly the shape that
    surfaces as an ``ImportError`` inside a ``spawn``ed worker and nowhere else,
    since a spawned child re-imports from scratch in an order the parent never
    exercised.

    Checked at **any** scope (``ast.walk``, not ``tree.body``) for that reason:
    a function-scope import here would satisfy a module-scope-only check while
    reintroducing precisely the latent cycle that was removed. ``parallel_embed``
    is now ``Auto3D.embedding``, which imports nothing from either side.
    """
    path = SRC_ROOT / "isomer_engine.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, path) or ""
            if module == "Auto3D.isomers" or module.startswith("Auto3D.isomers."):
                offenders.append(f"line {node.lineno}: from {module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "Auto3D.isomers" or alias.name.startswith(
                    "Auto3D.isomers."
                ):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
    assert not offenders, (
        "isomer_engine.py imports from the Auto3D.isomers package it is wrapped "
        "by, closing an import cycle:\n" + "\n".join(offenders)
    )


def test_only_documented_subpackages_define_all():
    """A subpackage may re-export only if api.rst documents it at that path."""
    offenders = {}
    for path in _source_files():
        if path.name != "__init__.py" or path.parent == SRC_ROOT:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        declares_all = any(
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "__all__" for t in node.targets)
            for node in tree.body
        )
        if declares_all:
            offenders[path.parent.name] = str(path.relative_to(SRC_ROOT.parent))
    assert set(offenders) == SUBPACKAGES_WITH_ALL, (
        "subpackages declaring __all__ changed; expected "
        f"{sorted(SUBPACKAGES_WITH_ALL)}, found {sorted(offenders)}"
    )


def test_no_src_module_imports_batchopt_reexports():
    """``batchopt`` is not the home of ``EnForce_ANI``/``n_steps``/``print_stats``.

    ``batchopt.py`` itself still *uses* the first two, so they resolve there and
    third-party code does not break today -- but a first-party import through
    that path is what kept the compat barrel alive and made the module look like
    the owner of a class defined in ``model_wrapper``.
    """
    offenders = []
    for path in _source_files():
        if path == SRC_ROOT / "batch_opt" / "batchopt.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if _absolute_module(node, path) != "Auto3D.batch_opt.batchopt":
                continue
            names = [a.name for a in node.names if a.name in BATCHOPT_REEXPORTS]
            if names:
                offenders.append(f"{path.relative_to(SRC_ROOT.parent)}:{node.lineno}: {names}")
    assert not offenders, (
        "imports of batchopt's compat re-exports:\n" + "\n".join(offenders)
    )


def test_batchopt_does_not_re_export_print_stats():
    """The one name ``batchopt`` re-exported without using it is gone."""
    batchopt = importlib.import_module("Auto3D.batch_opt.batchopt")
    assert not hasattr(batchopt, "print_stats"), (
        "batchopt still re-exports print_stats; its home is "
        "Auto3D.batch_opt.optimization_engine"
    )


# --------------------------------------------------------------------------- #
# utils/ is a leaf
# --------------------------------------------------------------------------- #

# Subprocess, for the same reason as the cost tests above: conftest imports
# every Auto3D module before any test runs, so in-process this would pass
# unconditionally.
_LEAF_PROBE_SOURCE = """
import json, sys
import Auto3D.utils.validation  # noqa: F401
print(json.dumps(sorted(
    m for m in sys.modules if m == "Auto3D.models" or m.startswith("Auto3D.models.")
)))
"""


def test_importing_utils_validation_does_not_load_models():
    """``utils/`` must not depend on the ``models/`` domain package.

    ``utils`` is a leaf by intent -- generic helpers every layer may use -- and
    ``validation.py`` was the single module breaking that, with two module-level
    imports of ``Auto3D.models.*`` whose only consumers are function-scope. A
    domain package reached from the bottom of the stack is what let one probe in
    ``Auto3D/__init__.py`` drag in the entire model layer.

    Scoped to ``Auto3D.models`` on purpose: ``validation.py`` legitimately
    imports torch and rdkit at module scope, and must keep doing so -- twenty-two
    test sites patch ``Auto3D.utils.validation.torch.cuda.is_available``, which
    requires ``torch`` to be a real attribute of this module's namespace.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _LEAF_PROBE_SOURCE],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not loaded, (
        f"importing Auto3D.utils.validation pulled in the models package: {loaded}"
    )


# Subprocess probe: what the split-out file-I/O modules cost to import.
_FILE_IO_PROBE_SOURCE = """
import json, sys
import Auto3D.id_mapping            # noqa: F401
import Auto3D.job_layout            # noqa: F401
import Auto3D.utils.output_guard    # noqa: F401
import Auto3D.utils.reconciliation  # noqa: F401
import Auto3D.utils.sdf_io          # noqa: F401
import Auto3D.utils.smi_io          # noqa: F401
print(json.dumps({
    "torch": any(m == "torch" or m.startswith("torch.") for m in sys.modules),
    "models": sorted(
        m for m in sys.modules if m == "Auto3D.models" or m.startswith("Auto3D.models.")
    ),
}))
"""


def test_file_io_modules_do_not_load_torch_or_the_model_tree():
    """Writing a ``.smi``/``.sdf`` file must not cost the neural-network stack.

    The six modules probed here are what ``utils/file_ops.py`` split into, and
    the reason ``check_output_overwrite``/``check_output_not_input`` were lifted
    out of ``utils/validation.py`` into the leaf ``utils/output_guard.py``
    first: the overwrite gate belongs on every one of these writers, and
    reaching it through ``validation`` would have pulled in that module's
    module-scope ``torch`` -- and, through the engine-name resolution it does
    at function scope, the whole ``Auto3D.models`` tree -- for a caller that
    only wanted to refuse clobbering a file.

    Subprocess, for the same reason as the probes above: ``conftest`` imports
    every ``Auto3D`` submodule before any test runs, so in-process this would
    pass unconditionally.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _FILE_IO_PROBE_SOURCE],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not result["torch"], "importing the .smi/.sdf writers pulled in torch"
    assert not result["models"], (
        "importing the .smi/.sdf writers pulled in the models package: "
        f"{result['models']}"
    )


def test_validation_imports_torch_at_module_scope():
    """The companion constraint, pinned so the leaf fix cannot overreach.

    ``torch`` is a third-party leaf, not a domain package, and twenty-two test
    sites patch ``Auto3D.utils.validation.torch.cuda.is_available``. Deferring
    this import would break every one of them at once, and the leaf test above
    would still pass -- so the prohibition is stated here rather than left to be
    inferred.
    """
    validation = importlib.import_module("Auto3D.utils.validation")
    assert hasattr(validation, "torch"), (
        "utils/validation.py no longer imports torch at module scope; "
        "monkeypatches targeting Auto3D.utils.validation.torch.* will silently "
        "target nothing"
    )
