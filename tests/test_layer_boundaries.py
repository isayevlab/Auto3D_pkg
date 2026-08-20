"""The dependency direction between layers, checked mechanically.

The package is already acyclic and close to correctly layered, but nothing says
so: the boundaries are held by docstrings, by convention, and in one case by a
deliberately deferred import. That is enough while the layout is stable and not
enough while modules are moving between packages, which is what the next several
items of the modernization plan do. A reviewer looking at a diff that relocates
six modules cannot tell a correct move from an incorrect one by reading it.

The layer of a module is now its **directory**, so this file no longer declares
which modules live where -- it reads the tree:

    L5  presentation/   cli/**, auto3Dcli, and the package root
    L4  entry/          auto3D, SPE, ASE/**, tautomer
    L3  orchestration/  workflow, workflow_workers, chunk_manager, job_layout,
                        processors, pipeline/**
    L2  engines/        models/**, model_factory, isomers/**, batch_opt/**
    L1  domain/         ranking, filtering, embedding, clash_relief, id_mapping
    L0  foundation/     config, constants, exceptions, registry, results,
                        torch_config, utils/**

The only thing declared below is the ORDER of the layers, which is the one fact
a directory name cannot carry. A per-module prefix list used to sit here and had
to be edited every time a file moved -- and could go stale in both directions:
a module in no layer (caught) and a layer prefix naming no module (not caught
until a staleness test was added for it). Neither failure is expressible now.

Two scopes, because they answer different questions. `test_no_module_imports_a_
higher_layer` reads module-scope imports only: those are what an import of the
package actually costs and what a circular-import crash is made of. Function-scope
imports are the sanctioned escape hatch today, and the second test reads them too,
so that a boundary held only by deferring an import is recorded here rather than
being invisible to the check that is supposed to find it.
"""

from __future__ import annotations

import ast
import collections
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
PKG = SRC / "Auto3D"

#: Layer number -> the directory that *is* that layer. The order is the whole
#: declaration: which modules belong to a layer is read from where they sit.
#:
#: The package root (``Auto3D/__init__.py``) is presentation: it is the barrel a
#: user imports, and it may reach anything below it.
LAYER_DIRS: dict[int, str] = {
    5: "presentation",
    4: "entry",
    3: "orchestration",
    2: "engines",
    1: "domain",
    0: "foundation",
}

_OWNER = {directory: layer for layer, directory in LAYER_DIRS.items()}

#: Upward module-scope edges that are known, named and dated. An entry here is a
#: debt with an owner, not a permission: each says what removes it.
#:
#: Empty, and the way the last entry left is worth recording. It read:
#: "tautomer.py runs the whole pipeline to score tautomers, so an engine-layer
#: module imports the entry layer. Removed by the plan's item 8, which makes the
#: pipeline a composable object that tautomer can hold instead of a function it
#: has to call top-down."
#:
#: That remedy would not have worked. A pipeline object belongs in L3, and
#: L2 -> L3 is still upward, so inverting the dependency would have moved the
#: violation rather than removed it. The edge was not a wiring problem at all:
#: ``tautomer`` was misfiled (see LAYERS above), and reclassifying it to L4 makes
#: ``tautomer -> auto3D`` a same-layer edge, which this map has always allowed.
UPWARD_EXEMPTIONS: dict[tuple[str, str], str] = {}

#: Mutually-importing package pairs, counting function-scope imports too.
#:
#: Empty, and it should stay that way. It held one entry when this file was
#: written -- ``batch_opt`` <-> ``models``, where ``models/adapter.py`` reached
#: back into ``batch_opt.ANI2xt_no_rep`` from inside a method, which was the only
#: reason that pair was not an import cycle. The plan's item 2 moved that module
#: to ``models/ani2xt.py``, beside the weights it loads, and this test reported
#: the exemption as stale on the next run.
CYCLE_EXEMPTIONS: dict[tuple[str, str], str] = {}


def _module_name(path: pathlib.Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _layer_of(module: str) -> int | None:
    """The layer a module lives in, read off its first path component.

    ``Auto3D`` itself (the package root) is presentation. Anything whose first
    component is not a layer directory returns ``None``, which
    :func:`test_every_module_lives_in_a_layer_directory` turns into a failure --
    that is the only way a module can now escape classification.
    """
    rest = module[len("Auto3D") :].lstrip(".")
    if not rest:
        return _OWNER["presentation"]
    return _OWNER.get(rest.split(".")[0])


def _modules() -> dict[str, pathlib.Path]:
    return {_module_name(p): p for p in sorted(PKG.rglob("*.py")) if "__pycache__" not in p.parts}


def _internal_edges(*, module_scope_only: bool) -> dict[str, set[str]]:
    """`{importer: {imported, ...}}` over Auto3D-internal imports."""
    modules = _modules()
    edges: dict[str, set[str]] = collections.defaultdict(set)

    for module, path in modules.items():
        tree = ast.parse(path.read_text(), filename=str(path))
        nodes = tree.body if module_scope_only else ast.walk(tree)
        for node in nodes:
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.rsplit(".", node.level)[0]
                    targets = [f"{base}.{node.module}" if node.module else base]
                elif node.module:
                    targets = [node.module]
            for target in targets:
                if target != "Auto3D" and not target.startswith("Auto3D."):
                    continue
                # `from Auto3D.foundation.utils.energy import x` may name a module or an
                # attribute of one; walk up to the nearest real module.
                while target and target not in modules:
                    target = target.rsplit(".", 1)[0] if "." in target else ""
                if target and target != module:
                    edges[module].add(target)
    return edges


def _package_of(module: str) -> str:
    rest = module[len("Auto3D") :].lstrip(".")
    if not rest:
        return "Auto3D"
    head = rest.split(".")[0]
    return f"Auto3D.{head}" if (PKG / head).is_dir() else module


def test_every_module_is_assigned_to_a_layer():
    """The map must cover the package, or the checks below silently shrink.

    A module nobody classified is a module no boundary applies to, and adding a
    file is exactly when that happens. This is the test that makes the map a
    decision rather than a list someone stopped updating.
    """
    unmapped = sorted(m for m in _modules() if _layer_of(m) is None)
    assert not unmapped, (
        "these modules are not inside a layer directory -- move them under one "
        "of {sorted(LAYER_DIRS.values())}:\n  " + "\n  ".join(unmapped)
    )


def test_every_layer_directory_exists():
    """The map must not outlive the tree it describes.

    The totality test above is one-directional: it catches a module outside every
    layer, never a layer naming nothing. When layers were prefix *lists* that
    asymmetry let dead entries accumulate silently -- ``isomer_engine`` sat in the
    map after PR #168 deleted it, classifying nothing.

    Naming directories instead shrinks the failure to one case, which is this
    one: a layer renamed on disk and not here. There is no longer any way for a
    layer to be partly stale.
    """
    missing = sorted(d for d in LAYER_DIRS.values() if not (PKG / d).is_dir())
    assert not missing, f"these LAYER_DIRS name no directory under {PKG}: {missing}"


def test_no_module_imports_a_higher_layer():
    """Dependencies point down, at module scope.

    Module-scope imports are the ones that decide what importing the package
    costs and the ones a circular import is built from, so they are the strict
    half of the rule.
    """
    edges = _internal_edges(module_scope_only=True)
    unexpected, stale = [], set(UPWARD_EXEMPTIONS)

    for importer in sorted(edges):
        for imported in sorted(edges[importer]):
            high, low = _layer_of(importer), _layer_of(imported)
            if high is None or low is None or low <= high:
                continue
            stale.discard((importer, imported))
            if (importer, imported) not in UPWARD_EXEMPTIONS:
                unexpected.append(f"L{high} {importer} -> L{low} {imported}")

    assert not unexpected, (
        "a lower layer imports a higher one; move the code or add a dated entry "
        "to UPWARD_EXEMPTIONS saying what removes it:\n  " + "\n  ".join(unexpected)
    )
    assert not stale, (
        "these UPWARD_EXEMPTIONS no longer correspond to a real edge -- delete "
        f"them: {sorted(stale)}"
    )


def test_no_two_packages_import_each_other():
    """No package pair imports both ways, counting deferred imports.

    Reading function-scope imports here is the point. A deferred import keeps a
    cycle from being an import-time crash, but the dependency is still there, and
    a check that could not see it would report a clean structure that the next
    refactor discovers is not.
    """
    edges = _internal_edges(module_scope_only=False)
    pairs: dict[tuple[str, str], list[str]] = collections.defaultdict(list)

    for importer in sorted(edges):
        for imported in sorted(edges[importer]):
            a, b = _package_of(importer), _package_of(imported)
            if a != b:
                pairs[(a, b)].append(f"{importer} -> {imported}")

    unexpected, stale = [], set(CYCLE_EXEMPTIONS)
    for (a, b), examples in sorted(pairs.items()):
        if (b, a) not in pairs or a > b:
            continue  # `a > b` reports each pair once, not once per direction
        stale.discard((a, b))
        if (a, b) not in CYCLE_EXEMPTIONS:
            unexpected.append(f"{a} <-> {b}: {examples[0]}, and {pairs[(b, a)][0]}")

    assert not unexpected, (
        "these packages import each other; break the cycle or add a dated entry "
        "to CYCLE_EXEMPTIONS:\n  " + "\n  ".join(sorted(set(unexpected)))
    )
    assert not stale, (
        f"these CYCLE_EXEMPTIONS no longer correspond to a real cycle: {sorted(stale)}"
    )
